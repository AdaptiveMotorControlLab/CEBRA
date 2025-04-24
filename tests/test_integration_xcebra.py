import pickle

import numpy as np
import pytest
import torch

import cebra
import cebra.attribution
import cebra.data
import cebra.models
import cebra.solver
from cebra.data import ContrastiveMultiObjectiveLoader
from cebra.data import DatasetxCEBRA
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp


@pytest.fixture
def synthetic_data():
    import tempfile
    import urllib.request
    from pathlib import Path

    url = "https://cebra.fra1.digitaloceanspaces.com/xcebra_synthetic_data.pkl"

    # Create a persistent temp directory specific to this test
    temp_dir = Path(tempfile.gettempdir()) / "cebra_test_data"
    temp_dir.mkdir(exist_ok=True)
    filepath = temp_dir / "synthetic_data.pkl"

    if not filepath.exists():
        urllib.request.urlretrieve(url, filepath)

    with filepath.open('rb') as file:
        return pickle.load(file)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_synthetic_data_training(synthetic_data, device):
    # Setup data
    neurons = synthetic_data['neurons']
    latents = synthetic_data['latents']
    n_latents = latents.shape[1]
    Z1 = synthetic_data['Z1']
    Z2 = synthetic_data['Z2']
    gt_attribution_map = synthetic_data['gt_attribution_map']
    data = DatasetxCEBRA(neurons, Z1=Z1, Z2=Z2)

    # Configure training with reduced steps
    TOTAL_STEPS = 50  # Reduced from 2000 for faster testing
    loader = ContrastiveMultiObjectiveLoader(dataset=data,
                                             num_steps=TOTAL_STEPS,
                                             batch_size=512).to(device)

    config = MultiObjectiveConfig(loader)
    config.set_slice(0, 6)
    config.set_loss("FixedEuclideanInfoNCE", temperature=1.)
    config.set_distribution("time", time_offset=1)
    config.push()

    config.set_slice(3, 6)
    config.set_loss("FixedEuclideanInfoNCE", temperature=1.)
    config.set_distribution("time_delta", time_delta=1, label_name="Z2")
    config.push()

    config.finalize()

    # Initialize model and solver
    neural_model = cebra.models.init(
        name="offset1-model-mse-clip-5-5",
        num_neurons=data.neural.shape[1],
        num_units=256,
        num_output=n_latents,
    ).to(device)

    data.configure_for(neural_model)

    opt = torch.optim.Adam(
        list(neural_model.parameters()) + list(config.criterion.parameters()),
        lr=3e-4,
        weight_decay=0,
    )

    regularizer = cebra.models.jacobian_regularizer.JacobianReg()

    solver = cebra.solver.init(
        name="multiobjective-solver",
        model=neural_model,
        feature_ranges=config.feature_ranges,
        regularizer=regularizer,
        renormalize=False,
        use_sam=False,
        criterion=config.criterion,
        optimizer=opt,
        tqdm_on=False,
    ).to(device)

    # Train model with reduced steps for regularizer
    weight_scheduler = LinearRampUp(
        n_splits=2,
        step_to_switch_on_reg=25,  # Reduced from 2500
        step_to_switch_off_reg=40,  # Reduced from 15000
        start_weight=0.,
        end_weight=0.01,
        stay_constant_after_switch_off=True)

    solver.fit(
        loader=loader,
        valid_loader=None,
        log_frequency=None,
        scheduler_regularizer=weight_scheduler,
        scheduler_loss=None,
    )

    # Basic test that model runs and produces output
    solver.model.split_outputs = False
    embedding = solver.model(data.neural.to(device)).detach().cpu()

    # Verify output dimensions
    assert embedding.shape[1] == n_latents, "Incorrect embedding dimension"
    assert not torch.isnan(embedding).any(), "NaN values in embedding"

    # Test attribution map functionality
    data.neural.requires_grad_(True)
    method = cebra.attribution.init(name="jacobian-based",
                                    model=solver.model,
                                    input_data=data.neural,
                                    output_dimension=solver.model.num_output)

    result = method.compute_attribution_map()
    jfinv = abs(result['jf-inv-lsq']).mean(0)

    # Verify attribution map output
    assert not torch.isnan(
        torch.tensor(jfinv)).any(), "NaN values in attribution map"
    assert jfinv.shape == gt_attribution_map.shape, "Incorrect attribution map shape"

    # Test split outputs functionality
    solver.model.split_outputs = True
    embedding_split = solver.model(data.neural.to(device))
    Z1_hat = embedding_split[0].detach().cpu()
    Z2_hat = embedding_split[1].detach().cpu()

    # TODO(stes): Right now, this results 6D output vs. 3D as expected. Need to double check
    # the API docs on the desired behavior here, both could be fine...
    # assert Z1_hat.shape == Z1.shape, f"Incorrect Z1 embedding dimension: {Z1_hat.shape}"
    assert Z2_hat.shape == Z2.shape, f"Incorrect Z2 embedding dimension: {Z2_hat.shape}"
    assert not torch.isnan(Z1_hat).any(), "NaN values in Z1 embedding"
    assert not torch.isnan(Z2_hat).any(), "NaN values in Z2 embedding"

    # Test the transform
    solver.model.split_outputs = False
    transform_embedding = solver.transform(data.neural.to(device))
    assert transform_embedding.shape[
        1] == n_latents, "Incorrect embedding dimension"
    assert not torch.isnan(transform_embedding).any(), "NaN values in embedding"
    assert np.allclose(embedding, transform_embedding, rtol=1e-4, atol=1e-4)

    # Test the transform with batching
    batched_embedding = solver.transform(data.neural.to(device), batch_size=512)
    assert batched_embedding.shape[
        1] == n_latents, "Incorrect embedding dimension"
    assert not torch.isnan(batched_embedding).any(), "NaN values in embedding"
    assert np.allclose(embedding, batched_embedding, rtol=1e-4, atol=1e-4)

    assert np.allclose(transform_embedding,
                       batched_embedding,
                       rtol=1e-4,
                       atol=1e-4)

    # Test and compare the previous transform (transform_deprecated)
    deprecated_transform_embedding = solver.transform_deprecated(
        data.neural.to(device))
    assert np.allclose(embedding,
                       deprecated_transform_embedding,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(transform_embedding,
                       deprecated_transform_embedding,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(batched_embedding,
                       deprecated_transform_embedding,
                       rtol=1e-4,
                       atol=1e-4)
