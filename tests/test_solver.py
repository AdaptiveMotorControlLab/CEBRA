#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import copy
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

import cebra.data
import cebra.datasets
import cebra.models
import cebra.solver


# ---- helpers ----


device = "cpu"


def _get_loader(data_name, loader_initfunc):
    data = cebra.datasets.init(data_name)
    kwargs = dict(num_steps=2, batch_size=32)
    loader = loader_initfunc(data, **kwargs)
    return loader, data


OUTPUT_DIMENSION = 3


def _make_model(dataset, model_architecture="offset10-model"):
    # TODO flexible input dimension
    # return nn.Sequential(
    #     nn.Conv1d(dataset.input_dimension, 5, kernel_size=10),
    #     nn.Flatten(start_dim=1, end_dim=-1),
    # )
    return cebra.models.init(model_architecture, dataset.input_dimension, 32,
                             OUTPUT_DIMENSION)


def _make_behavior_model(dataset):
    # TODO flexible input dimension
    return nn.Sequential(
        nn.Conv1d(dataset.input_dimension, 5, kernel_size=10),
        nn.Flatten(start_dim=1, end_dim=-1),
    )


def _assert_same_state_dict(first, second):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], torch.Tensor):
            assert torch.allclose(first[key], second[key]), key
        elif isinstance(first[key], dict):
            _assert_same_state_dict(first[key], second[key]), key
        else:
            assert first[key] == second[key]


def check_if_fit(model):
    """Check if a model was already fit.

    Args:
        model: The model to check.

    Returns:
        True if the model was already fit.
    """
    return hasattr(model, "n_features_")


def _assert_equal(original_solver, loaded_solver):
    for k in original_solver.model.state_dict():
        assert original_solver.model.state_dict()[k].all(
        ) == loaded_solver.model.state_dict()[k].all()
    assert check_if_fit(loaded_solver) == check_if_fit(original_solver)

    if check_if_fit(loaded_solver):
        _assert_same_state_dict(original_solver.state_dict_,
                                loaded_solver.state_dict_)
        X = np.random.normal(0, 1, (100, 1))

        if loaded_solver.num_sessions is not None:
            assert np.allclose(loaded_solver.transform(X, session_id=0),
                               original_solver.transform(X, session_id=0))
        else:
            assert np.allclose(loaded_solver.transform(X),
                               original_solver.transform(X))


def _test_single_session_transform(solver, X, offset):
    embedding = solver.transform(X)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(torch.Tensor(X))
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X, session_id=0)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X, pad_before_transform=False)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (
            (X.shape[0] - len(offset)) // solver.model.resample_factor + 1,
            OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0] - len(offset) + 1,
                                   OUTPUT_DIMENSION)

    with pytest.raises(ValueError, match="torch.Tensor"):
        solver.transform(X.numpy())
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = solver.transform(X, session_id=2)

    for param in solver.parameters():
        assert isinstance(param, torch.Tensor)

    fitted_solver = copy.deepcopy(solver)
    with tempfile.TemporaryDirectory() as temp_dir:
        solver.save(temp_dir)
        solver.load(temp_dir)
    _assert_equal(fitted_solver, solver)


# ---- single-session solvers ----


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [(dataset, model, loader, cebra.solver.SingleSessionSolver)
     for dataset, loader in [("demo-discrete", cebra.data.DiscreteDataLoader),
                             ("demo-continuous", cebra.data.ContinuousDataLoader
                             ), ("demo-mixed", cebra.data.MixedDataLoader)]
     for model in
     ["offset1-model", "offset10-model", "offset40-model-4x-subsample"]])
def test_single_session(data_name, loader_initfunc, model_architecture,
                        solver_initfunc):
    loader, data = _get_loader(data_name, loader_initfunc)
    model = _make_model(data, model_architecture)
    data.configure_for(model)
    offset = model.get_offset()
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer,
                             tqdm_on=False)

    batch = next(iter(loader))
    assert batch.reference.shape[:2] == (32, loader.dataset.input_dimension)
    log = solver.step(batch)
    assert isinstance(log, dict)

    X = loader.dataset.neural
    with pytest.raises(ValueError, match="not.*fitted"):
        solver.transform(X)

    solver.fit(loader)

    assert not hasattr(solver, 'num_sessions')
    assert solver.n_features == X.shape[1]

    _test_single_session_transform(solver, X, offset)


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [(dataset, model, loader, cebra.solver.SingleSessionSolver)
     for dataset, loader in [("demo-discrete", cebra.data.DiscreteDataLoader),
                             ("demo-continuous", cebra.data.ContinuousDataLoader
                             ), ("demo-mixed", cebra.data.MixedDataLoader)]
     for model in
     ["offset1-model", "offset10-model", "offset40-model-4x-subsample"]])
def test_single_session_auxvar(data_name, loader_initfunc, model_architecture,
                               solver_initfunc):

    pytest.skip("Not yet supported")

    loader = _get_loader(data_name, loader_initfunc)
    model = _make_model(loader.dataset)
    behavior_model = _make_behavior_model(loader.dataset)  # noqa: F841

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )

    batch = next(iter(loader))
    assert batch.reference.shape == (32, loader.dataset.input_dimension, 10)
    log = solver.step(batch)
    assert isinstance(log, dict)

    solver.fit(loader)

    assert not hasattr(solver, 'num_sessions')
    assert solver.n_features == loader.dataset.neural.shape[1]


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [("demo-continuous", model, cebra.data.HybridDataLoader,
      cebra.solver.SingleSessionHybridSolver)
     for model in ["offset1-model", "offset10-model"]])
def test_single_session_hybrid(data_name, loader_initfunc, model_architecture,
                               solver_initfunc):
    loader, data = _get_loader(data_name, loader_initfunc)
    model = _make_model(data, model_architecture)
    data.configure_for(model)
    offset = model.get_offset()
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    inference = solver._inference(batch)
    assert len(inference) == 2
    log = solver.step(batch)
    assert isinstance(log, dict)

    X = loader.dataset.neural
    with pytest.raises(ValueError, match="not.*fitted"):
        solver.transform(X)

    solver.fit(loader)

    assert not hasattr(solver, 'num_sessions')
    assert solver.n_features == X.shape[1]

    _test_single_session_transform(solver, X, offset)


# ---- multi-session solver ----


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [(dataset, model, loader, cebra.solver.MultiSessionSolver)
     for dataset, loader in [
         ("demo-discrete-multisession",
          cebra.data.DiscreteMultiSessionDataLoader),
         ("demo-continuous-multisession",
          cebra.data.ContinuousMultiSessionDataLoader),
     ]
     for model in ["offset1-model", "offset10-model"]])
def test_multi_session(data_name, loader_initfunc, model_architecture,
                       solver_initfunc):
    loader, data = _get_loader(data_name, loader_initfunc)
    model = nn.ModuleList([
        _make_model(dataset, model_architecture)
        for dataset in data.iter_sessions()
    ])
    data.configure_for(model)
    offset_length = len(model[0].get_offset())

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    for session_id, dataset in enumerate(loader.dataset.iter_sessions()):
        assert batch[session_id].reference.shape == (32,
                                                     dataset.input_dimension,
                                                     offset_length)
        assert batch[session_id].index is not None

    log = solver.step(batch)
    assert isinstance(log, dict)

    X = [
        loader.dataset.get_session(i).neural
        for i in range(loader.dataset.num_sessions)
    ]
    with pytest.raises(ValueError, match="not.*fitted"):
        solver.transform(X[0], session_id=0)

    solver.fit(loader)

    assert solver.num_sessions == 3
    assert solver.n_features == [X[i].shape[1] for i in range(len(X))]

    embedding = solver.transform(X[0], session_id=0)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X[0].shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X[1], session_id=1)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X[1].shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X[0], session_id=0, pad_before_transform=False)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X[0].shape[0] -
                               len(solver.model[0].get_offset()) + 1,
                               OUTPUT_DIMENSION)

    with pytest.raises(ValueError, match="torch.Tensor"):
        embedding = solver.transform(X[0].numpy(), session_id=0)

    with pytest.raises(ValueError, match="shape"):
        embedding = solver.transform(X[1], session_id=0)
    with pytest.raises(ValueError, match="shape"):
        embedding = solver.transform(X[0], session_id=1)

    with pytest.raises(RuntimeError, match="No.*session_id"):
        embedding = solver.transform(X[0])
    with pytest.raises(RuntimeError, match="session_id.*provided"):
        embedding = solver.transform(X)
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = solver.transform(X[0], session_id=5)
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = solver.transform(X[0], session_id=-1)

    for param in solver.parameters(session_id=0):
        assert isinstance(param, torch.Tensor)

    fitted_solver = copy.deepcopy(solver)
    with tempfile.TemporaryDirectory() as temp_dir:
        solver.save(temp_dir)
        solver.load(temp_dir)
    _assert_equal(fitted_solver, solver)


# ---- unified solver ----


def _make_val_data(dataset):
    if isinstance(dataset, cebra.datasets.demo.DemoDataset):
        return dataset.neural
    elif isinstance(dataset, cebra.datasets.demo.DemoDatasetUnified):
        return [session.neural for session in dataset.iter_sessions()], [
            session.continuous_index for session in dataset.iter_sessions()
        ]


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [(dataset, model, loader, cebra.solver.UnifiedSolver)
     for dataset, loader in [
         ("demo-continuous-unified", cebra.data.UnifiedLoader),
     ]
     for model in ["offset1-model", "offset10-model"]])
def test_unified_session(data_name, model_architecture, loader_initfunc,
                         solver_initfunc):
    loader, data = _get_loader(data_name, loader_initfunc)
    model = _make_model(data, model_architecture)
    data.configure_for(model)
    offset = model.get_offset()

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    assert batch.reference.shape == (32, loader.dataset.input_dimension,
                                     len(offset))

    log = solver.step(batch)
    assert isinstance(log, dict)

    solver.fit(loader)
    data, labels = _make_val_data(loader.dataset)

    assert solver.num_sessions == 3
    assert solver.n_features == sum(
        [data[i].shape[1] for i in range(len(data))])

    for i in range(loader.dataset.num_sessions):
        emb = solver.transform(data, labels, session_id=i)
        assert emb.shape == (loader.dataset.num_timepoints, 3)

        emb = solver.transform(data, labels, session_id=i, batch_size=300)
        assert emb.shape == (loader.dataset.num_timepoints, 3)


# ---- checkpoint loading ----


@pytest.mark.parametrize("scheduler_type", ["lambda", "cosine"])
def test_checkpoint_resumes_scheduler(tmp_path, scheduler_type):
    def make_solver():
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        if scheduler_type == "lambda":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.9**epoch)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10)
        return cebra.solver.SingleSessionSolver(
            model=model,
            criterion=cebra.models.InfoNCE(),
            optimizer=optimizer,
            scheduler=scheduler)

    original = make_solver()
    original.n_features = 4
    for _ in range(4):
        original.optimizer.step()
        original.scheduler.step()
    original.save(str(tmp_path), "model.pt")

    resumed = make_solver()
    resumed.load(str(tmp_path), "model.pt")
    assert resumed.scheduler.state_dict() == original.scheduler.state_dict()
    for _ in range(3):
        original.optimizer.step()
        original.scheduler.step()
        resumed.optimizer.step()
        resumed.scheduler.step()
        assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(
            original.optimizer.param_groups[0]["lr"])


def test_load_legacy_checkpoint_without_scheduler():
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    solver = cebra.solver.SingleSessionSolver(
        model=model,
        criterion=cebra.models.InfoNCE(),
        optimizer=optimizer)
    checkpoint = solver.state_dict()
    assert "scheduler" not in checkpoint

    solver.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10)
    scheduler_state = copy.deepcopy(solver.scheduler.state_dict())
    solver.load_state_dict(checkpoint)
    assert solver.scheduler.state_dict() == scheduler_state


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.skipif(
    tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 6),
    reason="torch<2.6 does not support named safe-global tuples")
def test_weights_only_checkpoint_with_numpy_scalar(tmp_path, dtype):
    path = tmp_path / "model.pt"
    torch.save({"value": dtype(4)}, path)
    with torch.serialization.safe_globals(
            cebra.solver.base._CHECKPOINT_SAFE_GLOBALS):
        checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["value"] == dtype(4)
    assert type(checkpoint["value"]) is dtype


def test_load_checkpoint_with_numpy_scalars(tmp_path):
    # Checkpoints of unified solvers saved before n_features was a Python int store it as a
    # numpy int64; torch>=2.6 loads with weights_only=True by default.
    model = cebra.models.init("offset1-model", num_neurons=4, num_units=8, num_output=3)
    solver = cebra.solver.SingleSessionSolver(
        model=model,
        criterion=cebra.models.InfoNCE(),
        optimizer=torch.optim.Adam(model.parameters()))
    solver.n_features = np.int64(4)
    solver.log[("loss_val", 0)] = [np.float64(0.5)]
    solver.save(str(tmp_path), "model.pt")

    other = cebra.solver.SingleSessionSolver(
        model=cebra.models.init("offset1-model", num_neurons=4, num_units=8, num_output=3),
        criterion=cebra.models.InfoNCE(),
        optimizer=torch.optim.Adam(model.parameters()))
    other.load(str(tmp_path), "model.pt")
    assert other.n_features == 4
    assert other.log[("loss_val", 0)] == [0.5]
    for key, value in solver.model.state_dict().items():
        assert torch.equal(value, other.model.state_dict()[key])


@pytest.mark.skipif(not hasattr(torch.serialization, "safe_globals"),
                    reason="torch<2.5 has no weights_only allowlist")
def test_load_checkpoint_rejects_other_objects(tmp_path):
    # Only numpy scalars are added to the allowlist: a checkpoint holding any other
    # object (here a numpy array) is still refused when torch loads with weights_only=True.
    path = tmp_path / "model.pt"
    torch.save({"n_features": np.arange(3)}, path)
    with pytest.raises(Exception):
        with torch.serialization.safe_globals(
                cebra.solver.base._CHECKPOINT_SAFE_GLOBALS):
            torch.load(path, weights_only=True)


# ---- unified solver: aligned inputs and batched transform ----


class _ConvEncoder(cebra.models.Model, cebra.models.ConvolutionalModelMixin):
    """Small convolutional-offset encoder."""

    def __init__(self, num_input, num_output, left=2, right=1):
        super().__init__(num_input=num_input, num_output=num_output)
        self.offset = cebra.data.Offset(left, right)
        self.conv = torch.nn.Conv1d(num_input, num_output, left + right)

    def forward(
            self,
            x):  # (batch, N, T) -> (batch, num_output, T - left - right + 1)
        return torch.tanh(self.conv(x))

    def get_offset(self):
        return self.offset


def _tied_sessions(
        lengths=(60, 90, 75), n_channels=(3, 4, 5), n_frames=15, seed=0):
    """Sessions whose labels repeat exactly (ties), as in repeated stimulus presentations."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(n_frames, 2, generator=gen)
    inputs, labels = [], []
    for length, n in zip(lengths, n_channels):
        reps = -(-length // n_frames)
        labels.append(base.repeat(reps, 1)[:length].clone())
        inputs.append(torch.randn(length, n, generator=gen))
    return inputs, labels


def _unified_solver(n_total, left=2, right=1, num_output=4):
    model = _ConvEncoder(n_total, num_output, left, right)
    solver = cebra.solver.UnifiedSolver(model=model,
                                        criterion=cebra.models.InfoNCE(),
                                        optimizer=torch.optim.Adam(
                                            model.parameters()))
    solver.n_features, solver.num_sessions = n_total, 3
    return solver


def test_aligned_inputs_shape_single_row_and_seed():
    inputs, labels = _tied_sessions()
    solver = _unified_solver(12)
    out = solver.aligned_inputs(inputs, labels, session_id=1)
    assert out.shape == (90, 12)
    torch.testing.assert_close(out[:, 3:7], inputs[1])

    one = solver.aligned_inputs(inputs, labels, session_id=1, indices=[7])
    assert one.shape == (1, 12)
    torch.testing.assert_close(one, out[7:8])

    a = solver.aligned_inputs(inputs, labels, 0, random_ties=True, seed=2)
    b = solver.aligned_inputs(inputs, labels, 0, random_ties=True, seed=2)
    c = solver.aligned_inputs(inputs, labels, 0, random_ties=True, seed=3)
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, c)
    # random_ties=False is deterministic and takes the first tie
    d = solver.aligned_inputs(inputs, labels, 0)
    torch.testing.assert_close(d, solver.aligned_inputs(inputs, labels, 0))


@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("batch_size", [1, 7, 64, None])
def test_unified_transform_batched_matches_full_session_reference(
        pad, batch_size):
    """Batch by batch inference equals one pass over the full aligned session."""
    inputs, labels = _tied_sessions(lengths=(60, 90, 75))
    solver = _unified_solver(12, left=5, right=5)
    ref_in = solver.aligned_inputs(inputs,
                                   labels,
                                   session_id=1,
                                   random_ties=True,
                                   seed=4)
    if pad:  # repeat the first and last samples, as Solver.transform
        ref_in = torch.cat(
            [ref_in[:1].repeat(5, 1), ref_in, ref_in[-1:].repeat(4, 1)])
    with torch.no_grad():
        ref = solver.model(ref_in.T.unsqueeze(0)).squeeze(0).T
    out = solver.transform(inputs,
                           labels,
                           session_id=1,
                           batch_size=batch_size,
                           pad_before_transform=pad,
                           random_ties=True,
                           seed=4)
    assert out.shape == ((90, 4) if pad else (81, 4))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_unified_transform_session_shorter_than_window():
    inputs, labels = _tied_sessions(lengths=(8, 90, 75))
    solver = _unified_solver(12, left=5, right=5)
    assert solver.transform(inputs, labels, session_id=0).shape == (8, 4)
    with pytest.raises(ValueError, match="receptive field"):
        solver.transform(inputs,
                         labels,
                         session_id=0,
                         pad_before_transform=False)


def test_unified_transform_random_ties_default_false_and_seed():
    inputs, labels = _tied_sessions()
    solver = _unified_solver(12)
    a = solver.transform(inputs, labels, session_id=0)
    torch.manual_seed(1)
    b = solver.transform(inputs, labels, session_id=0)
    torch.testing.assert_close(a, b)
    r1 = solver.transform(inputs,
                          labels,
                          session_id=0,
                          random_ties=True,
                          seed=0)
    r2 = solver.transform(inputs,
                          labels,
                          session_id=0,
                          random_ties=True,
                          seed=0)
    torch.testing.assert_close(r1, r2)
    with pytest.raises(ValueError):
        solver.transform(inputs, labels)  # session_id is required


def test_unified_transform_default_seed_and_random_ties_need_seed():
    """The default seed gives the same tie draws in every call; random ties need a seed."""
    inputs, labels = _tied_sessions()
    solver = _unified_solver(12)
    a = solver.transform(inputs, labels, session_id=0, random_ties=True)
    b = solver.transform(inputs, labels, session_id=0, random_ties=True)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        solver.aligned_inputs(inputs, labels, 0, random_ties=True),
        solver.aligned_inputs(inputs, labels, 0, random_ties=True, seed=0))
    with pytest.raises(ValueError, match="needs a seed"):
        solver.transform(inputs, labels, session_id=0, random_ties=True, seed=None)


# ---- unified xCEBRA solver ----


def _xcebra_solver():
    module = _ConvEncoder(12, 6)
    solver = cebra.solver.UnifiedSolverxCEBRA(
        model=module,
        criterion=cebra.models.InfoNCE(),
        optimizer=torch.optim.Adam(module.parameters()),
        feature_ranges=[slice(0, 3), slice(3, 6)],
        renormalize=True,
    )
    solver.n_features, solver.num_sessions = 12, 3
    return solver


@pytest.mark.parametrize("initial", [True, False])
def test_unified_xcebra_transform_concatenated_and_split_mode_restored(initial):
    """transform returns the single concatenated embedding and restores the model's split mode,
    also when it raises."""
    inputs, labels = _tied_sessions()
    solver = _xcebra_solver()
    solver.model.set_split_outputs(initial)
    emb = solver.transform(inputs, labels, session_id=1, batch_size=10)
    assert torch.is_tensor(emb) and emb.shape == (90, 6)
    assert solver.model.split_outputs is initial
    with pytest.raises(Exception):
        solver.transform(inputs, [l[:, :1] * 0 for l in labels][:2],
                         session_id=0)
    assert solver.model.split_outputs is initial


# ---- unified checkpoint loading ----


def test_unified_checkpoint_loads_with_weights_only(tmp_path):
    inputs, labels = _tied_sessions()
    dataset = cebra.data.UnifiedDataset(*[
        cebra.data.TensorDataset(x, continuous=l)
        for x, l in zip(inputs, labels)
    ])
    assert type(dataset.input_dimension) is int
    solver = _unified_solver(sum(x.shape[1] for x in inputs))
    solver._set_fitted_params(cebra.data.UnifiedLoader(dataset, num_steps=1))
    solver.save(str(tmp_path), "model.pt")

    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    assert checkpoint["n_features"] == dataset.input_dimension
    other = _unified_solver(sum(x.shape[1] for x in inputs))
    other.load(str(tmp_path), "model.pt")
    assert other.n_features == dataset.input_dimension
