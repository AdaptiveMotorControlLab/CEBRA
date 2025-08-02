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
import math

import numpy as np
import pytest
import torch

import cebra
import cebra.integrations.sklearn.cebra as cebra_sklearn_cebra
import cebra.integrations.sklearn.helpers as cebra_sklearn_helpers
import cebra.integrations.sklearn.metrics as cebra_sklearn_metrics


def test_imports():
    import cebra

    assert hasattr(cebra, "sklearn")


def test_sklearn_infonce_loss():
    max_loss_iterations = 2
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture="offset10-model",
        max_iterations=5,
        batch_size=128,
    )

    # Example data
    X = torch.tensor(np.random.uniform(0, 1, (1000, 50)))
    y_c1 = torch.tensor(np.random.uniform(0, 1, (1000, 5)))
    y_d = np.random.randint(0, 10, (1000,))

    X_test = torch.tensor(np.random.uniform(0, 1, (500, 50)))
    X_test_2 = torch.tensor(np.random.uniform(0, 1, (600, 30)))
    y_c1_test = torch.tensor(np.random.uniform(0, 1, (500, 5)))
    y_c1_test_2 = torch.tensor(np.random.uniform(0, 1, (600, 2)))
    y_d_test = np.random.randint(0, 10, (500,))

    # Single session
    cebra_model.fit(X, y_c1)
    score = cebra.sklearn.metrics.infonce_loss(cebra_model,
                                               X_test,
                                               y_c1_test,
                                               session_id=0,
                                               num_batches=max_loss_iterations)
    assert isinstance(score, float)
    score = cebra.sklearn.metrics.infonce_loss(cebra_model,
                                               X_test,
                                               y_c1_test,
                                               num_batches=max_loss_iterations)
    assert isinstance(score, float)
    score = cebra.sklearn.metrics.infonce_loss(
        cebra_model,
        X_test,
        y_c1_test,
        num_batches=max_loss_iterations,
        correct_by_batchsize=True,
    )
    assert isinstance(score, float)

    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test,
            y_c1_test,
            num_batches=max_loss_iterations,
            session_id=2,
        )
    with pytest.raises(ValueError, match="Labels.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test,
            y_d_test,
            num_batches=max_loss_iterations,
            session_id=0)
    with pytest.raises(ValueError, match="Number.*index.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model, X_test, num_batches=max_loss_iterations)
    with pytest.raises(ValueError, match="Number.*index.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test,
            y_c1_test,
            y_d_test,
            num_batches=max_loss_iterations)
    with pytest.raises(ValueError, match="Number.*index.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test,
            y_c1_test_2,
            y_d_test,
            num_batches=max_loss_iterations)

    cebra_model.fit(X, y_d)
    score = cebra.sklearn.metrics.infonce_loss(cebra_model,
                                               X_test,
                                               y_d_test,
                                               num_batches=max_loss_iterations)
    assert isinstance(score, float)
    score = cebra.sklearn.metrics.infonce_loss(cebra_model,
                                               X_test,
                                               y_d_test,
                                               num_batches=max_loss_iterations,
                                               session_id=0)
    assert isinstance(score, float)

    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test,
            y_d_test,
            num_batches=max_loss_iterations,
            session_id=2)
    with pytest.raises(ValueError, match="Labels.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test,
            y_c1_test,
            num_batches=max_loss_iterations,
            session_id=0,
        )
    with pytest.raises(ValueError, match="Number.*index.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model, X_test, num_batches=max_loss_iterations)

    # Multisession
    cebra_model.fit([X, X], [y_c1, y_c1])
    score = cebra.sklearn.metrics.infonce_loss(cebra_model,
                                               X_test,
                                               y_c1_test,
                                               num_batches=max_loss_iterations,
                                               session_id=0)
    assert isinstance(score, float)
    score = cebra.sklearn.metrics.infonce_loss(cebra_model,
                                               X_test,
                                               y_c1_test,
                                               num_batches=max_loss_iterations,
                                               session_id=1)
    assert isinstance(score, float)

    with pytest.raises(ValueError, match="Labels.*invalid"):
        cebra.sklearn.metrics.infonce_loss(cebra_model,
                                           X_test,
                                           y_d_test,
                                           num_batches=max_loss_iterations,
                                           session_id=0)
    with pytest.raises(ValueError, match="shape"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test_2,
            y_c1_test,
            num_batches=max_loss_iterations,
            session_id=0,
        )
    with pytest.raises(RuntimeError, match="No.*session_id"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model, X_test, y_c1_test, num_batches=max_loss_iterations)
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test,
            y_c1_test,
            num_batches=max_loss_iterations,
            session_id=3,
        )
    with pytest.raises(ValueError, match="Number.*index.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model, X_test, num_batches=max_loss_iterations, session_id=0)
    with pytest.raises(NotImplementedError, match="Data.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            [X_test, X_test_2],
            y_c1_test,
            num_batches=max_loss_iterations,
            session_id=0,
        )
    with pytest.raises(NotImplementedError, match="Labels.*invalid"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model,
            X_test,
            [y_c1_test, y_c1_test],
            num_batches=max_loss_iterations,
            session_id=0,
        )

    # No batch size
    cebra_model_no_bs = cebra_sklearn_cebra.CEBRA(
        model_architecture="offset10-model",
        max_iterations=max_loss_iterations,
        batch_size=None,
    )

    cebra_model_no_bs.fit(X_test)
    score = cebra.sklearn.metrics.infonce_loss(cebra_model_no_bs,
                                               X_test,
                                               num_batches=max_loss_iterations)

    with pytest.raises(ValueError, match="Batch.*size"):
        score = cebra.sklearn.metrics.infonce_loss(
            cebra_model_no_bs,
            X_test,
            num_batches=max_loss_iterations,
            correct_by_batchsize=True,
        )


def test_sklearn_datasets_consistency():
    # Example data
    np.random.seed(42)
    embedding1 = np.random.uniform(0, 1, (10000, 4))
    embedding2 = np.random.uniform(0, 1, (10000, 10))
    embedding3 = np.random.uniform(0, 1, (8000, 6))
    embedding4 = np.random.uniform(0, 1, (5000, 7))
    embeddings_datasets = [embedding1, embedding2, embedding3, embedding4]

    labels1 = np.random.uniform(0, 1, (10000,))
    labels1_invalid = np.random.uniform(0, 1, (10000, 3))
    labels2 = np.random.uniform(0, 1, (10000,))
    labels3 = np.random.uniform(0, 1, (8000,))
    labels4 = np.random.uniform(0, 1, (5000,))
    labels_datasets = [labels1, labels2, labels3, labels4]

    dataset_ids = ["achilles", "buddy", "cicero", "gatsby"]

    # random embeddings provide R2 close to 0
    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        embeddings_datasets,
        dataset_ids=dataset_ids,
        labels=labels_datasets,
        between="datasets",
    )
    assert scores.shape == (12,)
    assert pairs.shape == (12, 2)
    assert len(datasets) == 4
    assert math.isclose(scores[0], 0, abs_tol=0.05)

    # no labels
    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        embeddings_datasets, labels=labels_datasets, between="datasets")
    assert scores.shape == (12,)
    assert pairs.shape == (12, 2)
    assert len(datasets) == 4

    # identical embeddings provide R2 close to 1
    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        [embedding1, embedding1],
        dataset_ids=["achilles", "buddy"],
        labels=[labels1, labels1],
        between="datasets",
    )
    assert scores.shape == (2,)
    assert pairs.shape == (2, 2)
    assert len(datasets) == 2
    assert math.isclose(scores[0], 1, abs_tol=1e-9)

    # Tensor
    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        [torch.Tensor(embedding) for embedding in embeddings_datasets],
        dataset_ids=dataset_ids,
        labels=[torch.Tensor(label) for label in labels_datasets],
        between="datasets",
    )
    assert scores.shape == (12,)
    assert pairs.shape == (12, 2)
    assert len(datasets) == 4

    with pytest.raises(ValueError, match="Missing.*between"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            embeddings_datasets, labels=labels_datasets)
    with pytest.raises(ValueError, match="Missing.*labels"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            embeddings_datasets,
            between="datasets",
        )
    with pytest.raises(ValueError, match="Invalid.*labels"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            embeddings_datasets,
            labels=[labels1, labels2],
            between="datasets",
        )
    with pytest.raises(ValueError, match="Invalid.*dataset_ids"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            embeddings=[embedding1, embedding4],
            labels=[labels1, labels4],
            dataset_ids=["achilles", "achilles"],
            between="datasets",
        )
    with pytest.raises(ValueError, match="Invalid.*dtype"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            embeddings=[embedding1, embedding2],
            labels=[
                ["test" for i in range(len(embedding1))],
                ["test2" for j in range(len(embedding2))],
            ],
            dataset_ids=["achilles", "buddy"],
            between="datasets",
        )
    with pytest.raises(NotImplementedError, match="Invalid.*labels"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            embeddings=[embedding1, embedding2],
            labels=[
                labels1_invalid,
                labels2,
            ],
            dataset_ids=["achilles", "buddy"],
            between="datasets",
        )


def test_sklearn_runs_consistency():
    # Example data
    np.random.seed(42)
    embedding1 = np.random.uniform(0, 1, (10000, 4))
    embedding2 = np.random.uniform(0, 1, (10000, 10))
    embedding3 = np.random.uniform(0, 1, (8000, 10))
    embeddings_runs = [embedding1, embedding2, embedding1, embedding2]
    invalid_embeddings_runs = [embedding1, embedding2, embedding3]

    labels1 = np.random.uniform(0, 1, (10000,))
    labels2 = np.random.uniform(0, 1, (10000,))
    labels_runs = [labels1, labels2, labels1, labels2]

    # between-runs consistency
    scores, pairs, ids = cebra_sklearn_metrics.consistency_score(
        embeddings=embeddings_runs, between="runs")
    assert scores.shape == (12,)
    assert pairs.shape == (12, 2)
    assert len(ids) == 4
    assert math.isclose(scores[0], 0, abs_tol=0.05)

    scores, pairs, ids = cebra_sklearn_metrics.consistency_score(
        embeddings=[embedding1, embedding1], between="runs")
    assert scores.shape == (2,)
    assert pairs.shape == (2, 2)
    assert len(ids) == 2
    assert math.isclose(scores[0], 1, abs_tol=1e-9)

    scores, pairs, ids = cebra_sklearn_metrics.consistency_score(
        [torch.Tensor(embedding) for embedding in embeddings_runs],
        between="runs",
    )
    assert scores.shape == (12,)
    assert pairs.shape == (12, 2)
    assert len(ids) == 4

    with pytest.raises(ValueError, match="No.*dataset.*ID"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            embeddings_runs,
            dataset_ids=["run1", "run2", "run3", "run4"],
            between="runs")

    with pytest.raises(ValueError, match="Missing.*between"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(embeddings_runs)

    with pytest.raises(ValueError, match="No.*labels"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(embeddings_runs,
                                                          labels=labels_runs,
                                                          between="runs")

    with pytest.raises(ValueError, match="Invalid.*embeddings"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            embeddings=[embedding1], between="runs")

    with pytest.raises(ValueError, match="Invalid.*embeddings"):
        _, _, _ = cebra_sklearn_metrics.consistency_score(
            invalid_embeddings_runs, between="runs")


def test_align_embeddings():
    # Example data
    np.random.seed(42)
    embedding1 = np.random.uniform(0, 1, (10000, 4))
    embedding2 = np.random.uniform(0, 1, (10000, 10))
    embedding3 = np.random.uniform(0, 1, (8000, 6))
    embeddings_datasets = [embedding1, embedding2, embedding3]

    labels1 = np.random.uniform(0, 1, (10000,))
    labels2 = np.random.uniform(0, 1, (10000,))
    labels3 = np.random.uniform(0, 1, (8000,))
    labels_datasets = [labels1, labels2, labels3]

    embeddings = cebra_sklearn_helpers.align_embeddings(
        embeddings=embeddings_datasets,
        labels=labels_datasets,
        normalize=False,
        n_bins=100)

    normalized_embeddings = cebra_sklearn_helpers.align_embeddings(
        embeddings=embeddings_datasets,
        labels=labels_datasets,
        normalize=True,
        n_bins=100)

    assert len(embeddings) == len(embeddings_datasets)
    assert len(normalized_embeddings) == len(embeddings_datasets)
    assert len(embeddings) == len(normalized_embeddings)


@pytest.mark.parametrize("seed", [42, 24, 10])
def test_goodness_of_fit_score(seed):
    """
    Ensure that the GoF score is close to 0 for a model fit on random data.
    """
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture="offset1-model",
        max_iterations=5,
        batch_size=512,
    )
    generator = torch.Generator().manual_seed(seed)
    X = torch.rand(5000, 50, dtype=torch.float32, generator=generator)
    y = torch.rand(5000, 5, dtype=torch.float32, generator=generator)
    cebra_model.fit(X, y)
    score = cebra_sklearn_metrics.goodness_of_fit_score(cebra_model,
                                                        X,
                                                        y,
                                                        session_id=0,
                                                        num_batches=500)
    assert isinstance(score, float)
    assert np.isclose(score, 0, atol=0.01)


@pytest.mark.parametrize("seed", [42, 24, 10])
def test_goodness_of_fit_history(seed):
    """
    Ensure that the GoF score is higher for a model fit on data with underlying
    structure than for a model fit on random data.
    """

    # Generate data
    generator = torch.Generator().manual_seed(seed)
    X = torch.rand(1000, 50, dtype=torch.float32, generator=generator)
    y_random = torch.rand(len(X), 5, dtype=torch.float32, generator=generator)
    linear_map = torch.randn(50, 5, dtype=torch.float32, generator=generator)
    y_linear = X @ linear_map

    def _fit_and_get_history(X, y):
        cebra_model = cebra_sklearn_cebra.CEBRA(
            model_architecture="offset1-model",
            max_iterations=150,
            batch_size=512,
            device="cpu")
        cebra_model.fit(X, y)
        history = cebra_sklearn_metrics.goodness_of_fit_history(cebra_model)
        # NOTE(stes): Ignore the first 5 iterations, they can have nonsensical values
        # due to numerical issues.
        return history[5:]

    history_random = _fit_and_get_history(X, y_random)
    history_linear = _fit_and_get_history(X, y_linear)

    assert isinstance(history_random, np.ndarray)
    assert history_random.shape[0] > 0
    # NOTE(stes): Ignore the first 5 iterations, they can have nonsensical values
    # due to numerical issues.
    history_random_non_negative = history_random[history_random >= 0]
    np.testing.assert_allclose(history_random_non_negative, 0, atol=0.075)

    assert isinstance(history_linear, np.ndarray)
    assert history_linear.shape[0] > 0

    assert np.all(history_linear[-20:] > history_random[-20:])


@pytest.mark.parametrize("seed", [42, 24, 10])
@pytest.mark.parametrize("batch_size", [100, 200])
@pytest.mark.parametrize("num_negatives", [None, 100, 200])
def test_infonce_to_goodness_of_fit(seed, batch_size, num_negatives):
    """Test the conversion from InfoNCE loss to goodness of fit metric."""
    nats_to_bits = np.log2(np.e)

    # Test with model
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture="offset10-model",
        max_iterations=5,
        batch_size=batch_size,
        num_negatives=num_negatives,
    )
    if num_negatives is None:
        num_negatives = batch_size

    generator = torch.Generator().manual_seed(seed)
    X = torch.rand(1000, 50, dtype=torch.float32, generator=generator)
    cebra_model.fit(X)

    # Test single value
    gof = cebra_sklearn_metrics.infonce_to_goodness_of_fit(1.0,
                                                           model=cebra_model)
    assert isinstance(gof, float)
    assert np.isclose(gof, (np.log(num_negatives) - 1.0) * nats_to_bits)

    # Test array of values
    infonce_values = np.array([1.0, 2.0, 3.0])
    gof_array = cebra_sklearn_metrics.infonce_to_goodness_of_fit(
        infonce_values, model=cebra_model)
    assert isinstance(gof_array, np.ndarray)
    assert gof_array.shape == infonce_values.shape
    assert np.allclose(gof_array,
                       (np.log(num_negatives) - infonce_values) * nats_to_bits)

    # Test with explicit batch_size and num_sessions
    gof = cebra_sklearn_metrics.infonce_to_goodness_of_fit(
        1.0, batch_size=batch_size, num_sessions=1)
    assert isinstance(gof, float)
    assert np.isclose(gof, (np.log(batch_size) - 1.0) * nats_to_bits)

    # Test error cases
    with pytest.raises(ValueError, match="batch_size.*should not be provided"):
        cebra_sklearn_metrics.infonce_to_goodness_of_fit(1.0,
                                                         model=cebra_model,
                                                         batch_size=128)

    with pytest.raises(ValueError, match="batch_size.*should not be provided"):
        cebra_sklearn_metrics.infonce_to_goodness_of_fit(1.0,
                                                         model=cebra_model,
                                                         num_sessions=1)

    # Test with unfitted model
    unfitted_model = cebra_sklearn_cebra.CEBRA(max_iterations=5)
    with pytest.raises(RuntimeError, match="Fit the CEBRA model first"):
        cebra_sklearn_metrics.infonce_to_goodness_of_fit(1.0,
                                                         model=unfitted_model)

    # Test with model having batch_size=None
    none_batch_model = cebra_sklearn_cebra.CEBRA(batch_size=None,
                                                 max_iterations=5)
    none_batch_model.fit(X)
    with pytest.raises(ValueError, match="Computing the goodness of fit"):
        cebra_sklearn_metrics.infonce_to_goodness_of_fit(1.0,
                                                         model=none_batch_model)

    # Test missing batch_size or num_sessions when model is None
    with pytest.raises(ValueError, match="batch_size.*and num_sessions"):
        cebra_sklearn_metrics.infonce_to_goodness_of_fit(1.0, batch_size=128)

    with pytest.raises(ValueError, match="batch_size.*and num_sessions"):
        cebra_sklearn_metrics.infonce_to_goodness_of_fit(1.0, num_sessions=1)
