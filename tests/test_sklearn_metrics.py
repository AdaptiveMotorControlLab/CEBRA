#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
import math

import numpy as np
import pytest
import torch

import cebra
import cebra.integrations.sklearn.cebra as cebra_sklearn_cebra
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


def test_sklearn_consistency():
    # Example data
    np.random.seed(42)
    embedding1 = np.random.uniform(0, 1, (10000, 4))
    embedding2 = np.random.uniform(0, 1, (10000, 10))
    embedding3 = np.random.uniform(0, 1, (8000, 6))
    embedding4 = np.random.uniform(0, 1, (5000, 7))
    embeddings_datasets = [embedding1, embedding2, embedding3, embedding4]
    embeddings_runs = [embedding1, embedding2, embedding1, embedding2]

    labels1 = np.random.uniform(0, 1, (10000,))
    labels1_invalid = np.random.uniform(0, 1, (10000, 3))
    labels2 = np.random.uniform(0, 1, (10000,))
    labels3 = np.random.uniform(0, 1, (8000,))
    labels4 = np.random.uniform(0, 1, (5000,))
    labels_datasets = [labels1, labels2, labels3, labels4]
    labels_runs = [labels1, labels2, labels1, labels2]

    dataset_ids = ["achilles", "buddy", "buddy", "achilles"]

    # between-runs consistency
    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        embeddings=embeddings_runs, between="runs")
    assert scores.shape == (12,)
    assert pairs.shape == (12, 2)
    assert len(datasets) == 1
    assert math.isclose(scores[0], 0, abs_tol=0.05)

    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        embeddings=[embedding1, embedding1], between="runs")
    assert scores.shape == (2,)
    assert pairs.shape == (2, 2)
    assert len(datasets) == 1
    assert math.isclose(scores[0], 1, abs_tol=1e-9)

    # scores are put in the right part of the scores matrix
    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        embeddings=[embedding1, embedding2, embedding1, embedding2],
        between="runs")
    assert scores.shape == (12,)
    assert pairs.shape == (12, 2)
    assert len(datasets) == 1
    # scores should contain the consistencies in the following way:
    # [emb1-emb2, emb1-emb1, emb1-emb2, emb2-emb1, emb2-emb1, emb2-emb2,
    # emb1-emb1, emb1-emb2, emb1-emb2, emb2-emb1, emb2-emb2, emb2-emb1]
    # we check that emb1-emb1 larger than all other scores with emb1
    assert scores[1] > scores[0] and scores[1] > scores[2]
    assert scores[6] > scores[7] and scores[6] > scores[8]
    assert all(math.isclose(scores[i], 0, abs_tol=0.05) for i in [0, 2, 7, 8])

    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        embeddings_runs, dataset_ids=dataset_ids, between="runs")
    assert scores.shape == (2,)
    assert pairs.shape == (2, 2, 2)
    assert len(datasets) == 2

    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        [torch.Tensor(embedding) for embedding in embeddings_runs],
        dataset_ids=dataset_ids,
        between="runs",
    )
    assert scores.shape == (2,)
    assert pairs.shape == (2, 2, 2)
    assert len(datasets) == 2

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
            embeddings=[embedding1, embedding2, embedding1],
            dataset_ids=["achilles", "buddy", "buddy"],
            between="runs",
        )

    # between-datasets consistency
    # random embeddings provide R2 close to 0
    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        embeddings_datasets,
        dataset_ids=dataset_ids,
        labels=labels_datasets,
        between="datasets",
    )
    assert scores.shape == (2,)
    assert pairs.shape == (8, 2)
    assert len(datasets) == 2
    assert math.isclose(scores[0], 0, abs_tol=0.05)

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

    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        embeddings_datasets, labels=labels_datasets, between="datasets")
    assert scores.shape == (12,)
    assert pairs.shape == (12, 2)
    assert len(datasets) == 4

    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        [torch.Tensor(embedding) for embedding in embeddings_datasets],
        dataset_ids=dataset_ids,
        labels=[torch.Tensor(label) for label in labels_datasets],
        between="datasets",
    )
    assert scores.shape == (2,)
    assert pairs.shape == (8, 2)
    assert len(datasets) == 2

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
