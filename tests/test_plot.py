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
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import pytest
import torch
from sklearn.exceptions import NotFittedError

import cebra.integrations.matplotlib as cebra_plot
import cebra.integrations.sklearn.cebra as cebra_sklearn_cebra
import cebra.integrations.sklearn.metrics as cebra_sklearn_metrics


class _ConsistencyDataset():
    """A test dataset that can be indexed to obtain different permutations"""

    def __init__(self, seed=42424242):
        self._generator = np.random.default_rng(seed=seed)
        self.dataset_ids = "a", "b", "c"
        self.embeddings = [
            self._generator.normal(size=(1000, 3)) for _ in self.dataset_ids
        ]
        self.labels = [
            self._generator.uniform(0, 1, size=(1000,))
            for _ in self.dataset_ids
        ]

    def __getitem__(self, order):
        return {
            "dataset_ids": [self.dataset_ids[i] for i in order],
            "embeddings": [self.embeddings[i] for i in order],
            "labels": [self.labels[i] for i in order]
        }


def test_plot_imports():
    import cebra

    assert hasattr(cebra, "plot")
    assert hasattr(cebra, "plot_embedding")
    assert hasattr(cebra, "plot_temperature")
    assert hasattr(cebra, "plot_loss")
    assert hasattr(cebra, "plot_overview")
    assert hasattr(cebra, "compare_models")
    assert hasattr(cebra, "plot_consistency")


def test_colormaps():
    import matplotlib

    import cebra

    cmap = matplotlib.colormaps["cebra"]
    assert cmap is not None
    plt.scatter([1], [2], c=[2], cmap="cebra")


def test_plot_overview():
    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    y_c1 = np.random.uniform(0, 1, (1000, 5))
    y_c2 = np.random.uniform(0, 1, (800, 2))
    y_d = np.random.randint(0, 10, (1000,))

    # define a simple CEBRA model
    model = cebra_sklearn_cebra.CEBRA(max_iterations=10, batch_size=512)

    with pytest.raises(NotFittedError, match="not.*fitted"):
        _ = cebra_plot.plot_overview(model, X)
        plt.close()

    model.fit(X)

    fig, (ax1, ax2, ax3) = cebra_plot.plot_overview(model, X)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax1, matplotlib.axes.Axes)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert isinstance(ax3, matplotlib.axes.Axes)
    plt.close()
    fig, (ax1, ax2, ax3) = cebra_plot.plot_overview(
        model, X, embedding_kwargs={"embedding_labels": y_c1[:, 0]})
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax1, matplotlib.axes.Axes)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert isinstance(ax3, matplotlib.axes.Axes)
    plt.close()
    fig, (ax1, ax2, ax3) = cebra_plot.plot_overview(
        model, X, embedding_kwargs={"embedding_labels": y_d})
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax1, matplotlib.axes.Axes)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert isinstance(ax3, matplotlib.axes.Axes)
    plt.close()

    with pytest.raises(ValueError, match="Invalid.*embedding.*labels"):
        _ = cebra_plot.plot_overview(
            model, X, embedding_kwargs={"embedding_labels": y_c2[:, 0]})
        plt.close()
    with pytest.raises(NotImplementedError, match="Invalid.*embedding.*labels"):
        _ = cebra_plot.plot_overview(
            model, X, embedding_kwargs={"embedding_labels": y_c1[:, :1]})
        plt.close()


def test_plot_temperature():
    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()

    # define a simple CEBRA model
    model = cebra_sklearn_cebra.CEBRA(max_iterations=10, batch_size=512)

    with pytest.raises(NotFittedError, match="not.*fitted"):
        _ = cebra_plot.plot_temperature(model, ax=ax)
    with pytest.raises(NotFittedError, match="not.*fitted"):
        _ = cebra_plot.plot(model, which="temperature", ax=ax)

    model.fit(X)

    ax = cebra_plot.plot_temperature(model, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot(model, which="temperature", ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)

    plt.close()


def test_plot_loss():
    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()

    # define a simple CEBRA model
    model = cebra_sklearn_cebra.CEBRA(max_iterations=10, batch_size=512)

    with pytest.raises(NotFittedError, match="not.*fitted"):
        _ = cebra_plot.plot_loss(model, ax=ax)
    with pytest.raises(NotFittedError, match="not.*fitted"):
        _ = cebra_plot.plot(model, which="loss", ax=ax)

    model.fit(X)

    ax = cebra_plot.plot_loss(model, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot(model, which="loss", ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot(model, which="loss", ax=ax, label="test")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


@pytest.mark.parametrize("matplotlib_version",
                         ["3.3", "3.4.2", "3.5", "3.6", "3.7"])
def test_compare_models_with_different_versions(matplotlib_version):
    # example dataset
    X = np.random.uniform(0, 1, (1000, 2))
    n_models = 2

    fitted_models = []
    for _ in range(n_models):
        fitted_models.append(
            cebra_sklearn_cebra.CEBRA(max_iterations=10, batch_size=128).fit(X))

    # minimum version of matplotlib
    minimum_version = "3.6"

    if pkg_resources.parse_version(
            matplotlib_version) < pkg_resources.parse_version(minimum_version):
        with pytest.raises(ImportError):
            cebra_plot.compare_models(models=fitted_models,
                                      patched_version=matplotlib_version)


def test_compare_models():
    # example dataset
    X = np.random.uniform(0, 1, (100, 5))
    n_models = 4

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()

    models, labels = [], []
    # define CEBRA models
    for i in range(n_models):
        models.append(
            cebra_sklearn_cebra.CEBRA(max_iterations=10, batch_size=512))
        labels.append(f"model_{i}")

    with pytest.raises(NotFittedError, match="not.*fitted"):
        _ = cebra_plot.compare_models(models, ax=ax)

    for model in models:
        model.fit(X)

    ax = cebra_plot.compare_models(models, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.compare_models(models, ax=ax, cmap="viridis")
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.compare_models(models, ax=ax, labels=labels)
    assert isinstance(ax, matplotlib.axes.Axes)

    with pytest.raises(ValueError, match="Invalid.*models"):
        _ = cebra_plot.compare_models("test", ax=ax)
    with pytest.raises(ValueError, match="Invalid.*models"):
        invalid_models = copy.deepcopy(models)
        invalid_models.append("test")
        _ = cebra_plot.compare_models(invalid_models, ax=ax)
    with pytest.raises(ValueError, match="Invalid.*labels"):
        _ = cebra_plot.compare_models(models, labels="test", ax=ax)
    with pytest.raises(ValueError, match="Invalid.*labels"):
        long_labels = copy.deepcopy(labels)
        long_labels.append("test")
        _ = cebra_plot.compare_models(models, labels=long_labels, ax=ax)
    with pytest.raises(ValueError, match="Invalid.*labels"):
        invalid_labels = copy.deepcopy(labels)
        ele = invalid_labels.pop()
        invalid_labels.append(["a"])
        _ = cebra_plot.compare_models(models, labels=invalid_labels, ax=ax)

    plt.close()


@pytest.mark.parametrize("output_dimension, idx_order", [(8, (2, 3, 4)),
                                                         (2, (0, 1))])
def test_plot_embedding(output_dimension, idx_order):
    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    y_c1 = np.random.uniform(0, 1, (1000, 5))
    y_c2 = np.random.uniform(0, 1, (800, 2))
    y_d = np.random.randint(0, 10, (1000,))

    fig = plt.figure(figsize=(5, 5))
    if output_dimension < 3:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection="3d")

    # integration tests
    model = cebra_sklearn_cebra.CEBRA(max_iterations=10,
                                      batch_size=512,
                                      output_dimension=output_dimension)

    with pytest.raises(NotFittedError, match="not.*fitted"):
        _ = cebra_plot.plot(model, which="embedding", ax=ax, X=X)

    model.fit(X)
    embedding = model.transform(X)

    ax = cebra_plot.plot_embedding(embedding=embedding)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_embedding(embedding=embedding, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_embedding(embedding=embedding,
                                   embedding_labels=y_c1[:, 0],
                                   ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_embedding(embedding=embedding,
                                   idx_order=idx_order,
                                   ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_embedding(embedding=embedding,
                                   embedding_labels=y_d,
                                   ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)

    plt.close()

    with pytest.raises(ValueError):
        _ = cebra_plot.plot_embedding(embedding=embedding,
                                      idx_order=(10, 13, 15),
                                      ax=ax)
    with pytest.raises(ValueError):
        _ = cebra_plot.plot_embedding(embedding=embedding,
                                      idx_order=(10, 13),
                                      ax=ax)
    with pytest.raises(ValueError, match="Invalid.*embedding.*labels"):
        _ = cebra_plot.plot_embedding(embedding,
                                      embedding_labels=y_c2[:, 0],
                                      ax=ax)
    with pytest.raises(NotImplementedError, match="Invalid.*embedding.*labels"):
        _ = cebra_plot.plot_embedding(embedding,
                                      embedding_labels=y_c1[:, :1],
                                      ax=ax)

    ax = cebra_plot.plot(model, which="embedding", X=X, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot(model,
                         which="embedding",
                         X=X,
                         embedding_labels=y_c1[:, 0],
                         ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot(model,
                         which="embedding",
                         X=X,
                         embedding_labels=y_c1[:, 0],
                         ax=ax)

    with pytest.raises(ValueError, match="Invalid.*embedding.*labels"):
        _ = cebra_plot.plot(model,
                            which="embedding",
                            X=X,
                            embedding_labels=y_c2[:, 0],
                            ax=ax)
    with pytest.raises(NotImplementedError, match="Invalid.*embedding.*labels"):
        _ = cebra_plot.plot(model,
                            which="embedding",
                            X=X,
                            embedding_labels=y_c1[:, :1],
                            ax=ax)

    ax = cebra_plot.plot_embedding(embedding=embedding, idx_order=(0, 1))
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_embedding(embedding=embedding, idx_order=(0, 1, 1))
    assert isinstance(ax, matplotlib.axes.Axes)

    plt.close()


def test_plot_consistency():
    embedding1 = np.random.uniform(0, 1, (1000, 4))
    embedding2 = np.random.uniform(0, 1, (1000, 10))
    embedding3 = np.random.uniform(0, 1, (800, 6))
    embedding4 = np.random.uniform(0, 1, (500, 7))
    embeddings_datasets = [embedding1, embedding2, embedding3, embedding4]
    embeddings_runs = [embedding1, embedding2, embedding1, embedding2]

    labels1 = np.random.uniform(0, 1, (1000,))
    labels2 = np.random.uniform(0, 1, (1000,))
    labels3 = np.random.uniform(0, 1, (800,))
    labels4 = np.random.uniform(0, 1, (500,))
    labels_datasets = [labels1, labels2, labels3, labels4]

    dataset_ids = ["achilles", "buddy", "cicero", "gatsby"]

    figure = plt.figure(figsize=(5, 5))
    ax = figure.add_subplot()

    scores_subs, pairs_subs, datasets_subs = cebra_sklearn_metrics.consistency_score(
        embeddings_datasets,
        labels=labels_datasets,
        dataset_ids=dataset_ids,
        between="datasets",
    )
    scores_runs, pairs_runs, datasets_runs = cebra_sklearn_metrics.consistency_score(
        embeddings_runs, between="runs")

    # between datasets
    fig = cebra_plot.plot_consistency(scores_subs,
                                      pairs=pairs_subs,
                                      datasets=datasets_subs)
    assert isinstance(fig, matplotlib.axes.Axes)
    plt.close()
    ax = cebra_plot.plot_consistency(scores_subs,
                                     pairs=pairs_subs,
                                     datasets=datasets_subs,
                                     ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_consistency(
        torch.from_numpy(scores_subs),
        pairs=pairs_subs,
        datasets=datasets_subs,
        cmap="viridis",
        title="Test",
        text_color=None,
        colorbar_label=None,
        ax=ax,
    )
    assert isinstance(fig, matplotlib.axes.Axes)

    ax = cebra_plot.plot_consistency(torch.from_numpy(scores_subs),
                                     pairs=pairs_subs,
                                     datasets=datasets_subs,
                                     ax=ax)
    assert isinstance(fig, matplotlib.axes.Axes)

    ax = cebra_plot.plot_consistency(
        scores_subs.tolist(),
        pairs=pairs_subs.tolist(),
        datasets=datasets_subs.tolist(),
        ax=ax,
    )
    assert isinstance(fig, matplotlib.axes.Axes)

    with pytest.raises(ValueError, match="Missing.*datasets.*pairs"):
        _ = cebra_plot.plot_consistency(scores_subs, ax=ax)
    with pytest.raises(ValueError, match="Missing.*datasets.*pairs"):
        _ = cebra_plot.plot_consistency(scores_subs, pairs=pairs_subs, ax=ax)
    with pytest.raises(ValueError, match="Missing.*datasets.*pairs"):
        _ = cebra_plot.plot_consistency(scores_subs,
                                        datasets=datasets_subs,
                                        ax=ax)
    with pytest.raises(ValueError, match="Shape.*pairs"):
        _ = cebra_plot.plot_consistency(
            scores_subs,
            pairs=np.random.uniform(0, 1, (10, 2)),
            datasets=datasets_subs,
            ax=ax,
        )
    with pytest.raises(ValueError, match="Shape.*datasets"):
        _ = cebra_plot.plot_consistency(
            scores_subs,
            pairs=np.random.uniform(0, 1, (10, 2)),
            datasets=np.random.uniform(0, 1, (2,)),
            ax=ax,
        )
    with pytest.raises(ValueError, match="Invalid.*scores"):
        _ = cebra_plot.plot_consistency(
            np.random.uniform(0, 1, (12, 2, 2)),
            pairs=pairs_subs,
            datasets=datasets_subs,
            ax=ax,
        )

    # between runs
    fig = cebra_plot.plot_consistency(scores_runs,
                                      pairs=pairs_runs,
                                      datasets=datasets_runs)
    assert isinstance(fig, matplotlib.axes.Axes)
    plt.close()
    ax = cebra_plot.plot_consistency(scores_runs,
                                     pairs=pairs_runs,
                                     datasets=datasets_runs,
                                     ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_consistency(
        scores_runs,
        pairs=pairs_runs,
        datasets=datasets_runs,
        cmap="viridis",
        title="Test",
        text_color=None,
        colorbar_label=None,
        ax=ax,
    )
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_consistency(torch.from_numpy(scores_runs),
                                     pairs=pairs_runs,
                                     datasets=datasets_runs,
                                     ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax = cebra_plot.plot_consistency(
        scores_runs.tolist(),
        pairs=pairs_runs.tolist(),
        datasets=datasets_runs.tolist(),
        ax=ax,
    )
    assert isinstance(ax, matplotlib.axes.Axes)

    with pytest.raises(ValueError, match="Missing.*datasets.*pairs"):
        _ = cebra_plot.plot_consistency(scores_runs, ax=ax)
    with pytest.raises(ValueError, match="Missing.*datasets.*pairs"):
        _ = cebra_plot.plot_consistency(scores_runs, pairs=pairs_runs, ax=ax)
    with pytest.raises(ValueError, match="Missing.*datasets.*pairs"):
        _ = cebra_plot.plot_consistency(scores_runs,
                                        datasets=datasets_runs,
                                        ax=ax)
    with pytest.raises(ValueError, match="Shape.*datasets"):
        _ = cebra_plot.plot_consistency(
            scores_runs,
            pairs=np.random.uniform(0, 1, (10, 2)),
            datasets=np.random.uniform(0, 1, (4,)),
            ax=ax,
        )
    with pytest.raises(ValueError, match="Shape.*pairs"):
        _ = cebra_plot.plot_consistency(
            scores_runs,
            pairs=np.random.uniform(0, 1, (10, 2)),
            datasets=datasets_runs,
            ax=ax,
        )
    with pytest.raises(ValueError, match="Invalid.*dimensions"):
        _ = cebra_plot.plot_consistency(
            np.random.uniform(0, 1, (12, 2, 2)),
            pairs=pairs_runs,
            datasets=datasets_runs,
            ax=ax,
        )
    plt.close()


@pytest.mark.parametrize("seed", [None, 42, 1024, 454545])
def test_check_consistency_data(seed):
    """check the helper functions that generates the dataset for tests below."""
    if seed is None:
        data1 = _ConsistencyDataset()
        data2 = _ConsistencyDataset()
    else:
        data1 = _ConsistencyDataset(seed=seed)
        data2 = _ConsistencyDataset(seed=seed)

    assert all(
        np.allclose(a, b) for a, b in zip(data1.embeddings, data2.embeddings))
    assert all(np.allclose(a, b) for a, b in zip(data1.labels, data2.labels))
    assert all(a == b for a, b in zip(data1.dataset_ids, data2.dataset_ids))


@pytest.mark.parametrize("permutation", itertools.permutations([0, 1, 2]))
def test_consistency_permutations(permutation):
    dataset = _ConsistencyDataset()

    def compute_consistency(kwargs):
        scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
            between="datasets", **kwargs)
        return {tuple(pair): score for pair, score in zip(pairs, scores)}

    assert compute_consistency(dataset[0, 1,
                                       2]) != compute_consistency(dataset[1, 1,
                                                                          0])
    assert compute_consistency(dataset[0, 1, 2]) == compute_consistency(
        dataset[permutation])


@pytest.mark.parametrize("seed,permutation,pass_original_labels",
                         [(42, (0, 1, 2), False), (43, (0, 2, 1), False),
                          (44, (0, 2, 1), True), (45, (1, 2, 0), False)])
def test_plot_consistency_unordered_labels(seed, permutation,
                                           pass_original_labels):

    dataset = _ConsistencyDataset(seed=seed)

    scores, pairs, datasets = cebra_sklearn_metrics.consistency_score(
        between="datasets", **dataset[permutation])
    scores_dict = {tuple(pair): score for pair, score in zip(pairs, scores)}

    plot = cebra_plot._ConsistencyPlot(
        scores=scores,
        pairs=pairs,
        datasets=dataset[permutation]["dataset_ids"]
        if pass_original_labels else datasets,
        cmap="cebra",
        text_color="black",
        colorbar_label=None,
        title="testing",
        axis=plt.gca(),
        figsize=(3, 3),
        dpi=200)

    for i, label_i in enumerate(plot.labels):
        for j, label_j in enumerate(plot.labels):
            if i == j:
                # these values are nan, just check for that
                assert np.isnan(plot.scores[i, j])
            else:
                assert np.isclose(100 * scores_dict[label_i, label_j],
                                  plot.scores[i, j])
