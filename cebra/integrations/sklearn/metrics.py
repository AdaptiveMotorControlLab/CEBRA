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
from typing import Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import sklearn.linear_model
import sklearn.utils.validation as sklearn_utils_validation
import torch

import cebra.integrations.sklearn.cebra as cebra_sklearn_cebra
import cebra.integrations.sklearn.helpers as cebra_sklearn_helpers


def infonce_loss(
    cebra_model: cebra_sklearn_cebra.CEBRA,
    X: Union[npt.NDArray, torch.Tensor],
    *y,
    session_id: Optional[int] = None,
    num_batches: int = 500,
    correct_by_batchsize: bool = False,
) -> float:
    """Compute the InfoNCE loss on a *single session* dataset on the model.

    Args:
        cebra_model: The model to use to compute the InfoNCE loss on the samples.
        X: A 2D data matrix, corresponding to a *single session* recording.
        y: An arbitrary amount of continuous indices passed as 2D matrices, and up to one
            discrete index passed as a 1D array. Each index has to match the length of ``X``.
        session_id: The session ID, an :py:class:`int` between 0 and :py:attr:`cebra.CEBRA.num_sessions`
            for multisession, set to ``None`` for single session.
        num_batches: The number of iterations to consider to evaluate the model on the new data.
            Higher values will give a more accurate estimate. Set it to at least 500 iterations.
        correct_by_batchsize: If True, the loss is corrected by the batch size.

    Returns:
        The average InfoNCE loss estimated over ``num_batches`` batches from the data distribution.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> neural_data = np.random.uniform(0, 1, (1000, 20))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(neural_data)
        CEBRA(max_iterations=10)
        >>> loss = cebra.sklearn.metrics.infonce_loss(cebra_model,
        ...                                           neural_data,
        ...                                           num_batches=5)

    """
    sklearn_utils_validation.check_is_fitted(cebra_model, "n_features_")

    # score is computed on single session dataset only
    if isinstance(X, list) and isinstance(X[0], Iterable) and len(
            X[0].shape) == 2:
        raise NotImplementedError(
            f"Data invalid: score cannot be computed on multiple sessions,"
            f"got {len(X)} sessions.")
    if (isinstance(y, tuple) and len(y) > 0 and isinstance(y[0], list) and
            isinstance(y[0][0], Iterable) and len(y[0][0].shape) == 2):
        raise NotImplementedError(
            f"Labels invalid: score cannot be computed on multiple sessions,"
            f"got {len(y[0])} sessions.")

    model, _ = cebra_model._select_model(
        X, session_id)  # check session_id validity and corresponding model
    cebra_model._check_labels_types(y, session_id=session_id)

    dataset, is_multisession = cebra_model._prepare_data(X, y)  # single session
    loader, _ = cebra_model._prepare_loader(
        dataset,
        max_iterations=num_batches,
        is_multisession=is_multisession,
    )

    cebra_model._configure_for_all(dataset, model, is_multisession)

    solver = cebra_model.solver_
    solver.to(cebra_model.device_)
    avg_loss = solver.validation(loader=loader, session_id=session_id)
    if correct_by_batchsize:
        if cebra_model.batch_size is None:
            raise ValueError(
                "Batch size is None, please provide a model with a batch size to correct the InfoNCE."
            )
        else:
            avg_loss = avg_loss - np.log(cebra_model.batch_size)
    return avg_loss


def _consistency_scores(
    embeddings: List[Union[npt.NDArray, torch.Tensor]],
    datasets: List[Union[int, str]],
) -> Tuple[List[float], List[tuple]]:
    """Compute consistency scores and dataset pairs for a list of embeddings.

    A linear regression is fitted between 2 embeddings and the resulting R2 score is the
    consistency between both embeddings.

    Args:
        embeddings: List of embedding matrices.
        dataset_ids: List of dataset ID associated to each embedding. Multiple embeddings can be
            associated to the same dataset.

    Returns:
        List of the consistencies for each embeddings pair (first element) and
        list of dataset pairs corresponding to the scores (second element).
    """
    if len(embeddings) <= 1:
        raise ValueError(
            f"Invalid list of embeddings, provide at least 2 embeddings to compare, got {len(embeddings)}."
        )
    if datasets is None:
        raise ValueError(
            "Missing datasets, provide a list of datasets_id associated to each embeddings to compare."
        )

    scores = []
    pairs = []

    lin = sklearn.linear_model.LinearRegression()
    for n, i in enumerate(embeddings):
        for m, j in enumerate(embeddings):
            if n != m:
                if isinstance(i, torch.Tensor):
                    i = i.numpy()
                if isinstance(j, torch.Tensor):
                    j = j.numpy()
                scores.append(lin.fit(i, j).score(i, j))
                pairs.append((datasets[n], datasets[m]))
    return scores, pairs


def _consistency_datasets(
    embeddings: List[Union[npt.NDArray, torch.Tensor]],
    dataset_ids: Optional[List[Union[int, str, float]]],
    labels: List[Union[npt.NDArray, torch.Tensor]],
    num_discretization_bins: int = 100
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute consistency between embeddings from different datasets.

    To compute the consistency between datasets, the embeddings are aligned on a
    set of labels.

    If `dataset_ids` is None, then the embeddings are considered to be all coming
    from different datasets, so that one-to-one comparison between all provided embeddings is
    performed.

    Args:
        embeddings: List of embedding matrices.
        dataset_ids: List of dataset ID associated to each embedding. Multiple embeddings can be
            associated to the same dataset.
        labels: List of labels corresponding to each embedding and to use for alignment
            between them.
        num_discretization_bins: Number of values for the digitalized common labels. The discretized labels are used
            for embedding alignment. Also see the ``n_bins`` argument in
            :py:mod:`cebra.integrations.sklearn.helpers.align_embeddings` for more information on how this
            parameter is used internally. This argument is only used if ``labels``
            is not ``None`` and the given labels are continuous and not already discrete.

    Returns:
        A list of scores obtained between embeddings from different datasets (first element),
        a list of pairs of IDs corresponding to the scores (second element), and a list of the
        dataset IDs (third element).

    """
    if labels is None:
        raise ValueError(
            "Missing labels, computing consistency between datasets requires labels, expect "
            f"a set of labels for each embedding.")
    if len(embeddings) != len(labels):
        raise ValueError(
            "Invalid set of labels, computing consistency between datasets requires labels, "
            f"expect one set of labels for each embedding, got {len(embeddings)} embeddings "
            f"and {len(labels)} set of labels")
    for idx in range(len(labels)):
        if not isinstance(labels[idx], np.ndarray):
            if isinstance(labels[idx], list):
                labels[idx] = np.array(labels[idx])
            elif isinstance(labels[idx], torch.Tensor):
                labels[idx] = labels[idx].numpy()
            else:
                raise ValueError(
                    f"Invalid labels, expect np.array, torch.tensor or list, got {type(labels[idx])}."
                )
        if labels[idx].ndim > 1:
            raise NotImplementedError(
                f"Invalid label dimensions, expect 1D labels only, got {labels[idx].ndim}D."
            )

    # if no datasets IDs then all embeddings are considered as coming from different datasets
    if dataset_ids is None:
        dataset_ids = np.arange(len(embeddings))
    datasets = np.array(sorted(set(dataset_ids)))
    if len(datasets) <= 1:
        raise ValueError(
            "Invalid number of dataset_ids, expect more than one dataset to perform the comparison, "
            f"got {len(datasets)}")

    # NOTE(celia): with default values normalized=True and n_bins = 100
    aligned_embeddings = cebra_sklearn_helpers.align_embeddings(
        embeddings, labels, n_bins=num_discretization_bins)
    scores, pairs = _consistency_scores(aligned_embeddings,
                                        datasets=dataset_ids)
    between_dataset = [p[0] != p[1] for p in pairs]

    pairs = np.array(pairs)[between_dataset]
    scores = _average_scores(np.array(scores)[between_dataset], pairs)

    return (scores, pairs, np.array(dataset_ids))


def _average_scores(scores: Union[npt.NDArray, list], pairs: Union[npt.NDArray,
                                                                   list]):
    """Average scores across similar comparisons either between datasets or between runs.

    Args:
        scores: The list of scores computed between the embeddings.
        pairs: The list of pairs corresponding to each computed score.

    Returns:
        A :py:func:`numpy.array` with scores averaged across similar comparisons.
    """
    avg_scores = {}
    for score, pair in zip(scores, pairs):
        key = f"{pair[0]}-{pair[1]}"
        if key in avg_scores.keys():
            avg_scores[key].append(score)
        else:
            avg_scores[key] = [score]

    for key in avg_scores.keys():
        avg_scores[key] = sum(avg_scores[key]) / len(avg_scores[key])
    return np.array(list(avg_scores.values()))


def _consistency_runs(
    embeddings: List[Union[npt.NDArray, torch.Tensor]],
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute consistency between embeddings coming from the same dataset.

    Args:
        embeddings: List of embedding matrices.

    Returns:
        A list of lists of scores obtained between embeddings of the same dataset (first element),
        a list of lists of pairs of ids of the embeddings of the same datasets that were compared
        (second element), they are identified with :py:class:`numpy.int` from 0 to the number of
        embeddings for the dataset, and a list of the unique IDs (third element).
    """
    # NOTE(celia): The number of samples of the embeddings should be the same for all as there is
    # no realignment, the number of output dimensions can vary between the embeddings we compare.
    if not all(embeddings[0].shape[0] == embeddings[i].shape[0]
               for i in range(1, len(embeddings))):
        raise ValueError(
            f"Invalid embeddings, all embeddings should be the same shape to be compared in a between-runs way."
            f"If your embeddings are coming from different models, you can use between-datasets"
        )

    run_ids = np.arange(len(embeddings))
    scores, pairs = _consistency_scores(embeddings=embeddings, datasets=run_ids)

    return (
        _average_scores(scores, pairs),
        np.array(pairs),
        np.array(run_ids),
    )


def consistency_score(
    embeddings: List[Union[npt.NDArray, torch.Tensor]],
    between: Optional[Literal["datasets", "runs"]] = None,
    labels: Optional[List[Union[npt.NDArray, torch.Tensor]]] = None,
    dataset_ids: Optional[List[Union[int, str, float]]] = None,
    num_discretization_bins: int = 100
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute the consistency score between embeddings, either between runs or between datasets.

    Args:
        embeddings: List of embedding matrices.
        labels: List of labels corresponding to each embedding and to use for alignment
            between them. They are only required for a between-datasets comparison.
        dataset_ids: List of dataset ID associated to each embedding. Multiple embeddings can be
            associated to the same dataset. In both modes (``runs`` or ``datasets``), if no ``dataset_ids`` is
            provided, then all the provided embeddings are compared one-to-one. Internally, the function will
            consider that the embeddings are all different runs from the same dataset for between-runs mode and
            on the contrary, that they are all computed on different dataset in the between-datasets mode.
        between: A string describing the type of comparison to perform between the embeddings, either
            between ``all`` embeddings or between ``datasets`` or ``runs``.
            *Consistency between runs* means the consistency between embeddings obtained from multiple models
            trained on the **same dataset**. *Consistency between datasets* means the consistency between embeddings
            obtained from models trained on **different datasets**, such as different animals, sessions, etc.
        num_discretization_bins: Number of values for the digitalized common labels. The discretized labels are used
            for embedding alignment. Also see the ``n_bins`` argument in
            :py:mod:`cebra.integrations.sklearn.helpers.align_embeddings` for more information on how this
            parameter is used internally. This argument is only used if ``labels``
            is not ``None``, alignment between datasets is used (``between = "datasets"``), and the given labels
            are continuous and not already discrete.

    Returns:
        The list of scores computed between the embeddings (first returns), the list of pairs corresponding
        to each computed score (second returns) and the list of id of the entities present in the comparison,
        either different datasets in the between-datasets comparison or runs in the between-runs comparison
        (third returns).

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> embedding1 = np.random.uniform(0, 1, (1000, 5))
        >>> embedding2 = np.random.uniform(0, 1, (1000, 8))
        >>> labels1 = np.random.uniform(0, 1, (1000, ))
        >>> labels2 = np.random.uniform(0, 1, (1000, ))
        >>> # Between-runs consistency
        >>> scores, pairs, ids_runs = cebra.sklearn.metrics.consistency_score(embeddings=[embedding1, embedding2],
        ...                                                                   between="runs")
        >>> # Between-datasets consistency, by aligning on the labels
        >>> scores, pairs, ids_datasets = cebra.sklearn.metrics.consistency_score(embeddings=[embedding1, embedding2],
        ...                                                                   labels=[labels1, labels2],
        ...                                                                   dataset_ids=["achilles", "buddy"],
        ...                                                                   between="datasets")

    """
    if len(embeddings) <= 1:
        raise ValueError(
            f"Invalid number of embeddings for dataset, expect at least 2 embeddings "
            f"to be able to compare them, got {len(embeddings)}")

    if between is None:
        raise ValueError(
            'Missing between parameter, provide the type of comparison to run, either "datasets" or "runs".'
        )
    if between == "runs":
        if labels is not None:
            raise ValueError(
                f"No labels should be provided for between-runs consistency.")
        if dataset_ids is not None:
            raise ValueError(
                f"No dataset ID should be provided for between-runs consistency."
                f"All embeddings should be computed on the same dataset.")
        scores, pairs, ids = _consistency_runs(embeddings=embeddings,)
    elif between == "datasets":
        scores, pairs, ids = _consistency_datasets(
            embeddings=embeddings,
            dataset_ids=dataset_ids,
            labels=labels,
            num_discretization_bins=num_discretization_bins)
    else:
        raise NotImplementedError(
            f"Invalid comparison, got between={between}, expects either datasets or runs."
        )
    return scores.squeeze(), pairs.squeeze(), ids
