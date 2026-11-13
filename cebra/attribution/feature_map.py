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
"""Feature attribution maps: how each input channel drives each feature, through the embedding.

For one trained model:

1. ``W = fit_feature_readout(embedding, features, train_fraction=0.9)``: linear readout ``(D, F)`` from the
   embedding to the features.
2. ``M = feature_attribution_map(solver, inputs, W)``: the signed map ``M = J^T W`` of shape
   ``(N, F)``, with ``J = dz/dx`` the forward Jacobian of the encoder, averaged over time. The
   solver must have been trained with the Jacobian regularizer.
3. ``average_feature_maps([M_1, M_2, ...])``: mean over models.

Fit ``W`` for each model separately, on that model's own embedding: the embeddings of different
models are not aligned, so a ``W`` is only valid for the model it was fitted on.

``N`` is the number of input channels (summed over sessions for a unified model), ``D`` the
embedding dimension and ``F`` the number of features. The map is signed: no absolute value and no
pseudo-inverse.

Valid rows: a model with offset ``(left, right)`` gives one output per window of ``left + right``
input samples, so row ``r`` of the embedding and of the per-row maps is input sample ``left + r``:
features ``features[left:T - right + 1]`` match the rows of a session of ``T`` samples.
"""

import warnings

import numpy as np
import torch

import cebra.attribution
import cebra.models
from cebra.solver.unified_session import UnifiedSolverxCEBRA

__all__ = [
    "fit_feature_readout",
    "feature_attribution_map",
    "average_feature_maps",
]


def _train_rows(n_rows: int, train_fraction, train_index) -> np.ndarray:
    """Integer row indices of the readout's training subset (see :func:`fit_feature_readout`)."""
    if train_fraction is not None and train_index is not None:
        raise ValueError("Give train_fraction or train_index, not both.")
    if train_fraction is None and train_index is None:
        train_fraction = 0.8
        print(
            "fit_feature_readout: fitting on the first 80% of rows (train_fraction=0.8); "
            "pass train_fraction or train_index to choose.")
    if train_fraction is not None:
        if not 0 < float(train_fraction) <= 1:
            raise ValueError(
                f"train_fraction must be in (0, 1], got {train_fraction}.")
        rows = np.arange(int(round(n_rows * float(train_fraction))))
    else:
        index = np.asarray(train_index)
        if index.dtype == bool:
            if index.shape != (n_rows,):
                raise ValueError(
                    f"A boolean train_index must have shape ({n_rows},), got {index.shape}."
                )
            rows = np.flatnonzero(index)
        else:
            if index.ndim != 1 or (index.size and
                                   not np.issubdtype(index.dtype, np.integer)):
                raise ValueError(
                    "train_index must be a 1-d array of integer rows or a boolean mask."
                )
            if index.size and (index.min() < -n_rows or index.max() >= n_rows):
                raise ValueError(
                    f"train_index is out of range for {n_rows} rows.")
            rows = np.arange(n_rows)[index.astype(int)]
    if rows.size == 0:
        raise ValueError("The readout training subset is empty.")
    if np.unique(rows).size == n_rows:
        warnings.warn(
            "The readout is fitted on every row passed, so no row is held out. Pass only "
            "training data, or hold rows out with train_fraction < 1 or a train_index that "
            "leaves rows out.",
            UserWarning,
            stacklevel=3)
    return rows


def fit_feature_readout(
    embedding,
    features,
    alpha: float = 1.0,
    train_fraction=None,
    train_index=None,
) -> np.ndarray:
    """Fit a ridge readout from the embedding to the features.

    The rows of ``embedding`` and ``features`` must be aligned 1:1 (see the module notes for the
    valid rows). The readout is fitted on the first 80% of the rows by default, or on the rows
    chosen by ``train_fraction`` (the first rows, a contiguous block) or ``train_index`` (any
    rows), so the other rows stay free for a held-out check. Selecting every row is only right
    if the rows passed are all training data; it gives a warning.

    Discrete (categorical) features must be one-hot encoded, one column per category: integer
    labels would be fitted as one continuous target.

    Args:
        embedding: Array of shape ``(T, D)``.
        features: Array of shape ``(T, F)``, or ``(T,)`` for one feature. Used as given; z-score
            them first to put features on a comparable scale.
        alpha: Ridge penalty.
        train_fraction: Fit on the first ``round(T * train_fraction)`` rows, in ``(0, 1]``.
            Default (neither argument given): 0.8, with a message printed.
        train_index: Fit on these rows, as integer indices or a boolean mask of length ``T``.
            Not together with ``train_fraction``.

    Returns:
        The readout ``W`` of shape ``(D, F)``, such that ``embedding @ W`` approximates
        ``features`` (the intercept is dropped).

    Raises:
        ValueError: If both ``train_fraction`` and ``train_index`` are given, if the training
            rows are empty, or if the shapes do not match.

    Warns:
        UserWarning: If the selected rows are all the rows passed (no row held out).

    Example:
        >>> import numpy as np
        >>> from cebra.attribution import fit_feature_readout
        >>> rng = np.random.default_rng(0)
        >>> embedding = rng.normal(size=(100, 4))
        >>> features = embedding @ rng.normal(size=(4, 2))
        >>> W = fit_feature_readout(embedding, features, train_fraction=0.9)
        >>> W.shape
        (4, 2)
        >>> W = fit_feature_readout(embedding, features)  # first 80% of the rows
        fit_feature_readout: fitting on the first 80% of rows (train_fraction=0.8); pass train_fraction or train_index to choose.
        >>> held_out = np.corrcoef((embedding[90:] @ W)[:, 0], features[90:, 0])[0, 1]
        >>> bool(held_out > 0.99)
        True
    """
    from sklearn.linear_model import Ridge

    embedding = np.asarray(embedding, dtype=float)
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    if embedding.ndim != 2 or features.ndim != 2:
        raise ValueError(
            f"embedding and features must be 2-d, got {embedding.shape} and {features.shape}."
        )
    if embedding.shape[0] != features.shape[0]:
        raise ValueError(
            f"embedding and features must have the same rows, got {embedding.shape[0]} and "
            f"{features.shape[0]}.")
    rows = _train_rows(embedding.shape[0], train_fraction, train_index)

    readout = Ridge(alpha=float(alpha)).fit(embedding[rows], features[rows])
    coef = np.asarray(readout.coef_).reshape(features.shape[1],
                                             embedding.shape[1])
    return coef.T  # (D, F), also for a single feature


def _window(model) -> tuple:
    """(left, right) of the model window, as CEBRA applies it to the Jacobian input.

    CEBRA's Jacobian only windows the input for a ``ConvolutionalModelMixin``. Every other model is
    evaluated sample by sample, i.e. ``(left, right) = (0, 1)``.
    """
    if isinstance(model, cebra.models.ConvolutionalModelMixin):
        offset = model.get_offset()
        return int(offset.left), int(offset.right)
    return 0, 1


def _num_valid_rows(model, n_samples: int) -> int:
    """Number of valid rows (see the module notes) of the model on ``n_samples`` inputs."""
    left, right = _window(model)
    n_valid = int(n_samples) - (left + right) + 1
    if n_valid < 1:
        raise ValueError(
            f"{n_samples} input samples are too few for the model window ({left + right}).")
    return n_valid


def _is_cuda_oom(exc: Exception) -> bool:
    """Whether ``exc`` is a CUDA out-of-memory error."""
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _check_readout(W, n_outputs: int) -> np.ndarray:
    """Readout as a float array of shape ``(n_outputs, F)``, or a clear error."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError(
            f"The readout W must be 2-d (outputs, features), got shape {W.shape}."
        )
    if W.shape[0] != n_outputs:
        raise ValueError(
            f"The readout W has {W.shape[0]} rows, but the attributed model outputs are {n_outputs} "
            f"(the model output dimension, or len(output_slice)). Fit W on the same embedding columns."
        )
    return W


#: Attribution methods that return CEBRA's Jacobian ``jf``, usable as ``method``.
JACOBIAN_METHODS = (
    "jacobian-based-batched-optimized",
    "jacobian-based-batched",
    "jacobian-based",
)


def _check_method(method: str):
    """Raise if ``method`` is not one of :data:`JACOBIAN_METHODS`."""
    if method not in JACOBIAN_METHODS:
        raise ValueError(
            f"method must be one of {JACOBIAN_METHODS}, got {method!r}.")


def _compute_jf(method, model, inputs, batch_size, latent_batch_size,
                return_full):
    """CEBRA's Jacobian ``jf`` with attribution ``method``: ``(T, D, N)``, or its ``(D, N)`` mean.

    ``inputs`` must require gradients. ``latent_batch_size`` is only used by
    ``jacobian-based-batched-optimized``, which also averages without storing every sample.
    """
    _check_method(method)
    attribution = cebra.attribution.init(
        method,
        model=model,
        input_data=inputs,
        output_dimension=int(model.num_output),
    )
    if method == "jacobian-based-batched-optimized":
        result = attribution.compute_attribution_map(
            batch_size=batch_size,
            return_full=return_full,
            selected_keys=["jf"],
            latent_batch_size=latent_batch_size,
        )
        return np.asarray(result["jf"], dtype=np.float64)
    if method == "jacobian-based-batched":
        result = attribution.compute_attribution_map(batch_size=batch_size,
                                                     selected_keys=["jf"])
    else:
        result = attribution.compute_attribution_map(selected_keys=["jf"])
    jf = np.asarray(result["jf"], dtype=np.float64)
    return jf if return_full else jf.mean(axis=0)


def _row_feature_maps(model, inputs, start, stop, W, output_slice,
                      latent_batch_size, method):
    """Per-row maps ``J_t[output_slice]^T W`` for valid rows ``start`` to ``stop - 1``.

    The input samples ``start`` to ``stop + left + right - 2`` form the windows of these rows, so
    CEBRA's Jacobian (``jf``: the window Jacobian averaged over the offset, one entry per row) is
    computed on that slice only. Returns ``(stop - start, N, F)`` float64.
    """
    left, right = _window(model)
    device = next(model.parameters()).device
    window = (inputs[start:stop + left + right - 1].to(device=device,
                                                       copy=True).detach())
    window.requires_grad_(True)
    jac = _compute_jf(method,
                      model,
                      window,
                      batch_size=stop - start,
                      latent_batch_size=latent_batch_size,
                      return_full=True)  # (rows, D, N)
    if output_slice is not None:
        jac = jac[:, output_slice, :]
    return np.matmul(jac.transpose(0, 2, 1), W)  # (rows, N, F)


def _feature_maps(model, inputs, W, output_slice, batch_size, latent_batch_size,
                  average, method):
    """Shared implementation of the per-row and time-mean maps.

    Rows are processed in blocks of ``batch_size``; each block is multiplied by ``W`` at once, so
    the full Jacobian over time is never stored. On CUDA out-of-memory errors the block size is
    halved (down to 1, then the error is raised).
    """
    inputs = torch.as_tensor(inputs)
    n_valid = _num_valid_rows(model, inputs.shape[0])
    device = next(model.parameters()).device
    step = min(max(1, int(batch_size)), n_valid)

    blocks, total, start = [], None, 0
    # A model trained with several losses returns one output per loss (split_outputs);
    # the Jacobian needs the single concatenated output.
    previous = getattr(model, "split_outputs", None)
    if previous is not None:
        model.split_outputs = False
    # The Jacobian is taken in eval mode (no dropout, fixed normalization statistics).
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        while start < n_valid:
            stop = min(start + step, n_valid)
            try:
                block = _row_feature_maps(model, inputs, start, stop, W,
                                          output_slice, latent_batch_size,
                                          method)
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or step <= 1:
                    raise
                step = max(1, step // 2)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            if average:
                total = (block.sum(axis=0) if total is None else total +
                         block.sum(axis=0))
            else:
                blocks.append(block)
            start = stop
    finally:
        for module, training in modes:
            module.training = training
        if previous is not None:
            model.split_outputs = previous

    if average:
        return total / n_valid
    return np.concatenate(blocks, axis=0)


def average_feature_maps(maps) -> np.ndarray:
    """Mean of feature attribution maps, e.g. over models trained with different seeds.

    Average the signed maps: the mean of ``|M|`` is not ``|mean M|``. Each map must come from
    its own model with a ``W`` fitted on that model's own embedding (the embeddings of different
    models are not aligned): never share a ``W`` across models.

    Args:
        maps: Sequence of arrays of the same shape, e.g. ``(N, F)`` per model.

    Returns:
        Array with the shape of one map.

    Example:
        >>> import numpy as np
        >>> from cebra.attribution import average_feature_maps
        >>> maps = [np.ones((5, 2)), 3 * np.ones((5, 2))]
        >>> average_feature_maps(maps).shape
        (5, 2)
        >>> float(average_feature_maps(maps)[0, 0])
        2.0
    """
    maps = [np.asarray(m) for m in maps]
    if not maps:
        raise ValueError("average_feature_maps needs at least one map.")
    return np.mean(np.stack(maps), axis=0)


def _check_regularized(solver):
    """Raise unless ``solver`` was set up with the Jacobian regularizer."""
    if isinstance(solver, cebra.CEBRA):
        raise TypeError(
            "A cebra.CEBRA object is not accepted: the sklearn API does not train regularized "
            "models. Train with a regularized solver, e.g. 'unified-xcebra' or the xCEBRA "
            "solvers.")
    if (getattr(solver, "regularizer", None) is None and
            getattr(solver, "jac_regularizer", None) is None):
        raise TypeError(
            f"feature_attribution_map needs a solver trained with the Jacobian regularizer "
            f"(e.g. 'unified-xcebra', an xCEBRA solver or RegularizedSolver), got "
            f"{type(solver).__name__}.")


def feature_attribution_map(
    solver,
    inputs,
    W,
    *,
    labels=None,
    session_id=None,
    indices=None,
    output_slice=None,
    average: bool = True,
    random_ties: bool = False,
    seed=0,
    batch_size: int = 64,
    latent_batch_size: int = 32,
    method: str = "jacobian-based-batched-optimized",
) -> np.ndarray:
    """Signed feature attribution map ``M = J^T W`` of one model trained with the Jacobian regularizer.

    ``M[n, f] = sum_d dz_d/dx_n * W[d, f]``, with ``J`` the forward Jacobian of the embedding and
    ``W`` the readout from :func:`fit_feature_readout`. A positive value means that raising
    channel ``n`` raises the readout of feature ``f``. ``J(t)`` is the Jacobian of ``z(t)`` with
    respect to the whole input window of the model, averaged over the window (CEBRA's ``jf``);
    the mean map is the mean of the per-timepoint maps.

    Fit ``W`` for each model separately, on that model's own embedding (embeddings of different
    models are not aligned, so a ``W`` is only valid for its model); never share a ``W`` across
    models. For a unified solver, ``inputs`` and ``labels`` are per-session lists, as in
    ``solver.transform``, and the map has one row per channel of all sessions concatenated. Use
    the same ``random_ties`` and ``seed`` as the ``transform`` that gave the embedding ``W`` was
    fitted on.

    Args:
        solver: Solver trained with the Jacobian regularizer: a unified solver (e.g.
            ``"unified-xcebra"``) or a single-session one (an xCEBRA solver or
            :py:class:`~cebra.solver.regularized.RegularizedSolver`). Plain modules, unregularized
            solvers and :py:class:`cebra.CEBRA` objects raise a ``TypeError``.
        inputs: ``(T, N)`` array or tensor, or for a unified solver a list of ``(T_s, N_s)``.
        W: Readout of shape ``(D, F)``, with ``D`` the output dimension (or
            ``len(output_slice)``).
        labels: Unified only, required. Auxiliary variables of each session, ``(T_s, d)``.
        session_id: Unified only, required. Reference session.
        indices: Rows of the (reference) session's inputs at which the map is computed. For a
            unified solver the other sessions are aligned to these rows. Default: all.
        output_slice: Single session only. Embedding columns of one head, e.g. one subspace of a
            :py:class:`~cebra.models.SubspaceMultiobjectiveModel`.
        average: ``True`` for the time mean ``(N, F)``, ``False`` for one map per valid row
            ``(T_valid, N, F)`` (see the module notes).
        random_ties: Unified only. Break nearest-label ties at random instead of taking the
            first.
        seed: Unified only. Seed of the tie draws (default 0). Use the same seed as the
            ``transform`` that gave the embedding ``W`` was fitted on.
        batch_size: Rows per Jacobian batch, halved on CUDA out-of-memory.
        latent_batch_size: Embedding dimensions per autograd pass. Only used by the default
            ``method``.
        method: Attribution method that computes the Jacobian, one of
            :data:`JACOBIAN_METHODS`.

    Returns:
        Array, float64: ``(N, F)`` for ``average=True``, ``(T_valid, N, F)`` for ``average=False``.

    Raises:
        TypeError: If ``solver`` was not trained with the Jacobian regularizer.
        ValueError: If ``W`` does not match the output dimension, if the input is shorter than
            the model window, if ``labels`` or ``session_id`` is missing for a unified solver,
            if unified-only arguments are given for a single-session solver, or if ``method`` is
            not a Jacobian method.

    Example:
        Readout, map, then the average over models, on small random data:

        >>> import numpy as np
        >>> import torch
        >>> import cebra
        >>> from cebra.attribution import (average_feature_maps,
        ...                                feature_attribution_map,
        ...                                fit_feature_readout)
        >>> rng = np.random.default_rng(0)
        >>> x = torch.tensor(rng.normal(size=(100, 10)), dtype=torch.float32)
        >>> features = rng.normal(size=(100, 2))
        >>> maps = []
        >>> for seed in range(3):  # one model per seed; here untrained encoders
        ...     _ = torch.manual_seed(seed)
        ...     model = cebra.models.init("offset1-model", num_neurons=10, num_units=16,
        ...                               num_output=4)
        ...     solver = cebra.solver.RegularizedSolver(
        ...         model=model, criterion=cebra.models.InfoNCE(),
        ...         optimizer=torch.optim.Adam(model.parameters()))
        ...     embedding = model(x).detach().numpy()
        ...     W = fit_feature_readout(embedding, features, train_fraction=0.9)
        ...     maps.append(feature_attribution_map(solver, x, W))
        >>> maps[0].shape
        (10, 2)
        >>> average_feature_maps(maps).shape
        (10, 2)

        For a unified model, pass the per-session data and labels, with the same ``random_ties``
        and ``seed`` as in the ``transform`` that produced the embedding::

            z = solver.transform(inputs, labels, session_id=0, random_ties=True, seed=0)
            W = fit_feature_readout(z, features, train_fraction=0.9)
            M = feature_attribution_map(solver, inputs, W, labels=labels, session_id=0,
                                        random_ties=True, seed=0)
    """
    _check_regularized(solver)
    _check_method(method)
    model = solver.model

    if isinstance(solver, UnifiedSolverxCEBRA):
        if labels is None:
            raise ValueError(
                "labels are required for a unified solver (a list of per-session "
                "auxiliary variables, as in solver.transform).")
        if session_id is None:
            raise ValueError(
                "session_id is required for a unified solver (the reference session)."
            )
        if output_slice is not None:
            raise ValueError(
                "output_slice is not supported for a unified solver.")
        W = _check_readout(W, int(model.num_output))
        aligned = solver.aligned_inputs(inputs,
                                        labels,
                                        session_id,
                                        indices=indices,
                                        random_ties=random_ties,
                                        seed=seed)
        return _feature_maps(model, aligned, W, None, batch_size,
                             latent_batch_size, average, method)

    if labels is not None or session_id is not None or random_ties:
        raise ValueError(
            "labels, session_id and random_ties are only for unified solvers."
        )
    if isinstance(inputs, (list, tuple)):
        raise ValueError(
            "A single-session model needs inputs of shape (T, N), not a list.")
    neural = torch.as_tensor(inputs, dtype=torch.float32)
    if neural.ndim != 2:
        raise ValueError(
            f"inputs must have shape (T, N), got {tuple(neural.shape)}.")
    if indices is not None:
        neural = neural[torch.as_tensor(indices, dtype=torch.long)]
    n_outputs = int(model.num_output)
    if output_slice is not None:
        n_outputs = len(range(n_outputs)[output_slice])
    W = _check_readout(W, n_outputs)
    return _feature_maps(model, neural, W, output_slice, batch_size,
                         latent_batch_size, average, method)
