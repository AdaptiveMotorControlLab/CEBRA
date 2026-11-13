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
"""Unified session solver for multi-session contrastive learning.

We added support for training contrastive models on unified-session datasets.
This allows users to align and embed multiple sessions into a common latent
space using a single shared model.

This module implements the :py:class:`~cebra.solver.unified.UnifiedSolver`, which
is designed for training a single embedding model across multiple recording sessions.
Unlike the standard multi-session solvers, the unified session approach uses
a global model that requires session-specific information for sampling but maintains
a shared representation across all data.

Features:
- Single model inference across all sessions.
- Batched transform.
- Compatibility with :py:class:`~cebra.data.UnifiedDataset` and :py:class:`~cebra.data.UnifiedLoader`.

See Also:
    :py:class:`~cebra.solver.base.Solver`
    :py:class:`~cebra.data.UnifiedDataset`
    :py:class:`~cebra.data.UnifiedLoader`
"""

from typing import List, Optional, Union

import literate_dataclasses as dataclasses
import numpy as np
import torch

import cebra
import cebra.data
import cebra.distributions
import cebra.models
import cebra.solver.base as abc_
from cebra.solver import register
from cebra.solver.multiobjective import MultiobjectiveSolverBase


def _check_sessions(inputs, labels, session_id):
    """Check that inputs and labels have the same sessions and that ``session_id`` exists."""
    num_sessions = len(inputs)
    if len(labels) != num_sessions:
        raise ValueError(
            f"Got {num_sessions} sessions of inputs but {len(labels)} of labels."
        )
    if not 0 <= session_id < num_sessions:
        raise ValueError(
            f"Invalid session_id {session_id}: must be between 0 and {num_sessions - 1}."
        )


def _build_aligned_indices(inputs: List[torch.Tensor],
                           labels: List[torch.Tensor],
                           session_id: int,
                           indices=None,
                           random_ties: bool = False,
                           seed: Optional[int] = 0) -> torch.Tensor:
    """Indices of the samples matched to the reference samples of one session.

    In every other session, the sample with the nearest auxiliary label is chosen. Only indices
    are returned; :py:func:`_gather_aligned_inputs` builds the model input from them.

    Args:
        inputs: Neural data of each session, ``(T_s, N_s)``.
        labels: Auxiliary variables of each session, ``(T_s, d)``, shared label space.
        session_id: Reference session.
        indices: Samples of the reference session, ``(T,)`` integers. ``None``: the whole session.
        random_ties: Break nearest-label ties uniformly at random, otherwise take the first one.
        seed: Seed of the tie draws. The default 0 gives the same draws in every
            call (``transform``, ``aligned_inputs``, ``feature_attribution_map``); keep the same
            seed in all of them. ``None`` is only allowed with ``random_ties=False``.

    Returns:
        Long tensor ``(num_sessions, len(indices))`` on CPU; row ``s`` holds the sample of
        session ``s`` matched to each reference sample (row ``session_id`` is ``indices``).
    """
    _check_sessions(inputs, labels, session_id)
    num_sessions = len(inputs)
    if indices is None:
        indices = torch.arange(len(inputs[session_id]))
    indices = torch.as_tensor(indices, dtype=torch.long).cpu().reshape(-1)

    # Always built on CPU: the draws and the distance computations are device independent.
    # The neural data is not used by the search: zero-width placeholders avoid copying it.
    dataset = cebra.data.UnifiedDataset(*[
        cebra.data.TensorDataset(torch.empty(len(labels[i]), 0),
                                 continuous=torch.as_tensor(labels[i]).cpu(),
                                 offset=cebra.data.Offset(0, 1),
                                 device="cpu") for i in range(num_sessions)
    ])
    sampler = cebra.distributions.UnifiedSampler(dataset,
                                                 time_offset=1,
                                                 device="cpu",
                                                 random_ties=bool(random_ties))
    if random_ties and seed is None:
        raise ValueError(
            "random_ties=True needs a seed, so that the embedding and the attribution map use "
            "the same tie draws. Pass the same seed to transform and feature_attribution_map "
            "(default 0).")
    generator = None if seed is None else torch.Generator().manual_seed(
        int(seed))
    return sampler.sample_all_sessions(ref_idx=indices,
                                       session_id=session_id,
                                       generator=generator).cpu().long()


def _gather_aligned_inputs(inputs: List[torch.Tensor],
                           aligned_indices: torch.Tensor,
                           rows=None,
                           device=None) -> torch.Tensor:
    """Model input of the unified model for the given aligned indices.

    Args:
        inputs: Neural data of each session, ``(T_s, N_s)``.
        aligned_indices: Output of :py:func:`_build_aligned_indices`, ``(num_sessions, T)``.
        rows: Columns of ``aligned_indices`` to gather. ``None``: all of them.
        device: Device of the output. Default: device of the first input.

    Returns:
        Tensor ``(len(rows), sum_s N_s)``: the channels of all sessions, concatenated.
    """
    idx = aligned_indices if rows is None else aligned_indices[:, rows]
    if device is None:
        device = torch.as_tensor(inputs[0]).device
    out = torch.empty(idx.shape[1],
                      sum(x.shape[1] for x in inputs),
                      dtype=torch.float32,
                      device=device)
    start = 0
    for s, x in enumerate(inputs):
        x = torch.as_tensor(x)
        out[:, start:start + x.shape[1]] = x[idx[s].to(
            x.device)].float().to(device)
        start += x.shape[1]
    return out


class _AlignedInputsMixin:
    """Shared aligned-input building and whole-session transform of the unified solvers."""

    def aligned_inputs(self,
                       inputs: List[torch.Tensor],
                       labels: List[torch.Tensor],
                       session_id: int,
                       indices=None,
                       random_ties: bool = False,
                       seed: Optional[int] = 0) -> torch.Tensor:
        """Model input of the unified model for the reference samples of one session.

        In every other session, the sample with the nearest auxiliary label (squared Euclidean
        distance; one-hot encode discrete labels) is chosen, and the channels of all sessions are
        concatenated.

        Args:
            inputs: Neural data of each session, ``(T_s, N_s)``.
            labels: Auxiliary variables of each session, ``(T_s, d)``, same ``d`` for all.
            session_id: Reference session.
            indices: Samples of the reference session, ``(T,)`` integers. ``None``: all of them.
            random_ties: Draw one of several equally near samples at random instead of taking
                the first. Training uses ``True``.
            seed: Seed of the tie draws. The default 0 gives the same draws in every
                call (``transform``, ``aligned_inputs``, ``feature_attribution_map``); keep the same
                seed in all of them. ``None`` is only allowed with ``random_ties=False``.

        Returns:
            Tensor ``(T, sum_s N_s)`` on the model device.

        Example:
            >>> import torch
            >>> import cebra
            >>> inputs = [torch.rand(50, 3), torch.rand(60, 5)]
            >>> labels = [torch.rand(50, 2), torch.rand(60, 2)]
            >>> model = cebra.models.init("offset1-model", num_neurons=8, num_units=16,
            ...                           num_output=4)
            >>> solver = cebra.solver.UnifiedSolver(model=model,
            ...                                     criterion=cebra.models.InfoNCE(),
            ...                                     optimizer=torch.optim.Adam(model.parameters()))
            >>> x = solver.aligned_inputs(inputs, labels, session_id=0, indices=[0, 1, 2],
            ...                           random_ties=True, seed=0)
            >>> tuple(x.shape)
            (3, 8)
        """
        idx = _build_aligned_indices(inputs,
                                     labels,
                                     session_id,
                                     indices=indices,
                                     random_ties=random_ties,
                                     seed=seed)
        return _gather_aligned_inputs(inputs,
                                      idx,
                                      device=next(
                                          self.model.parameters()).device)

    def _transform_session(self, inputs, labels, pad_before_transform,
                           session_id, batch_size, random_ties, seed):
        """Align the sessions on ``session_id`` and embed the session batch by batch.

        The matches are computed once; each batch of ``batch_size`` output rows is then gathered
        with the context of the receptive field, so the result does not depend on ``batch_size``.
        Arguments are those of :py:meth:`UnifiedSolver.transform`.

        Returns:
            Tensor ``(T, output_dimension)`` (``T - len(offset) + 1`` rows without padding)
            on the model device.
        """
        if not isinstance(inputs, list):
            raise ValueError(
                f"Inputs to transform() should be a list, not {type(inputs)}.")
        self._check_is_fitted()
        if session_id is None:
            raise ValueError("Session ID is required for multi-session models.")

        aligned_idx = _build_aligned_indices(inputs,
                                             labels,
                                             session_id,
                                             random_ties=random_ties,
                                             seed=seed)
        device = next(self.model.parameters()).device
        offset = self.model.get_offset()
        num_samples = aligned_idx.shape[1]
        if pad_before_transform:
            targets = torch.arange(num_samples)
        else:
            targets = torch.arange(
                offset.left, max(offset.left, num_samples - offset.right + 1))
        if len(targets) == 0:
            raise ValueError(
                f"Session {session_id} has {num_samples} samples, fewer than the "
                f"receptive field of the model ({len(offset)}).")
        if batch_size is None:
            batch_size = len(targets)
        window = torch.arange(-offset.left, offset.right)

        self.model.eval()
        outputs = []
        for start in range(0, len(targets), batch_size):
            t = targets[start:start + batch_size]
            # input rows of the batch: its first window to its last one, edges repeated
            rows = torch.arange(t[0] + window[0], t[-1] + window[-1] + 1)
            rows = rows.clamp(0, num_samples - 1)
            batch = _gather_aligned_inputs(inputs,
                                           aligned_idx,
                                           rows=rows,
                                           device=device)
            outputs.append(abc_._inference_transform(self.model, batch))
        return torch.cat(outputs, dim=0)


@register("unified-session")
@dataclasses.dataclass
class UnifiedSolver(_AlignedInputsMixin, abc_.Solver):
    """Multi session training, considering a single model for all sessions."""

    _variant_name = "unified-session"

    def _set_fitted_params(self, loader: cebra.data.Loader):  # mix
        """Set parameters once the solver is fitted.

        In single session solver, the number of session is set to None and the number of
        features is set to the number of neurons in the dataset.

        Args:
            loader: Loader used to fit the solver.
        """
        self.num_sessions = loader.dataset.num_sessions
        self.n_features = loader.dataset.input_dimension

    def _check_is_inputs_valid(self, inputs: Union[torch.Tensor,
                                                   List[torch.Tensor]],
                               session_id: int):
        """Check that the inputs can be inferred using the selected model.

        Note: This method checks that the number of neurons in the input is
        similar to the input dimension to the selected model.

        Args:
            inputs: Data to infer using the selected model.
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.
        """

        if isinstance(inputs, list):
            inputs_shape = 0
            for i in range(len(inputs)):
                inputs_shape += inputs[i].shape[1]
        elif isinstance(inputs,
                        torch.Tensor):  #NOTE(celia): flexible input at training
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.n_features != inputs_shape:
            raise ValueError(
                f"Invalid input shape: model requires an input of shape"
                f"(n_samples, {self.n_features}), got (n_samples, {inputs.shape[1]})."
            )

    def _check_is_session_id_valid(
        self,
        session_id: Optional[int] = None,
    ):  # same as multi
        """Check that the session ID provided is valid for the solver instance.

        The session ID must be non-null and between 0 and the number session in the dataset.

        Args:
            session_id: The session ID to check.
        """

        if session_id is None:
            raise RuntimeError(
                "No session_id provided: unified model requires a session_id as the target session to use to align the sessions."
            )
        if session_id >= self.num_sessions or session_id < 0:
            raise RuntimeError(
                f"Invalid session_id {session_id}: session_id for the current unified model must be between 0 and {self.num_sessions-1}."
            )

    def _get_model(self, session_id: Optional[int] = None):
        """Get the model for the given session ID.

        Args:
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Returns:
            The model for the given session ID.
        """
        self._check_is_session_id_valid(session_id=session_id)
        self._check_is_fitted()
        return self.model

    def _single_model_inference(self, batch: cebra.data.Batch,
                                model: torch.nn.Module) -> cebra.data.Batch:
        """Given a single batch of input examples, computes the feature representation/embedding.

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.
            model: The model to use for inference.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        batch.to(self.device)
        ref = model(batch.reference)
        pos = model(batch.positive)
        neg = model(batch.negative)

        ref = ref.unsqueeze(0)
        pos = pos.unsqueeze(0)
        neg = neg.unsqueeze(0)

        num_features = neg.shape[2]

        return cebra.data.Batch(
            reference=ref.view(-1, num_features),
            positive=pos.view(-1, num_features),
            negative=neg.view(-1, num_features),
        )

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Given batches of input examples, computes the feature representations/embeddings.

        Args:
            batches: A list of input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        return self._single_model_inference(batch, self.model)

    @torch.no_grad()
    def transform(self,
                  inputs: List[torch.Tensor],
                  labels: List[torch.Tensor],
                  pad_before_transform: bool = True,
                  session_id: Optional[int] = None,
                  batch_size: Optional[int] = 512,
                  random_ties: bool = False,
                  seed: Optional[int] = 0) -> torch.Tensor:
        """Compute the embedding of session ``session_id``.

        The other sessions are aligned on ``session_id`` (see :py:meth:`aligned_inputs`), then
        the session is embedded in batches. Each batch keeps its real neighbouring samples, so
        the result does not depend on ``batch_size``.

        Args:
            inputs: Neural data of all sessions, ``(T_s, N_s)``.
            labels: Auxiliary variables of all sessions, ``(T_s, d)``, used for the alignment.
            pad_before_transform: Pad the session edges by replication, so that the output has
                one row per sample.
            session_id: Session to embed, between 0 and the number of sessions - 1.
            batch_size: Output rows per forward pass. ``None``: the whole session at once.
            random_ties: Break nearest-label ties at random instead of taking the first.
                Training uses ``True``.
            seed: Seed of the tie draws. The default 0 gives the same draws in every
                call (``transform``, ``aligned_inputs``, ``feature_attribution_map``); keep the same
                seed in all of them. ``None`` is only allowed with ``random_ties=False``.

        Returns:
            The embedding ``(T, output_dimension)``, with ``T - len(offset) + 1`` rows without
            padding.

        Example:
            >>> import torch
            >>> import cebra
            >>> inputs = [torch.rand(50, 3), torch.rand(60, 5)]
            >>> labels = [torch.rand(50, 2), torch.rand(60, 2)]
            >>> model = cebra.models.init("offset1-model", num_neurons=8, num_units=16,
            ...                           num_output=4)
            >>> solver = cebra.solver.UnifiedSolver(model=model,
            ...                                     criterion=cebra.models.InfoNCE(),
            ...                                     optimizer=torch.optim.Adam(model.parameters()))
            >>> solver.n_features, solver.num_sessions = 8, 2  # set by ``fit`` normally
            >>> emb = solver.transform(inputs, labels, session_id=1, random_ties=True, seed=0)
            >>> tuple(emb.shape)
            (60, 4)
        """
        return self._transform_session(inputs, labels, pad_before_transform,
                                       session_id, batch_size, random_ties,
                                       seed)

    @torch.no_grad()
    def decoding(self,
                 train_loader: cebra.data.Loader,
                 valid_loader: Optional[cebra.data.Loader] = None,
                 decode: str = "ridge",
                 max_sessions: int = 5,
                 max_timesteps: int = 512) -> float:
        """Sample a fixed number of sessions to compute the decoding score.

        Args:
            train_loader: The training data loader.
            valid_loader: The validation data loader, if available.
            decode: The type of decoder to use, either "knn" or "ridge".
            max_sessions: The maximum number of sessions to sample for decoding.
            max_timesteps: The maximum number of timesteps to consider for each session.
        Returns:
            The average decoding score across the sampled sessions. If a validation loader is provided,
            returns a tuple of the average training score and the average validation score.
        """
        if train_loader.dataset.num_sessions > max_sessions:
            sessions = np.random.choice(np.arange(
                train_loader.dataset.num_sessions),
                                        size=max_sessions,
                                        replace=False)
        else:
            sessions = np.arange(train_loader.dataset.num_sessions)

        train_scores, valid_scores = [], []
        for i in sessions:
            if train_loader.dataset.get_session(
                    i).neural.shape[0] > max_timesteps:
                train_end = max_timesteps
            else:
                train_end = -1
            train_x = self.transform([
                train_loader.dataset.get_session(j).neural[:train_end]
                for j in range(train_loader.dataset.num_sessions)
            ], [
                train_loader.dataset.get_session(j).continuous_index[:train_end]
                if train_loader.dataset.get_session(j).continuous_index
                is not None else
                train_loader.dataset.get_session(j).discrete_index[:train_end]
                for j in range(train_loader.dataset.num_sessions)
            ],
                                     session_id=i,
                                     batch_size=128)
            train_y = train_loader.dataset.get_session(
                i
            ).continuous_index[:train_end] if train_loader.dataset.get_session(
                i
            ).continuous_index is not None else train_loader.dataset.get_session(
                i).discrete_index[:train_end]

            if valid_loader is not None:
                if valid_loader.dataset.get_session(
                        i).neural.shape[0] > max_timesteps:
                    valid_end = max_timesteps
                else:
                    valid_end = -1
                valid_x = self.transform([
                    valid_loader.dataset.get_session(j).neural[:valid_end]
                    for j in range(valid_loader.dataset.num_sessions)
                ], [
                    valid_loader.dataset.get_session(
                        j).continuous_index[:valid_end]
                    if valid_loader.dataset.get_session(j).continuous_index
                    is not None else valid_loader.dataset.get_session(
                        j).discrete_index[:valid_end]
                    for j in range(valid_loader.dataset.num_sessions)
                ],
                                         session_id=i,
                                         batch_size=128)
                valid_y = valid_loader.dataset.get_session(
                    i
                ).continuous_index[:valid_end] if valid_loader.dataset.get_session(
                    i
                ).continuous_index is not None else valid_loader.dataset.get_session(
                    i).discrete_index[:valid_end]

            if decode == "knn":
                decoder = cebra.KNNDecoder()
            elif decode == "ridge":
                decoder = cebra.RidgeRegressor()
            else:
                raise NotImplementedError(f"Decoder {decode} not implemented.")

            decoder.fit(train_x.cpu().numpy(), train_y.cpu().numpy())
            train_scores.append(
                decoder.score(train_x.cpu().numpy(),
                              train_y.cpu().numpy()))

            if valid_loader is not None:
                valid_scores.append(
                    decoder.score(valid_x.cpu().numpy(),
                                  valid_y.cpu().numpy()))

        if valid_loader is None:
            return np.array(train_scores).mean()
        else:
            return np.array(train_scores).mean(), np.array(valid_scores).mean()


@register("unified-multiobjective-solver",
          deprecated=True)  # NOTE: earlier name, kept for compatibility
@register("unified-xcebra")
@dataclasses.dataclass
class UnifiedSolverxCEBRA(_AlignedInputsMixin, MultiobjectiveSolverBase):
    """Unified CEBRA solver with Jacobian regularization (unified xCEBRA).

    Trains one model shared by all sessions, like :py:class:`~cebra.solver.unified_session.UnifiedSolver`,
    and adds the Jacobian regularization of xCEBRA
    (:py:class:`~cebra.models.jacobian_regularizer.JacobianReg`, passed as ``regularizer``, with its
    weight set over training by ``scheduler_regularizer`` in :py:meth:`fit`). It builds on the
    multi-objective solver only because that solver supports the regularizer.

    Unified xCEBRA uses a single objective: one slice covering the whole embedding, with a
    contrastive loss on the auxiliary variables, set with
    :py:class:`~cebra.solver.multiobjective.MultiObjectiveConfig` (one ``set_slice`` /
    ``set_loss`` / ``set_distribution`` / ``push``). Several slices (subspaces) are accepted by
    the code but have not been used or validated with the unified sampler.

    Takes batches from :py:class:`~cebra.data.multiobjective.UnifiedContrastiveMultiObjectiveLoader`,
    in which the samples of all sessions are already aligned and concatenated.

    Example:
        >>> # loader: a UnifiedContrastiveMultiObjectiveLoader over the unified dataset
        >>> config = cebra.solver.MultiObjectiveConfig(loader)  # doctest: +SKIP
        >>> config.set_slice(0, 128)                    # one objective: the whole embedding  # doctest: +SKIP
        >>> config.set_loss("FixedCosineInfoNCE", temperature=1.0)  # doctest: +SKIP
        >>> config.set_distribution("unified_time_delta", time_delta=10)  # doctest: +SKIP
        >>> config.push(); config.finalize()  # doctest: +SKIP
        >>> solver = cebra.solver.init("unified-xcebra", model=model,
        ...     feature_ranges=config.feature_ranges, criterion=config.criterion,
        ...     regularizer=cebra.models.jacobian_regularizer.JacobianReg(),
        ...     renormalize=True, use_sam=False, optimizer=optimizer)  # doctest: +SKIP
        >>> solver.fit(loader, scheduler_regularizer=weight_scheduler)  # doctest: +SKIP

    See Also:
        :py:class:`cebra.data.multiobjective.UnifiedContrastiveMultiObjectiveLoader`
        :py:class:`cebra.solver.multiobjective.MultiObjectiveConfig`
        :py:class:`cebra.models.multicriterions.MultiCriterions`
    """

    _variant_name = "unified-xcebra"

    def _set_fitted_params(self, loader: cebra.data.Loader):
        """Set parameters once the solver is fitted for unified multiobjective training.

        For unified training, num_sessions is the number of sessions in the unified dataset,
        and n_features is the total concatenated feature dimension across all sessions.

        Args:
            loader: Loader used to fit the solver.
        """
        self.num_sessions = loader.dataset.num_sessions
        self.n_features = loader.dataset.input_dimension

    def _check_is_inputs_valid(self,
                               inputs: Optional[List[torch.Tensor]],
                               session_id: Optional[int] = None):
        """Check that the inputs can be inferred using the unified model.

        For unified training, inputs should be a list of per-session tensors,
        and the concatenated feature dimension must match the model's input dimension.

        Args:
            inputs: List of input tensors, one per session. Each has shape
                   (n_samples, num_features_session, ...).
            session_id: The target session ID (required for unified models).
        """
        if inputs is None:
            return

        if isinstance(inputs, list):
            inputs_shape = 0
            for i in range(len(inputs)):
                inputs_shape += inputs[i].shape[1]
        elif isinstance(inputs, torch.Tensor):
            raise NotImplementedError(
                "Unified models require a list of per-session tensors, not a single tensor."
            )
        else:
            raise NotImplementedError(
                f"Invalid input type: {type(inputs)}. Expected list of tensors."
            )

        if self.n_features != inputs_shape:
            raise ValueError(
                f"Invalid input shape: model requires an input of shape "
                f"(n_samples, {self.n_features}), but got total feature dimension "
                f"(n_samples, {inputs_shape}) from concatenated sessions.")

    def _check_is_session_id_valid(
        self,
        session_id: Optional[int] = None,
    ):
        """Check that the session ID provided is valid for the unified solver.

        The session ID must be non-null and between 0 and the number of sessions.

        Args:
            session_id: The session ID to check.
        """
        if session_id is None:
            raise RuntimeError(
                "No session_id provided: unified multiobjective model requires a session_id "
                "as the target session to use to align the sessions.")
        if session_id >= self.num_sessions or session_id < 0:
            raise RuntimeError(
                f"Invalid session_id {session_id}: session_id for the current unified model "
                f"must be between 0 and {self.num_sessions-1}.")

    def _get_model(self, session_id: Optional[int] = None):
        """Get the unified model (independent of session_id).

        For unified models, there is only one shared model regardless of session.
        The session_id is still validated but not used for model selection.

        Args:
            session_id: The session ID, required for validation.

        Returns:
            The shared unified model.
        """
        self._check_is_session_id_valid(session_id=session_id)
        self._check_is_fitted()
        return self.model

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Compute predictions for unified multiobjective batch.

        For unified batches, reference, positive, and negative samples are already
        aligned across sessions (concatenated along the neuron dimension). We apply
        the single model to each and return per-objective prediction batches.

        Args:
            batch: A Batch with reference, positive (list), and negative tensors
                   already unified across sessions.

        Returns:
            A list of Batch objects, one per objective, with model predictions.
        """
        pred_refs = self.model(batch.reference)
        pred_negs = self.model(batch.negative)

        prediction_batches = []
        for i, positive in enumerate(batch.positive):
            pred_pos = self.model(positive)
            prediction_batches.append(
                cebra.data.Batch(reference=pred_refs[i].unsqueeze(0).view(
                    -1, pred_refs[i].shape[-1]),
                                 positive=pred_pos[i].unsqueeze(0).view(
                                     -1, pred_pos[i].shape[-1]),
                                 negative=pred_negs[i].unsqueeze(0).view(
                                     -1, pred_negs[i].shape[-1])))

        return prediction_batches

    @torch.no_grad()
    def transform(self,
                  inputs: List[torch.Tensor],
                  labels: List[torch.Tensor],
                  pad_before_transform: bool = True,
                  session_id: Optional[int] = None,
                  batch_size: Optional[int] = 512,
                  random_ties: bool = False,
                  seed: Optional[int] = 0) -> torch.Tensor:
        """Compute the embedding of session ``session_id``.

        Same as :py:meth:`UnifiedSolver.transform`. The model is run with ``split_outputs``
        off (its training mode returns one output per slice) and its mode is restored
        afterwards, also if an error is raised.

        Args:
            inputs: Neural data of all sessions, ``(T_s, N_s)``.
            labels: Auxiliary variables of all sessions, ``(T_s, d)``, used for the alignment.
            pad_before_transform: Pad the session edges by replication, so that the output has
                one row per sample.
            session_id: Session to embed. Required.
            batch_size: Output rows per forward pass. ``None``: the whole session at once.
            random_ties: Break nearest-label ties at random instead of taking the first.
                Training uses ``True``.
            seed: Seed of the tie draws. The default 0 gives the same draws in every
                call (``transform``, ``aligned_inputs``, ``feature_attribution_map``); keep the same
                seed in all of them. ``None`` is only allowed with ``random_ties=False``.

        Returns:
            The embedding ``(T, output_dimension)``.

        Example:
            >>> # ``solver`` is a fitted UnifiedSolverxCEBRA, ``inputs``/``labels``
            >>> # the lists of per-session neural data and auxiliary variables
            >>> emb = solver.transform(inputs, labels, session_id=0)  # doctest: +SKIP
            >>> emb = solver.transform(inputs, labels, session_id=0,
            ...                        random_ties=True, seed=0)  # doctest: +SKIP
        """
        previous = self.model.split_outputs
        try:
            # Training runs the model with one output per slice (one per loss);
            # inference needs the single concatenated embedding.
            self.model.set_split_outputs(False)
            return self._transform_session(inputs, labels, pad_before_transform,
                                           session_id, batch_size, random_ties,
                                           seed)
        finally:
            self.model.set_split_outputs(previous)
