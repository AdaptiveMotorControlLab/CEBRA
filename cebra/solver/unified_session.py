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


@register("unified-session")
@dataclasses.dataclass
class UnifiedSolver(abc_.Solver):
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
                  batch_size: Optional[int] = 512) -> torch.Tensor:
        """Compute the embedding for the `session_id`th session of the dataset.

        Note:
            Compared to the other :py:class:`cebra.solver.base.Solver`, we need all the sessions of
            the dataset to transform the data, as the sampling is across all the sessions.

        Args:
            inputs: The input signal for all sessions.
            labels: The auxiliary variables to use for sampling.
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1.
            batch_size: If not None, batched inference will be applied.

        Note:
            The ``session_id`` is needed in order to sample the corresponding number of samples and
            return an embedding of the expected shape.

        Note:
            The batched inference will be required in most cases. Default is set to ``100`` for that reason.

        Returns:
            The output embedding for the session corresponding to the provided ID `session_id`. The shape
            is (num_samples(session_id), output_dimension)``.

        """
        if not isinstance(inputs, list):
            raise ValueError(
                f"Inputs to transform() should be a list, not {type(inputs)}.")

        self._check_is_fitted()

        if session_id is None:
            raise ValueError("Session ID is required for multi-session models.")

        # Sampling according to session_id required
        dataset = cebra.data.UnifiedDataset(
            cebra.data.TensorDataset(
                inputs[i], continuous=labels[i], offset=cebra.data.Offset(0, 1))
            for i in range(len(inputs))).to(self.device)

        # Only used to sample the reference samples
        loader = cebra.data.UnifiedLoader(dataset, num_steps=1)

        # Sampling in batch
        refs_data_batch_embeddings = []
        batch_range = range(0, len(dataset.get_session(session_id)), batch_size)
        if len(batch_range) < 2:
            raise ValueError(
                "Not enough data to perform the batched transform. Please provide a larger dataset or reduce the batch size."
            )
        for batch_start in batch_range:
            batch_end = min(batch_start + batch_size,
                            len(dataset.get_session(session_id)))

            if batch_start == batch_range[-2]:  # one before last batch
                last_start = batch_start
                continue
            if batch_start == batch_range[-1]:  # last batch, likely incomplete
                batch_start = last_start
                batch_end = len(dataset.get_session(session_id))

            refs_idx_batch = loader.sampler.sample_all_sessions(
                ref_idx=torch.arange(batch_start, batch_end),
                session_id=session_id).to(self.device)

            refs_data_batch = torch.cat([
                session[refs_idx_batch[session_id]]
                for session_id, session in enumerate(dataset.iter_sessions())
            ],
                                        dim=1).squeeze()

            if len(self.model.get_offset()) < 2 and pad_before_transform:
                pad_before_transform = False

            self.model.eval()
            refs_data_batch_embeddings.append(
                self._transform(model=self.model,
                                inputs=refs_data_batch,
                                pad_before_transform=pad_before_transform,
                                offset=self.model.get_offset(),
                                batch_size=batch_size))

        return torch.cat(refs_data_batch_embeddings, dim=0)

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
