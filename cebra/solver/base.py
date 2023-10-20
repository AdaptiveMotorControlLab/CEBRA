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
"""This package contains abstract base classes for different solvers.

Solvers are used to package models, criterions and optimizers and implement training
loops. When subclassing abstract solvers, in the simplest case only the
:py:meth:`Solver._inference` needs to be overridden.

For more complex use cases, the :py:meth:`Solver.step` and
:py:meth:`Solver.fit` method can be overridden to
implement larger changes to the training loop.
"""

import abc
import os
from typing import Callable, Dict, List, Literal, Optional, Union

import literate_dataclasses as dataclasses
import numpy as np
import torch
import tqdm

import cebra
import cebra.data
import cebra.io
import cebra.models
import cebra.solver.util as cebra_solver_util
from cebra.solver.util import Meter
from cebra.solver.util import ProgressBar


@dataclasses.dataclass
class Solver(abc.ABC, cebra.io.HasDevice):
    """Solver base class.

    A solver contains helper methods for bundling a model, criterion and optimizer.

    Attributes:
        model: The encoder for transforming reference, positive and negative samples.
        criterion: The criterion computed from the similarities between positive pairs
            and negative pairs. The criterion can have trainable parameters on its own.
        optimizer: A PyTorch optimizer for updating model and criterion parameters.
        history: Deprecated since 0.0.2. Use :py:attr:`log`.
        decode_history: Deprecated since 0.0.2. Use a hook during training for validation and
            decoding. See the arguments of :py:meth:`fit`.
        log: The logs recorded during training, typically contains the ``total`` loss as well
            as the logs for positive (``pos``) and negative (``neg``) pairs. For the standard
            criterions in CEBRA, also contains the value of the ``temperature``.
        tqdm_on: Use ``tqdm`` for showing a progress bar during training.
    """

    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    history: List = dataclasses.field(default_factory=list)
    decode_history: List = dataclasses.field(default_factory=list)
    log: Dict = dataclasses.field(default_factory=lambda: ({
        "pos": [],
        "neg": [],
        "total": [],
        "temperature": []
    }))
    tqdm_on: bool = True

    def __post_init__(self):
        cebra.io.HasDevice.__init__(self)
        self.best_loss = float("inf")

    def state_dict(self) -> dict:
        """Return a dictionary fully describing the current solver state.

        Returns:
            State dictionary, including the state dictionary of the models and
            optimizer. Also contains the training history and the CEBRA version
            the model was trained with.
        """

        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": torch.tensor(self.history),
            "decode": self.decode_history,
            "criterion": self.criterion.state_dict(),
            "version": cebra.__version__,
            "log": self.log,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Update the solver state with the given state_dict.

        Args:
            state_dict: Dictionary with parameters for the `model`, `optimizer`,
                and the past loss history for the solver.
            strict: Make sure all states can be loaded. Set to `False` to allow
                to partially load the state for all given keys.
        """

        def _contains(key):
            if key in state_dict:
                return True
            elif strict:
                raise KeyError(
                    f"Key {key} missing in state_dict. Contains: {list(state_dict.keys())}."
                )
            return False

        def _get(key):
            return state_dict.get(key)

        if _contains("model"):
            self.model.load_state_dict(_get("model"))
        if _contains("criterion"):
            self.criterion.load_state_dict(_get("criterion"))
        if _contains("optimizer"):
            self.optimizer.load_state_dict(_get("optimizer"))
        # TODO(stes): This will be deprecated at some point; the "log" attribute
        # holds the same information.
        if _contains("loss"):
            self.history = _get("loss").cpu().numpy().tolist()
        if _contains("decode"):
            self.decode_history = _get("decode")
        if _contains("log"):
            self.log = _get("log")

    @property
    def num_parameters(self) -> int:
        """Total number of parameters in the encoder and criterion."""
        return sum(p.numel() for p in self.parameters())

    def parameters(self):
        """Iterate over all parameters."""
        for parameter in self.model.parameters():
            yield parameter

        for parameter in self.criterion.parameters():
            yield parameter

    def _get_loader(self, loader):
        return ProgressBar(
            loader,
            "tqdm" if self.tqdm_on else "off",
        )

    def fit(
        self,
        loader: cebra.data.Loader,
        valid_loader: cebra.data.Loader = None,
        *,
        save_frequency: int = None,
        valid_frequency: int = None,
        decode: bool = False,
        logdir: str = None,
        save_hook: Callable[[int, "Solver"], None] = None,
    ):
        """Train model for the specified number of steps.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            valid_loader: Data loader used for validation of the model.
            save_frequency: If not `None`, the frequency for automatically saving model checkpoints
                to `logdir`.
            valid_frequency: The frequency for running validation on the ``valid_loader`` instance.
            logdir:  The logging directory for writing model checkpoints. The checkpoints
                can be read again using the `solver.load` function, or manually via loading the
                state dict.

        TODO:
            * Refine the API here. Drop the validation entirely, and implement this via a hook?
        """

        self.to(loader.device)

        iterator = self._get_loader(loader)
        self.model.train()
        for num_steps, batch in iterator:
            stats = self.step(batch)
            iterator.set_description(stats)

            if save_frequency is None:
                continue
            save_model = num_steps % save_frequency == 0
            run_validation = (valid_loader
                              is not None) and (num_steps % valid_frequency
                                                == 0)
            if run_validation:
                validation_loss = self.validation(valid_loader)
                if self.best_loss is None or validation_loss < self.best_loss:
                    self.best_loss = validation_loss
                    self.save(logdir, f"checkpoint_best.pth")
            if save_model:
                if decode:
                    self.decode_history.append(
                        self.decoding(loader, valid_loader))
                if save_hook is not None:
                    save_hook(num_steps, self)
                self.save(logdir, f"checkpoint_{num_steps:#07d}.pth")

    def step(self, batch: cebra.data.Batch) -> dict:
        """Perform a single gradient update.

        Args:
            batch: The input samples

        Returns:
            Dictionary containing training metrics.
        """
        self.optimizer.zero_grad()
        prediction = self._inference(batch)
        loss, align, uniform = self.criterion(prediction.reference,
                                              prediction.positive,
                                              prediction.negative)

        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())
        stats = dict(
            pos=align.item(),
            neg=uniform.item(),
            total=loss.item(),
            temperature=self.criterion.temperature,
        )
        for key, value in stats.items():
            self.log[key].append(value)
        return stats

    def validation(self,
                   loader: cebra.data.Loader,
                   session_id: Optional[int] = None):
        """Compute score of the model on data.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            session_id: The session ID, an integer between 0 and the number of sessions in the
                multisession model, set to None for single session.

        Returns:
            Loss averaged over iterations on data batch.
        """
        assert (session_id is None) or (session_id == 0)
        iterator = self._get_loader(loader)
        total_loss = Meter()
        self.model.eval()
        for _, batch in iterator:
            prediction = self._inference(batch)
            loss, _, _ = self.criterion(prediction.reference,
                                        prediction.positive,
                                        prediction.negative)
            total_loss.add(loss.item())
        return total_loss.average

    @torch.no_grad()
    def decoding(self, train_loader, valid_loader):
        """Deprecated since 0.0.2."""
        train_x = self.transform(train_loader.dataset[torch.arange(
            len(train_loader.dataset.neural))])
        train_y = train_loader.dataset.index
        valid_x = self.transform(valid_loader.dataset[torch.arange(
            len(valid_loader.dataset.neural))])
        valid_y = valid_loader.dataset.index
        decode_metric = train_loader.dataset.decode(
            train_x.cpu().numpy(),
            train_y.cpu().numpy(),
            valid_x.cpu().numpy(),
            valid_y.cpu().numpy(),
        )
        return decode_metric

    def _inference_transform(self, model, inputs):

        if isinstance(model, cebra.models.ConvolutionalModelMixin):
            # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
            inputs = inputs.transpose(1, 0).unsqueeze(0)
            output = model(inputs).squeeze(0).transpose(1, 0)
        else:
            output = model(inputs)

        return output

    def _select_model(self, inputs: torch.Tensor, session_id: int):
        """ Select the right model based on the type of solver we have."""

        self.num_sessions = self.loader.dataset.num_sessions if isinstance(
            inputs, list) else None
        if self.num_sessions is not None:  # multisession implementation
            if session_id is None:
                raise RuntimeError(
                    "No session_id provided: multisession model requires a session_id to choose the model corresponding to your data shape."
                )
            if session_id >= self.num_sessions or session_id < 0:
                raise RuntimeError(
                    f"Invalid session_id {session_id}: session_id for the current multisession model must be between 0 and {self.num_sessions-1}."
                )
            if self.n_features_[session_id] != X.shape[1]:
                raise ValueError(
                    f"Invalid input shape: model for session {session_id} requires an input of shape"
                    f"(n_samples, {self.n_features_[session_id]}), got (n_samples, {X.shape[1]})."
                )

            model = self.model[session_id]
            model.to(self.device_)  #TODO: why do I need to do this?

        else:  # single session
            if session_id is not None and session_id > 0:
                raise RuntimeError(
                    f"Invalid session_id {session_id}: single session models only takes an optional null session_id."
                )

            if isinstance(
                    self,
                    cebra.solver.single_session.SingleSessionHybridSolver):
                # NOTE: This is different from the sklearn API implementation. The issue is that here the
                # model is a cebra.models.MultiObjective instance, and therefore to do inference I need
                # to get the module inside this model.
                model = self.model.module
            else:
                model = self.model

        offset = model.get_offset()
        return model, offset

    @torch.no_grad()
    def _batched_transform(self, model, inputs, offset, batch_size,
                           pad_before_transform) -> torch.Tensor:
        output = []
        batches = cebra_solver_util.get_batches_of_data(
            inputs=inputs,
            batch_size=batch_size,
            padding=pad_before_transform,
            offset=offset)

        # NOTE: If we move this inside the `cebra_solver_util.get_batches_of_data`or similar
        # we avoid a second for loop. Is it good practice to do inference outside the solver?
        for batch in batches:
            output_batch = self._inference_transform(model, batch)
            output.append(output_batch)

        output = torch.cat(output)
        return output

    @torch.no_grad()
    def _transform(self, model, inputs, offset,
                   pad_before_transform) -> torch.Tensor:

        if pad_before_transform:
            inputs = np.pad(inputs, ((offset.left, offset.right - 1), (0, 0)),
                            mode="edge")
            inputs = torch.from_numpy(inputs)

        output = self._inference_transform(model, inputs)
        return output

    @torch.no_grad()
    def transform(
            self,
            inputs: torch.Tensor,
            pad_before_transform: bool = True,  #TODO: what should be the default?
            session_id: Optional[int] = None,
            batch_size: Optional[int] = None) -> torch.Tensor:
        """Compute the embedding.

        This function by default only applies the ``forward`` function
        of the given model, after switching it into eval mode.

        Args:
            inputs: The input signal
            pad_before_transform: If ``False``, no padding is applied to the input sequence.
            and the output sequence will be smaller than the input sequence due to the
            receptive field of the model. If the input sequence is ``n`` steps long,
            and a model with receptive field ``m`` is used, the output sequence would
            only be ``n-m+1`` steps long.
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Returns:
            The output embedding.
        """
        model, offset = self._select_model(inputs, session_id)
        model.eval()

        if len(offset) < 2 and pad_before_transform:
            raise ValueError(
                "Padding does not make sense when the offset of the model is < 2"
            )

        if batch_size is not None:
            output = self._batched_transform(
                model=model,
                inputs=inputs,
                offset=offset,
                batch_size=batch_size,
                pad_before_transform=pad_before_transform,
            )

        else:
            output = self._transform(model=model,
                                     inputs=inputs,
                                     offset=offset,
                                     pad_before_transform=pad_before_transform)

        return output

    @abc.abstractmethod
    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Given a batch of input examples, return the model outputs.

        TODO: make this a public function?

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        raise NotImplementedError

    def load(self, logdir, filename="checkpoint.pth"):
        """Load the experiment from its checkpoint file.

        Args:
            filename (str): Checkpoint name for loading the experiment.
        """

        savepath = os.path.join(logdir, filename)
        if not os.path.exists(savepath):
            print("Did not find a previous experiment. Starting from scratch.")
            return
        checkpoint = torch.load(savepath, map_location=self.device)
        self.load_state_dict(checkpoint, strict=True)

    def save(self, logdir, filename="checkpoint_last.pth"):
        """Save the model and optimizer params.

        Args:
            logdir: Logging directory for this model.
            filename: Checkpoint name for saving the experiment.
        """
        if not os.path.exists(os.path.dirname(logdir)):
            os.makedirs(logdir)
        savepath = os.path.join(logdir, filename)
        torch.save(
            self.state_dict(),
            savepath,
        )


@dataclasses.dataclass
class MultiobjectiveSolver(Solver):
    """Train models to satisfy multiple learning objectives.

    This variant of the standard :py:class:`cebra.solver.base.Solver` implements multi-objective
    or "hybrid" training.

    Attributes:
        model: A multi-objective CEBRA model
        optimizer: The optimizer used for training.
        num_behavior_features: The feature dimension for the features dedicated
            to satisfy the behavior contrastive objective. The remainder is used
            for time contrastive learning.
        renormalize_features: If ``True``, normalize the behavior and time
            contrastive features individually before computing similarity scores.
    """

    num_behavior_features: int = 3
    renormalize_features: bool = False
    output_mode: Literal["overlapping", "separate"] = "overlapping"

    @property
    def num_time_features(self):
        return self.num_total_features - self.num_behavior_features

    @property
    def num_total_features(self):
        return self.model.num_output

    def __post_init__(self):
        super().__post_init__()
        self._check_dimensions()
        self.model = cebra.models.MultiobjectiveModel(
            self.model,
            dimensions=(self.num_behavior_features, self.model.num_output),
            renormalize=self.renormalize_features,
            output_mode=self.output_mode,
        )

    def _check_dimensions(self):
        """Check the feature dimensions for behavior/time contrastive learning.

        Raises:
            ValueError: If feature dimensions are larger than the model features,
                or not sufficiently large for renormalization.
        """
        if self.output_mode == "separate":
            if self.num_behavior_features >= self.num_total_features:
                raise ValueError(
                    "For multi-objective training, the number of features for "
                    f"behavior contrastive learning ({self.num_behavior_features}) cannot be as large or larger "
                    f"than the total feature dimension ({self.num_total_features})."
                )
            if self.num_time_features >= self.num_total_features:
                raise ValueError(
                    "For multi-objective training, the number of features for "
                    f"time contrastive learning ({self.num_time_features}) cannot be as large or larger "
                    f"than the total feature dimension ({self.num_total_features})."
                )
        if self.renormalize_features:
            if self.num_behavior_features < 2:
                raise ValueError(
                    "When renormalizing the features, the feature dimension needs "
                    "to be at least 2 for behavior. "
                    "Check the values of 'renormalize_features' and 'num_behavior_features'."
                )
            if self.num_time_features < 2:
                raise ValueError(
                    "When renormalizing the features, the feature dimension needs "
                    "to be at least 2 for behavior. "
                    "Check the values of 'renormalize_features' and 'num_time_features'."
                )

    def step(self, batch: cebra.data.Batch) -> dict:
        """Perform a single gradient update with multiple objectives.

        Args:
            batch: The input samples

        Returns:
            Dictionary containing training metrics.
        """
        self.optimizer.zero_grad()
        prediction_behavior, prediction_time = self._inference(batch)

        behavior_loss, behavior_align, behavior_uniform = self.criterion(
            prediction_behavior.reference,
            prediction_behavior.positive,
            prediction_behavior.negative,
        )

        time_loss, time_align, time_uniform = self.criterion(
            prediction_time.reference,
            prediction_time.positive,
            prediction_time.negative,
        )

        loss = behavior_loss + time_loss
        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())
        return dict(
            behavior_pos=behavior_align.item(),
            behavior_neg=behavior_uniform.item(),
            behavior_total=behavior_loss.item(),
            time_pos=time_align.item(),
            time_neg=time_uniform.item(),
            time_total=time_loss.item(),
        )
