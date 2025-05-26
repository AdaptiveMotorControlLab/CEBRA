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
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import literate_dataclasses as dataclasses
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import cebra
import cebra.data
import cebra.io
import cebra.models
from cebra.solver.util import Meter
from cebra.solver.util import ProgressBar


def _check_indices(batch_start_idx: int, batch_end_idx: int,
                   offset: cebra.data.Offset, num_samples: int):
    """Check that indices in a batch are in a correct range.

    First and last index must be positive integers, smaller than
    the total length of inputs in the dataset, the first index
    must be smaller than the last and the batch size cannot be
    smaller than the offset of the model.

    Args:
        batch_start_idx: Index of the first sample in the batch.
        batch_end_idx: Index of the first sample in the batch.
        offset: Model offset.
        num_samples: Total number of samples in the input.
    """

    if batch_start_idx < 0 or batch_end_idx < 0:
        raise ValueError(
            f"batch_start_idx ({batch_start_idx}) and batch_end_idx ({batch_end_idx}) must be positive integers."
        )
    if batch_start_idx > batch_end_idx:
        raise ValueError(
            f"batch_start_idx ({batch_start_idx}) cannot be greater than batch_end_idx ({batch_end_idx})."
        )
    if batch_end_idx > num_samples:
        raise ValueError(
            f"batch_end_idx ({batch_end_idx}) cannot exceed the length of inputs ({num_samples})."
        )

    batch_size_length = batch_end_idx - batch_start_idx
    if batch_size_length <= len(offset):
        raise ValueError(
            f"The batch has length {batch_size_length} which "
            f"is smaller or equal than the required offset length {len(offset)}."
            f"Either choose a model with smaller offset or the batch should contain 3 times more samples."
        )


def _add_batched_zero_padding(batched_data: torch.Tensor,
                              offset: cebra.data.Offset, batch_start_idx: int,
                              batch_end_idx: int,
                              num_samples: int) -> torch.Tensor:
    """Add zero padding to the input data before inference.

    Args:
        batched_data: Data to apply the inference on.
        offset: Offset of the model to consider when padding.
        batch_start_idx: Index of the first sample in the batch.
        batch_end_idx: Index of the first sample in the batch.
        num_samples (int): Total number of samples in the data.

    Returns:
        The padded batch.
    """
    if batch_start_idx > batch_end_idx:
        raise ValueError(
            f"batch_start_idx ({batch_start_idx}) cannot be greater than batch_end_idx ({batch_end_idx})."
        )
    if batch_start_idx < 0 or batch_end_idx < 0:
        raise ValueError(
            f"batch_start_idx ({batch_start_idx}) and batch_end_idx ({batch_end_idx}) must be positive integers."
        )

    reversed_dims = torch.arange(batched_data.ndim - 1, -1, -1)

    if batch_start_idx == 0:  # First batch
        batched_data = F.pad(batched_data.permute(*reversed_dims),
                             (offset.left, 0),
                             'replicate').permute(*reversed_dims)
    elif batch_end_idx == num_samples:  # Last batch
        batched_data = F.pad(batched_data.permute(*reversed_dims),
                             (0, offset.right - 1),
                             'replicate').permute(*reversed_dims)

    return batched_data


def _get_batch(inputs: torch.Tensor, offset: Optional[cebra.data.Offset],
               batch_start_idx: int, batch_end_idx: int,
               pad_before_transform: bool) -> torch.Tensor:
    """Get a batch of samples between the `batch_start_idx` and `batch_end_idx`.

    Args:
        inputs: Input data.
        offset: Model offset.
        batch_start_idx: Index of the first sample in the batch.
        batch_end_idx: Index of the last sample in the batch.
        pad_before_transform: If True zero-pad the batched data.

    Returns:
        The batch.
    """
    if offset is None:
        raise ValueError("offset cannot be null.")

    if batch_start_idx == 0:  # First batch
        indices = batch_start_idx, (batch_end_idx + offset.right - 1)
    elif batch_end_idx == len(inputs):  # Last batch
        indices = (batch_start_idx - offset.left), batch_end_idx
    else:
        indices = batch_start_idx - offset.left, batch_end_idx + offset.right - 1

    _check_indices(indices[0], indices[1], offset, len(inputs))
    batched_data = inputs[slice(*indices)]

    if pad_before_transform:
        batched_data = _add_batched_zero_padding(
            batched_data=batched_data,
            offset=offset,
            batch_start_idx=batch_start_idx,
            batch_end_idx=batch_end_idx,
            num_samples=len(inputs))

    return batched_data


def _inference_transform(model: cebra.models.Model,
                         inputs: torch.Tensor) -> torch.Tensor:
    """Compute the embedding on the inputs using the model provided.

    Args:
        model: Model to use for inference.
        inputs: Data.

    Returns:
        The embedding.
    """
    inputs = inputs.float().to(next(model.parameters()).device)

    if isinstance(model, cebra.models.ConvolutionalModelMixin):
        # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
        inputs = inputs.transpose(1, 0).unsqueeze(0)
        output = model(inputs).squeeze(0).transpose(1, 0)
    else:
        output = model(inputs)
    return output


def _not_batched_transform(
    model: cebra.models.Model,
    inputs: torch.Tensor,
    pad_before_transform: bool,
    offset: cebra.data.datatypes.Offset,
) -> torch.Tensor:
    """Compute the embedding.

    Args:
        model: The model to use for inference.
        inputs: Input data.
        pad_before_transform: If True, the input data is zero padded before inference.
        offset: Model offset.

    Returns:
        torch.Tensor: The (potentially) padded data.

    Raises:
        ValueError: If add_padding is True and offset is not provided.
    """
    if pad_before_transform:
        inputs = F.pad(inputs.T, (offset.left, offset.right - 1), 'replicate').T
    output = _inference_transform(model, inputs)
    return output


def _batched_transform(model: cebra.models.Model, inputs: torch.Tensor,
                       batch_size: int, pad_before_transform: bool,
                       offset: cebra.data.datatypes.Offset) -> torch.Tensor:
    """Compute the embedding on batched inputs.

    Args:
        model: The model to use for inference.
        inputs: Input data.
        batch_size: Integer corresponding to the batch size.
        pad_before_transform: If True, the input data is zero padded before inference.
        offset: Model offset.

    Returns:
        The embedding.
    """

    class IndexDataset(Dataset):

        def __init__(self, inputs):
            self.inputs = inputs

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return idx

    index_dataset = IndexDataset(inputs)
    index_dataloader = DataLoader(index_dataset, batch_size=batch_size)

    if len(index_dataloader) < 2:
        raise ValueError(
            f"Number of batches must be greater than 1, you can use transform "
            f"without batching instead, got {len(index_dataloader)}.")

    output = []
    for batch_idx, index_batch in enumerate(index_dataloader):
        # NOTE(celia): This is to prevent that adding the offset to the
        # penultimate batch for larger offset make the batch_end_idx larger
        # than the input length, while we also don't want to drop the last
        # samples that do not fit in a complete batch.
        if batch_idx == (len(index_dataloader) - 2):
            # penultimate batch, last complete batch
            last_batch = index_batch
            continue
        if batch_idx == (len(index_dataloader) - 1):
            # last batch, incomplete
            index_batch = torch.cat((last_batch, index_batch), dim=0)

            if index_batch[-1] + 1 != len(inputs):
                raise ValueError(
                    f"Last batch index {index_batch[-1]} + 1 should be equal to the length of inputs {len(inputs)}."
                )

        # Batch start and end so that `batch_size` size with the last batch including 2 batches
        batch_start_idx, batch_end_idx = index_batch[0], index_batch[-1] + 1
        batched_data = _get_batch(inputs=inputs,
                                  offset=offset,
                                  batch_start_idx=batch_start_idx,
                                  batch_end_idx=batch_end_idx,
                                  pad_before_transform=pad_before_transform)

        output_batch = _inference_transform(model, batched_data)
        output.append(output_batch)

    output = torch.cat(output, dim=0)
    return output


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

        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": torch.tensor(self.history),
            "decode": self.decode_history,
            "criterion": self.criterion.state_dict(),
            "version": cebra.__version__,
            "log": self.log,
        }

        if hasattr(self, "n_features"):
            state_dict["n_features"] = self.n_features
        if hasattr(self, "num_sessions"):
            state_dict["num_sessions"] = self.num_sessions

        return state_dict

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

        # Not defined if the model was saved before being fitted.
        if "n_features" in state_dict:
            self.n_features = _get("n_features")
        if "num_sessions" in state_dict:
            self.num_sessions = _get("num_sessions")

    @property
    def num_parameters(self) -> int:
        """Total number of parameters in the encoder and criterion."""
        return sum(p.numel() for p in self.parameters())

    @abc.abstractmethod
    def parameters(self, session_id: Optional[int] = None):
        """Iterate over all parameters of the model.

        Args:
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Yields:
            The parameters of the model.
        """
        raise NotImplementedError

    def _compute_features(
        self,
        batch: cebra.data.Batch,
        model: Optional[torch.nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the features of the reference, positive and negative samples.

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.
            model: The model to use for inference.
                If not provided, the model of the solver is used.

        Returns:
            Tuple of reference, positive and negative features.
        """
        if model is None:
            model = self.model

        batch.to(self.device)
        ref = model(batch.reference)
        pos = model(batch.positive)
        neg = model(batch.negative)
        return ref, pos, neg

    def _get_loader(self, loader):
        return ProgressBar(
            loader,
            "tqdm" if self.tqdm_on else "off",
        )

    @abc.abstractmethod
    def _set_fitted_params(self, loader: cebra.data.Loader):
        """Set parameters once the solver is fitted.

        Args:
            loader: Loader used to fit the solver.
        """

        raise NotImplementedError

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
        self._set_fitted_params(loader)
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
                    self.save(logdir, "checkpoint_best.pth")
            if save_model:
                if decode:
                    self.decode_history.append(
                        self.decoding(loader, valid_loader))
                if save_hook is not None:
                    save_hook(num_steps, self)
                if logdir is not None:
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
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Returns:
            Loss averaged over iterations on data batch.
        """
        if session_id is not None and session_id != 0:
            raise ValueError(
                f"session_id should be set to None or 0, got {session_id}")

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

    def _check_is_inputs_valid(self, inputs: torch.Tensor, session_id: int):
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
            raise ValueError(
                "Inputs to transform() should be the data for a single session, but received a list."
            )
        elif not isinstance(inputs, torch.Tensor):
            raise ValueError(
                f"Inputs should be a torch.Tensor, not {type(inputs)}.")

    @abc.abstractmethod
    def _check_is_session_id_valid(self, session_id: Optional[int] = None):
        """Check that the session ID provided is valid for the solver instance.

        Args:
            session_id: The session ID to check.
        """
        raise NotImplementedError

    def _select_model(
        self, inputs: Union[torch.Tensor,
                            List[torch.Tensor]], session_id: Optional[int]
    ) -> Tuple[Union[List[torch.nn.Module], torch.nn.Module],
               cebra.data.datatypes.Offset]:
        """ Select the model based on the input dimension and session ID.

        Args:
            inputs: Data to infer using the selected model.
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Returns:
            The model (first returns) and the offset of the model (second returns).
        """
        model = self._get_model(session_id=session_id)
        offset = model.get_offset()

        self._check_is_inputs_valid(inputs, session_id=session_id)
        return model, offset

    @abc.abstractmethod
    def _get_model(self,
                   session_id: Optional[int] = None) -> cebra.models.Model:
        """Get the model to use for inference.

        Args:
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Returns:
            The model.
        """
        raise NotImplementedError

    def _check_is_fitted(self):
        """Check if the model is fitted.

        If the model is fitted, the solver should have a `n_features` attribute.

        Raises:
            ValueError: If the model is not fitted.
        """
        if not hasattr(self, "n_features"):
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator.")

    @torch.no_grad()
    def transform(self,
                  inputs: torch.Tensor,
                  pad_before_transform: Optional[bool] = True,
                  session_id: Optional[int] = None,
                  batch_size: Optional[int] = None) -> torch.Tensor:
        """Compute the embedding.

        This function by default only applies the ``forward`` function
        of the given model, after switching it into eval mode.

        Args:
            inputs: The input signal (T, N).
            pad_before_transform: If ``False``, no padding is applied to the input
                sequence and the output sequence will be smaller than the input
                sequence due to the receptive field of the model. If the
                input sequence is ``n`` steps long, and a model with receptive
                field ``m`` is used, the output sequence would  only be
                ``n-m+1`` steps long.
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.
            batch_size: If not None, batched inference will not be applied.

        Returns:
            The output embedding.
        """
        self._check_is_fitted()
        model, offset = self._select_model(inputs, session_id)

        if len(offset) < 2 and pad_before_transform:
            pad_before_transform = False

        model.eval()
        return self._transform(model=model,
                               inputs=inputs,
                               pad_before_transform=pad_before_transform,
                               offset=offset,
                               batch_size=batch_size)

    @torch.no_grad()
    def _transform(self, model: cebra.models.Model, inputs: torch.Tensor,
                   pad_before_transform: bool,
                   offset: cebra.data.datatypes.Offset,
                   batch_size: Optional[int]) -> torch.Tensor:
        """Compute the embedding on the inputs using the model provided.

        Args:
            model: Model to use for inference.
            inputs: Data.
            pad_before_transform: If True zero-pad the batched data.
            offset: Offset of the model to consider when padding.
            batch_size: If not None, batched inference will not be applied.

        Returns:
            The embedding.
        """
        if batch_size is not None and inputs.shape[0] > int(
                batch_size * 2) and not (isinstance(
                    self._get_model(0), cebra.models.ResampleModelMixin)):
            # NOTE(celia): resampling models are not supported for batched inference.
            output = _batched_transform(
                model=model,
                inputs=inputs,
                offset=offset,
                batch_size=batch_size,
                pad_before_transform=pad_before_transform,
            )
        else:
            output = _not_batched_transform(
                model=model,
                inputs=inputs,
                offset=offset,
                pad_before_transform=pad_before_transform)
        return output

    @abc.abstractmethod
    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Given a batch of input examples, return the model outputs.

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

    def load(self, logdir: str, filename: str = "checkpoint.pth"):
        """Load the experiment from its checkpoint file.

        Args:
            logdir: Logging directory.
            filename: Checkpoint name for loading the experiment.
        """

        savepath = os.path.join(logdir, filename)
        if not os.path.exists(savepath):
            print("Did not find a previous experiment. Starting from scratch.")
            return
        checkpoint = torch.load(savepath, map_location=self.device)
        self.load_state_dict(checkpoint, strict=True)

        n_features = self.n_features
        self.n_features = ([
            session_n_features for session_n_features in n_features
        ] if isinstance(n_features, list) else n_features)

    def save(self, logdir: str, filename: str = "checkpoint_last.pth"):
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
        ignore_deprecation_warning: If ``True``, suppress the deprecation warning.

    Note:
        This solver will be deprecated in a future version. Please use the functionality in
        :py:mod:`cebra.solver.multiobjective` instead, which provides more versatile
        multi-objective training capabilities. Instantiation of this solver will raise a
        deprecation warning.
    """

    num_behavior_features: int = 3
    renormalize_features: bool = False
    output_mode: Literal["overlapping", "separate"] = "overlapping"
    ignore_deprecation_warning: bool = False

    @property
    def num_time_features(self):
        return self.num_total_features - self.num_behavior_features

    @property
    def num_total_features(self):
        return self.model.num_output

    def __post_init__(self):
        super().__post_init__()
        if not self.ignore_deprecation_warning:
            warnings.warn(
                "MultiobjectiveSolver is deprecated since CEBRA 0.6.0 and will be removed in a future version. "
                "Use the new functionality in cebra.solver.multiobjective instead, which is more versatile. "
                "If you see this warning when using the scikit-learn interface, no action is required.",
                DeprecationWarning,
                stacklevel=2)
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


class AuxiliaryVariableSolver(Solver):

    @torch.no_grad()
    def transform(self,
                  inputs: torch.Tensor,
                  pad_before_transform: bool = True,
                  session_id: Optional[int] = None,
                  batch_size: Optional[int] = None,
                  use_reference_model: bool = False) -> torch.Tensor:
        """Compute the embedding.
        This function by default use ``model`` that was trained to encode the positive
        and negative samples. To use ``reference_model`` instead of ``model``
        ``use_reference_model`` should be equal ``True``.
        Args:
            inputs: The input signal
            use_reference_model: Flag for using ``reference_model``
        Returns:
            The output embedding.
        """
        self._check_is_fitted()
        model, offset = self._select_model(
            inputs, session_id, use_reference_model=use_reference_model)

        if len(offset) < 2 and pad_before_transform:
            pad_before_transform = False

        model.eval()
        return self._transform(model=model,
                               inputs=inputs,
                               pad_before_transform=pad_before_transform,
                               offset=offset,
                               batch_size=batch_size)
