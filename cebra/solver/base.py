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
import torch
import tqdm

import cebra
import cebra.data
import cebra.io
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

    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    history: List = dataclasses.field(default_factory=list)
    log: Dict = dataclasses.field(default_factory=lambda: ({

    def __post_init__(self):
        cebra.io.HasDevice.__init__(self)
        return {
            "model": self.model.state_dict(),
            "loss": torch.tensor(self.history),
            "version": cebra.__version__,
            "log": self.log,
        }

        Args:
        """



        if _contains("criterion"):
            self.criterion.load_state_dict(_get("criterion"))
        # TODO(stes): This will be deprecated at some point; the "log" attribute
        # holds the same information.

    @property
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

            valid_loader: Data loader used for validation of the model.
            valid_frequency: The frequency for running validation on the ``valid_loader`` instance.

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
            run_validation = (valid_loader
                              is not None) and (num_steps % valid_frequency
                                                == 0)
            if run_validation:
                validation_loss = self.validation(valid_loader)
                if self.best_loss is None or validation_loss < self.best_loss:
                    self.best_loss = validation_loss
            if save_model:
                if decode:
                    self.decode_history.append(
                        self.decoding(loader, valid_loader))
                if save_hook is not None:
                    save_hook(num_steps, self)


            Dictionary containing training metrics.
        self.optimizer.zero_grad()
        prediction = self._inference(batch)
        loss, align, uniform = self.criterion(prediction.reference,
                                              prediction.positive,
                                              prediction.negative)

        loss.backward()
        self.optimizer.step()
            self.log[key].append(value)

        total_loss = Meter()
            total_loss.add(loss.item())
    @torch.no_grad()
        """Deprecated since 0.0.2."""
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the embedding.

        This function by default only applies the ``forward`` function
        of the given model, after switching it into eval mode.

        Args:
            inputs: The input signal

        Returns:
            The output embedding.

        TODO:
            * Remove eval mode
        """

        self.model.eval()
        return self.model(inputs)

    @abc.abstractmethod
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.
            ``batch.index`` should be set to ``None``.
        raise NotImplementedError


@dataclasses.dataclass
class MultiobjectiveSolver(Solver):
    """Train models to satisfy multiple learning objectives.

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
            self.model,
            output_mode=self.output_mode,

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
        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())
