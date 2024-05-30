#
# Regularized contrastive learning implementation.
#
# Not licensed yet. Distribution for review.
# Code will be open-sourced upon publication.
#
"""Multiobjective contrastive learning."""

import abc
import logging
import os
import time
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import literate_dataclasses as dataclasses
import numpy as np
import torch
import tqdm

import cebra
import cebra.data
import cebra.io
import cebra.models
import cebra.solver.base as abc_
from cebra.solver import register
from cebra.solver.base import Solver
from cebra.solver.util import Meter


class MultiObjectiveConfig:
    """Configuration class for setting up multi-objective learning with Cebra.

    Args:
        loader: Data loader used for configurations.
    """

    def __init__(self, loader):
        self.loader = loader
        self.total_info = []
        self.current_info = {}

    def _check_overwriting_key(self, key):
        if key in self.current_info:
            warnings.warn(
                f"Configuration key already exists. Overwriting existing value. "
                f"If you don't want to overwrite you should call push() before."
            )

    def _check_pushed_status(self):
        if "slice" not in self.current_info:
            raise RuntimeError(
                "Slice configuration is missing. Add it before pushing it.")
        if "distributions" not in self.current_info:
            raise RuntimeError(
                "Distributions configuration is missing. Add it before pushing it."
            )
        if "losses" not in self.current_info:
            raise RuntimeError(
                "Losses configuration is missing. Add it before pushing it.")

    def set_slice(self, start, end):
        """Select the index range of the embedding.

        The configured loss will be applied to the ``start:end`` slice of the
        embedding space. Make sure the selected dimensionality is appropriate
        for the chosen loss function and distribution.
        """
        self._check_overwriting_key("slice")
        self.current_info['slice'] = (start, end)

    def set_loss(self, loss_name, **kwargs):
        """Select the loss function to apply.

        Select a valid loss function from :py:mod:`cebra.models.criterions`.
        Common choices are:

        - `FixedEuclideanInfoNCE`
        - `FixedCosineInfoNCE`

        which can be passed as string values to ``loss_name``. The loss
        will be applied to the range specified with ``set_slice``.
        """
        self._check_overwriting_key("losses")
        self.current_info["losses"] = {"name": loss_name, "kwargs": kwargs}

    def set_distribution(self, distribution_name, **kwargs):
        """Select the distribution to sample from.

        The loss function specified in ``set_loss`` is applied to positive
        and negative pairs sampled from the specified distribution.
        """
        self._check_overwriting_key("distributions")
        self.current_info["distributions"] = {
            "name": distribution_name,
            "kwargs": kwargs
        }

    def push(self):
        """Add a slice/loss/distribution setting to the config.

        After calling all of ``set_slice``, ``set_loss``, ``set_distribution``,
        add this group to the config by calling this function.

        Once all configuration parts are pushed, call ``finalize`` to finish
        the configuration.
        """
        self._check_pushed_status()
        print(f"Adding configuration for slice: {self.current_info['slice']}")
        self.total_info.append(self.current_info)
        self.current_info = {}

    def finalize(self):
        """Finalize the multiobjective configuration."""
        self.losses = []
        self.feature_ranges = []
        self.feature_ranges_tuple = []

        for info in self.total_info:
            self._process_info(info)

        if len(set(self.feature_ranges_tuple)) != len(
                self.feature_ranges_tuple):
            raise RuntimeError(
                f"Feature ranges are not unique. Please check again and remove the duplicates. "
                f"Feature ranges: {self.feature_ranges_tuple}")

        print("Creating MultiCriterion")
        self.criterion = cebra.models.MultiCriterions(losses=self.losses,
                                                      mode="contrastive")

    def _process_info(self, info):
        """
        Processes individual configuration info and updates the losses and feature ranges.

        Args:
            info (dict): The configuration info to process.
        """
        slice_info = info["slice"]
        losses_info = info["losses"]
        distributions_info = info["distributions"]

        self.losses.append(
            dict(indices=(slice_info[0], slice_info[1]),
                 contrastive_loss=dict(name=losses_info['name'],
                                       kwargs=losses_info['kwargs'])))

        self.feature_ranges.append(slice(slice_info[0], slice_info[1]))
        self.feature_ranges_tuple.append((slice_info[0], slice_info[1]))

        print(f"Adding distribution of slice: {slice_info}")
        self.loader.add_config(
            dict(distribution=distributions_info["name"],
                 kwargs=distributions_info["kwargs"]))


@dataclasses.dataclass
class MultiobjectiveSolverBase(Solver):

    feature_ranges: List[slice] = None
    renormalize: bool = None
    log: Dict[Tuple,
              List[float]] = dataclasses.field(default_factory=lambda: ({}))
    use_sam: bool = False
    regularizer: torch.nn.Module = None
    metadata: Dict = dataclasses.field(default_factory=lambda: ({
        "timestamp": None,
        "batches_seen": None,
    }))

    def __post_init__(self):
        super().__post_init__()

        self.model = cebra.models.create_multiobjective_model(
            module=self.model,
            feature_ranges=self.feature_ranges,
            renormalize=self.renormalize,
        )

    def fit(self,
            loader: cebra.data.Loader,
            valid_loader: cebra.data.Loader = None,
            *,
            valid_frequency: int = None,
            log_frequency: int = None,
            save_hook: Callable[[int, "Solver"], None] = None,
            scheduler_regularizer: "Scheduler" = None,
            scheduler_loss: "Scheduler" = None,
            logger: logging.Logger = None):
        """Train model for the specified number of steps.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            valid_loader: Data loader used for validation of the model.
            valid_frequency: The frequency for running validation on the ``valid_loader`` instance.
            logdir:  The logging directory for writing model checkpoints. The checkpoints
                can be read again using the `solver.load` function, or manually via loading the
                state dict.
            save_hook: callback. It will be called when we run validation.
            log_frequency: how frequent we log things.
            logger: logger to log progress. None by default.

        """

        def _run_validation():
            stats_val = self.validation(valid_loader, logger=logger)
            if save_hook is not None:
                save_hook(solver=self, step=num_steps)
            return stats_val

        self.to(loader.device)

        iterator = self._get_loader(loader,
                                    logger=logger,
                                    log_frequency=log_frequency)
        self.model.train()
        for num_steps, batch in iterator:
            weights_regularizer = None
            if scheduler_regularizer is not None:
                weights_regularizer = scheduler_regularizer.get_weights(
                    step=num_steps)
                # NOTE(stes): Both SAM and Jacobian regularization is not yet supported.
                # For this, we need to re-implement the closure logic below (right now,
                # the closure function applies the non-regularized loss in the second
                # step, it is unclear if that is the correct behavior.
                assert not self.use_sam

            weights_loss = None
            if scheduler_loss is not None:
                weights_loss = scheduler_loss.get_weights()

            stats = self.step(batch,
                              weights_regularizer=weights_regularizer,
                              weights_loss=weights_loss)

            self._update_metadata(num_steps)
            iterator.set_description(stats)
            run_validation = (valid_loader
                              is not None) and (num_steps % valid_frequency
                                                == 0)
            if run_validation:
                _run_validation()

        #TODO
        #_run_validation()

    def _get_loader(self, loader, **kwargs):
        return super()._get_loader(loader)

    def _update_metadata(self, num_steps):
        self.metadata["timestamp"] = time.time()
        self.metadata["batches_seen"] = num_steps

    def compute_regularizer(self, predictions, inputs):
        regularizer = []
        for prediction in predictions:
            R = self.regularizer(inputs, prediction.reference)
            regularizer.append(R)

        return regularizer

    def create_closure(self, batch, weights_loss):

        def inner_closure():
            predictions = self._inference(batch)
            losses = self.criterion(predictions)

            if weights_loss is not None:
                assert len(weights_loss) == len(
                    losses
                ), "Number of weights should match the number of losses"
                losses = [
                    weight * loss for weight, loss in zip(weights_loss, losses)
                ]

            loss = sum(losses)
            loss.backward()
            return loss

        return inner_closure

    def step(self,
             batch: cebra.data.Batch,
             weights_loss: Optional[List[float]] = None,
             weights_regularizer: Optional[List[float]] = None) -> dict:
        """Perform a single gradient update with multiple objectives."""

        closure = None
        if self.use_sam:
            closure = self.create_closure(batch, weights_loss)

        if weights_regularizer is not None:
            assert isinstance(batch.reference, torch.Tensor)
            batch.reference.requires_grad_(True)

        predictions = self._inference(batch)
        losses = self.criterion(predictions)

        for i, loss_value in enumerate(losses):
            key = "loss_train", i
            self.log.setdefault(key, []).append(loss_value.item())

        if weights_loss is not None:
            losses = [
                weight * loss for weight, loss in zip(weights_loss, losses)
            ]

        loss = sum(losses)

        if weights_regularizer is not None:
            regularizer = self.compute_regularizer(predictions=predictions,
                                                   inputs=batch.reference)
            assert len(weights_regularizer) == len(regularizer) == len(losses)
            loss = loss + sum(
                weight * reg
                for weight, reg in zip(weights_regularizer, regularizer))

        loss.backward()
        self.optimizer.step(closure)
        self.optimizer.zero_grad()

        if weights_regularizer is not None:
            for i, (weight,
                    reg) in enumerate(zip(weights_regularizer, regularizer)):
                assert isinstance(weight, float)
                self.log.setdefault(("regularizer", i), []).append(reg.item())
                self.log.setdefault(("regularizer_weight", i),
                                    []).append(weight)

        if weights_loss is not None:
            for i, weight in enumerate(weights_loss):
                assert isinstance(weight, float) or isinstance(weight, int)
                self.log.setdefault(("loss_weight", i), []).append(weight)

        # add sum_loss_train
        self.log.setdefault(("sum_loss_train",), []).append(loss.item())
        return {"sum_loss_train": loss.item()}

    @torch.no_grad()
    def _compute_metrics(self):
        # NOTE: We set split_outputs = False when we compute
        # validation metrics, otherwise it returns a tuple
        # which led to a bug before.
        embeddings = {}
        self.model.set_split_outputs(False)
        for split in self.metrics.splits:
            embedding_tensor = self.transform(
                self.metrics.datasets[split].neural)
            embedding_np = embedding_tensor.cpu().numpy()
            assert embedding_np.shape[1] == self.model.num_output
            embeddings[split] = embedding_np

        self.model.set_split_outputs(True)
        return self.metrics.compute_metrics(embeddings)

    @torch.no_grad()
    def validation(
        self,
        loader: cebra.data.Loader,
        logger=None,
        weights_loss: Optional[List[float]] = None,
    ):
        self.model.eval()
        total_loss = Meter()

        losses_dict = {}
        for _, batch in enumerate(loader):
            predictions = self._inference(batch)
            losses = self.criterion(predictions)

            if weights_loss is not None:
                assert len(weights_loss) == len(
                    losses
                ), "Number of weights should match the number of losses"
                losses = [
                    weight * loss for weight, loss in zip(weights_loss, losses)
                ]

            total_loss.add(sum(losses).item())

            for i, loss_value in enumerate(losses):
                key = "loss_val", i
                losses_dict.setdefault(key, []).append(loss_value.item())

        losses_dict_mean = {k: np.mean(v) for k, v in losses_dict.items()}
        stats_val = {**losses_dict_mean}

        if self.metrics is not None:
            metrics = self._compute_metrics()
            stats_val.update(metrics)

        for key, value in stats_val.items():
            self.log.setdefault(key, []).append(value)

        if logger is not None:
            formatted_loss = ', '.join([
                f"{'_'.join(map(str, key))}:{value:.3f}"
                for key, value in stats_val.items()
                if key[0].startswith("loss")
            ])
            formatted_r2 = ', '.join([
                f"{'_'.join(map(str, key))}:{value:.3f}"
                for key, value in stats_val.items()
                if key[0].startswith("r2")
            ])
            logger.info(f"Val: {formatted_loss}")
            logger.info(f"Val: {formatted_r2}")

        # add sum_loss_valid
        sum_loss_valid = total_loss.average
        self.log.setdefault(("sum_loss_val",), []).append(sum_loss_valid)
        return stats_val

    @torch.no_grad()
    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        offset = self.model.get_offset()
        self.model.eval()
        X = inputs.cpu().numpy()
        X = np.pad(X, ((offset.left, offset.right - 1), (0, 0)), mode="edge")
        X = torch.from_numpy(X).float().to(self.device)

        if isinstance(self.model.module, cebra.models.ConvolutionalModelMixin):
            # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
            X = X.transpose(1, 0).unsqueeze(0)
            outputs = self.model(X)

            # switch back from (1, C, T) -> (T, C)
            if isinstance(outputs, torch.Tensor):
                assert outputs.dim() == 3 and outputs.shape[0] == 1
                outputs = outputs.squeeze(0).transpose(1, 0)
            elif isinstance(outputs, tuple):
                assert all(tensor.dim() == 3 and tensor.shape[0] == 1
                           for tensor in outputs)
                outputs = (
                    output.squeeze(0).transpose(1, 0) for output in outputs)
                outputs = tuple(outputs)
            else:
                raise ValueError("Invalid condition in solver.transform")
        else:
            # Standard evaluation, (T, C, dt)
            outputs = self.model(X)

        return outputs


class SupervisedMultiobjectiveSolverxCEBRA(MultiobjectiveSolverBase):
    """Supervised neural network training with MSE loss"""

    _variant_name = "supervised-solver-xcebra"

    def _inference(self, batch):
        """Compute predictions (discrete/continuous) for the batch."""
        pred_refs = self.model(batch.reference)
        prediction_batches = []
        for i, label_data in enumerate(batch.positive):
            prediction_batches.append(
                cebra.data.Batch(reference=pred_refs[i],
                                 positive=label_data,
                                 negative=None))
        return prediction_batches


@register("multiobjective-solver")
@dataclasses.dataclass
class ContrastiveMultiobjectiveSolverxCEBRA(MultiobjectiveSolverBase):

    _variant_name = "contrastive-solver-xcebra"

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        pred_refs = self.model(batch.reference)
        pred_negs = self.model(batch.negative)

        prediction_batches = []
        for i, positive in enumerate(batch.positive):
            pred_pos = self.model(positive)
            prediction_batches.append(
                cebra.data.Batch(pred_refs[i], pred_pos[i], pred_negs[i]))

        return prediction_batches
