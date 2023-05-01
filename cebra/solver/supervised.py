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
"""Solvers for supervised training

Note:
    It is inclear whether these will be kept. Consider the implementation
    as experimental/outdated, and the API for this particular package unstable.
"""
import abc
import os
from collections.abc import Iterable
from typing import List

import literate_dataclasses as dataclasses
import torch
import tqdm

import cebra
import cebra.data
import cebra.models
import cebra.solver.base as abc_


class SupervisedNNSolver(abc_.Solver):
    """Supervised neural network training with MSE loss"""

    _variant_name = "supervised-nn"

    def fit(self,
            loader: torch.utils.data.DataLoader,
            num_steps: int,
            valid_loader=None,
            *,
            save_frequency=None,
            valid_frequency=None,
            decode: bool = False,
            logdir: str = None):
        """Train model for the specified number of steps.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            save_frequency: If not `None`, the frequency for automatically saving model checkpoints
                to `logdir`.
            logdir:  The logging directory for writing model checkpoints. The checkpoints
                can be read again using the `solver.load` function, or manually via loading the
                state dict.
        """

        self.model.train()
        step_idx = 0
        while True:
            for _, batch in enumerate(loader):
                stats = self.step(batch)
                self._log_checkpoint(num_steps, loader, valid_loader)
                step_idx += 1
                if step_idx >= num_steps:
                    break

    def step(self, batch) -> dict:
        """Perform a single gradient update.

        Args:
            batch: The input samples

        Returns:
            Dictionary containing training metrics TODO
        """
        self.optimizer.zero_grad()
        prediction = self._inference(batch)
        loss = self.criterion(prediction, batch["label"].squeeze())
        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())
        return dict(total=loss.item())

    def _inference(self, batch):
        """Compute predictions (discrete/continuous) for the batch."""
        feature, prediction = self.model(batch["neural"])
        return prediction

    def validation(self, valid_loader):
        """Deprecated since 0.0.2."""
        total_loss = 0
        for batch in valid_loader:
            prediction = self._inference(batch)
            loss = self.criterion(prediction, batch["label"].squeeze())
            total_loss += loss.item()
        return total_loss
