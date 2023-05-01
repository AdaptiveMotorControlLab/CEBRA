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
"""Utility functions for solvers and their training loops."""

from collections.abc import Iterable
from typing import Dict

import literate_dataclasses as dataclasses
import tqdm


def _description(stats: Dict[str, float]):
    stats_str = [f"{key}: {value: .4f}" for key, value in stats.items()]
    return " ".join(stats_str)


class Meter:
    """Track statistics of a metric."""

    __slots__ = ["_num_elements", "_total"]

    def __init__(self):
        self._num_elements = 0
        self._total = 0

    def add(self, value: float, num_elements: int = 1):
        """Add the value to the meter.

        Args:
            value: The value to add to the meter.
            num_elements: Optional, if the value was already obtained
                by summing multiple elements (for example, loss values
                within a batch of data samples) and the average should
                be computed with respect to this unit.
        """
        self._total += value
        self._num_elements += num_elements

    @property
    def average(self) -> float:
        """Return the average value of the tracked metric."""
        if self._num_elements == 0:
            return float("nan")
        return self._total / float(self._num_elements)

    @property
    def sum(self) -> float:
        """Return the sum of all tracked values."""
        return self._total


@dataclasses.dataclass
class ProgressBar:
    "Log and display values during training."

    loader: Iterable
    log_format: str

    _valid_formats = ["tqdm", "off"]

    @property
    def use_tqdm(self) -> bool:
        """Display ``tqdm`` as the progress bar."""
        return self.log_format == "tqdm"

    def __post_init__(self):
        if self.log_format not in self._valid_formats:
            raise ValueError(
                f"log_format must be one of {self._valid_formats}, "
                f"but got {self.log_formats}")

    def __iter__(self):
        self.iterator = self.loader
        if self.use_tqdm:
            self.iterator = tqdm.tqdm(self.iterator)
        for num_batch, batch in enumerate(self.iterator):
            yield num_batch, batch

    def set_description(self, stats: Dict[str, float]):
        """Update the progress bar description.

        The description is updated by computing a formatted string
        from the given ``stats`` in the format ``{key}: {value: .4f}``
        with a space as the divider between dictionary elements.

        Behavior depends on the selected progress bar.
        """
        if self.use_tqdm:
            self.iterator.set_description(_description(stats))
