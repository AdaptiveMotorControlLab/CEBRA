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
import abc
import random
from typing import List, Tuple, Union

import numpy as np
import torch

__all__ = [
    "Mask", "RandomNeuronMask", "RandomTimestepMask", "NeuronBlockMask",
    "TimeBlockMask"
]


class Mask:

    def __init__(self, masking_value: Union[float, List[float], Tuple[float]]):
        self._check_masking_parameters(masking_value)

    @abc.abstractmethod
    def apply_mask(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _select_masking_params():
        raise NotImplementedError

    def _check_masking_parameters(self, masking_value: Union[float, List[float],
                                                             Tuple[float]]):
        """
        The masking values are the masking ratio to apply.
        It can be a single ratio, a list of ratio that will be picked randomly or
        a tuple of (min, max, step_size) that will be used to create a list of ratios
        from which to sample randomly.
        """
        if isinstance(masking_value, float):
            assert 0.0 < masking_value < 1.0, (
                f"Masking ratio {masking_value} for {self.__name__()} "
                "should be between 0.0 and 1.0.")

        elif isinstance(masking_value, list):
            assert all(isinstance(ratio, float) for ratio in masking_value), (
                f"Masking ratios {masking_value} for {self.__name__()} "
                "should be between 0.0 and 1.0.")
            assert all(0.0 < ratio < 1.0 for ratio in masking_value), (
                f"Masking ratios {masking_value} for {self.__name__()} "
                "should be between 0.0 and 1.0.")

        elif isinstance(masking_value, tuple):
            assert len(masking_value) == 3, (
                f"Masking ratios {masking_value} for {self.__name__()} "
                "should be a tuple of (min, max, step).")
            assert 0.0 <= masking_value[0] < masking_value[1] <= 1.0, (
                f"Masking ratios {masking_value} for {self.__name__()} "
                "should be between 0.0 and 1.0.")
            assert masking_value[2] < masking_value[1] - masking_value[0], (
                f"Masking step {masking_value[2]} for {self.__name__()} "
                "should be between smaller than the diff between min "
                f"({masking_value[0]}) and max ({masking_value[1]}).")

        else:
            raise ValueError(
                f"Masking ratio {masking_value} for {self.__name__()} "
                "should be a float, list of floats or a tuple of (min, max, step)."
            )


class RandomNeuronMask(Mask):

    def __init__(self, masking_value: Union[float, List[float], Tuple[float]]):
        super().__init__(masking_value)
        self.mask_ratio = masking_value

    def __name__(self):
        return "RandomNeuronMask"

    def apply_mask(self, data: torch.Tensor) -> torch.Tensor:
        """ Apply random masking on the neuron dimension.

        Args:
            data: batch of size (batch_size, n_neurons, offset).
            mask_ratio: Proportion of neurons to mask. Default value 0.3 comes
                from the MtM paper: https://arxiv.org/pdf/2407.14668v2

        Returns:
            torch.Tensor: The mask, a tensor of the same size as the input data with the
                masked neurons set to 1.
        """
        batch_size, n_neurons, offset_length = data.shape
        mask_ratio = self._select_masking_params()

        # Random mask: shape [batch_size, n_neurons], different per batch and neurons
        masked = torch.rand(batch_size, n_neurons,
                            device=data.device) < mask_ratio
        return (~masked).int().unsqueeze(2).expand(
            -1, -1, offset_length)  # Expand to all timesteps

    def _select_masking_params(self) -> float:
        """
        The masking values are the masking ratio to apply.
        It can be a single ratio, a list of ratio that will be picked randomly or
        a tuple of (min, max, step_size) that will be used to create a list of ratios
        from which to sample randomly.
        """
        if isinstance(self.mask_ratio, float):
            selected_value = self.mask_ratio

        elif isinstance(self.mask_ratio, list):
            selected_value = random.choice(self.mask_ratio)

        elif isinstance(self.mask_ratio, tuple):
            min_val, max_val, step_size = self.mask_ratio
            selected_value = random.choice(
                np.arange(min_val, max_val + step_size, step_size).tolist())

        else:
            raise ValueError(
                f"Masking ratio {self.mask_ratio} for {self.__name__()} "
                "should be a float, list of floats or a tuple of (min, max, step)."
            )

        return selected_value


class RandomTimestepMask(Mask):

    def __init__(self, masking_value: Union[float, List[float], Tuple[float]]):
        super().__init__(masking_value)
        self.mask_ratio = masking_value

    def __name__(self):
        return "RandomTimestepMask"

    def apply_mask(self, data: torch.Tensor) -> torch.Tensor:
        """ Apply random masking on the time dimension.

        Args:
            data: batch of size (batch_idx, feature_dim, seq_len). With seq_len
                corresponding to the offset.
            mask_ratio: Proportion of timesteps masked. Not necessarliy consecutive.
                Default value 0.3 comes from the MtM paper: https://arxiv.org/pdf/2407.14668v2

        Returns:
            torch.Tensor: The mask, a tensor of the same size as the input data with the
                masked neurons set to 1.

        """
        batch_idx, n_neurons, offset_length = data.shape
        mask_ratio = self._select_masking_params()

        # Random mask: shape [batbatch_idxch_size, offset_length], different per batch and timestamp
        masked = torch.rand(batch_idx, offset_length,
                            device=data.device) < mask_ratio
        return (~masked).int().unsqueeze(1).expand(-1, n_neurons,
                                                   -1)  # Expand to all neurons

    def _select_masking_params(self) -> float:
        """
        The masking values are the masking ratio to apply.
        It can be a single ratio, a list of ratio that will be picked randomly or
        a tuple of (min, max, step_size) that will be used to create a list of ratios
        from which to sample randomly.
        """
        if isinstance(self.mask_ratio, float):
            selected_value = self.mask_ratio

        elif isinstance(self.mask_ratio, list):
            selected_value = random.choice(self.mask_ratio)

        elif isinstance(self.mask_ratio, tuple):
            min_val, max_val, step_size = self.mask_ratio
            selected_value = random.choice(
                np.arange(min_val, max_val + step_size, step_size).tolist())

        else:
            raise ValueError(
                f"Masking ratio {self.mask_ratio} for {self.__name__()} "
                "should be a float, list of floats or a tuple of (min, max, step)."
            )

        return selected_value


class NeuronBlockMask(Mask):

    def __init__(self, masking_value: Union[float, List[float], Tuple[float]]):
        super().__init__(masking_value)
        self.mask_prop = masking_value

    def __name__(self):
        return "NeuronBlockMask"

    def apply_mask(self, data: torch.Tensor) -> torch.Tensor:
        """ Apply masking to a contiguous block of neurons.

        Args:
            data: batch of size (batch_size, n_neurons, offset).
            self.mask_prop: Proportion of neurons to mask. The neurons are masked in a
                contiguous block.

        Returns:
            torch.Tensor: The mask, a tensor of the same size as the input data with the
                masked neurons set to 1.
        """
        batch_size, n_neurons, offset_length = data.shape

        mask_prop = self._select_masking_params()
        num_mask = int(n_neurons * mask_prop)
        mask = torch.ones((batch_size, n_neurons),
                          dtype=torch.int,
                          device=data.device)

        if num_mask == 0:
            return mask.unsqueeze(2)

        for batch_idx in range(batch_size):  # Create a mask for each batch
            # Select random the start index for the block of neurons to mask
            start_idx = torch.randint(0, n_neurons - num_mask + 1, (1,)).item()
            end_idx = min(start_idx + num_mask, n_neurons)
            mask[batch_idx, start_idx:end_idx] = 0  # set masked neurons to 0

        return mask.unsqueeze(2).expand(
            -1, -1, offset_length)  # Expand to all timesteps

    def _select_masking_params(self) -> float:
        """
        The masking values are the masking ratio to apply.
        It can be a single ratio, a list of ratio that will be picked randomly or
        a tuple of (min, max, step_size) that will be used to create a list of ratios
        from which to sample randomly.
        """
        if isinstance(self.mask_prop, float):
            selected_value = self.mask_prop

        elif isinstance(self.mask_prop, list):
            selected_value = random.choice(self.mask_prop)

        elif isinstance(self.mask_prop, tuple):
            min_val, max_val, step_size = self.mask_prop
            selected_value = random.choice(
                np.arange(min_val, max_val + step_size, step_size).tolist())

        else:
            raise ValueError(
                f"Masking ratio {self.mask_prop} for {self.__name__()} "
                "should be a float, list of floats or a tuple of (min, max, step)."
            )

        return selected_value


class TimeBlockMask(Mask):

    def __init__(self, masking_value: Union[float, List[float], Tuple[float]]):
        super().__init__(masking_value)
        self.sampled_rate, self.masked_seq_len = masking_value

    def __name__(self):
        return "TimeBlockMask"

    def apply_mask(self, data: torch.Tensor) -> torch.Tensor:
        """ Apply contiguous block masking on the time dimension.

        When choosing which block of timesteps to mask, each timestep is considered
        a candidate starting time-step with probability ``self.sampled_rate`` where
        ``self.masked_seq_len`` is the length of each masked span starting from the respective
        time step. Sampled starting time steps are expanded to length ``self.masked_seq_len``
        and spans can overlap. Inspirede by the wav2vec 2.0 masking strategy.

        Default values from the wav2vec paper: https://arxiv.org/abs/2006.11477.

        Args:
            data (torch.Tensor): The input tensor of shape (batch_size, seq_len, feature_dim).
            self.sampled_rate (float): The probability of each time-step being a candidate for masking.
            self.masked_seq_len (int): The length of each masked span starting from the sampled time-step.

        Returns:
            torch.Tensor: A boolean mask of shape (batch_size, seq_len) where True
                indicates masked positions.
        """
        batch_size, n_neurons, offset_length = data.shape

        sampled_rate, masked_seq_len = self._select_masking_params()

        num_masked_starting_points = int(offset_length * sampled_rate)
        mask = torch.ones((batch_size, offset_length),
                          dtype=int,
                          device=data.device)
        for batch_idx in range(batch_size):
            # Sample starting points for masking in the current batch
            start_indices = torch.randperm(
                offset_length, device=data.device)[:num_masked_starting_points]

            # Apply masking spans
            for start in start_indices:
                end = min(start + masked_seq_len, offset_length)
                mask[batch_idx, start:end] = 0  # set masked timesteps to 0

        return mask.unsqueeze(1).expand(-1, n_neurons,
                                        -1)  # Expand to all neurons

    def _check_masking_parameters(self, masking_value: Union[float, List[float],
                                                             Tuple[float]]):
        """
        The masking values are the parameters for the timeblock masking.
        It needs to be a tuple of (sampled_rate, masked_seq_len)
        sampled_rate: The probability of each time-step being a candidate for masking.
        masked_seq_len: The length of each masked span starting from the sampled time-step.
        """
        assert isinstance(masking_value, tuple) and len(masking_value) == 2, (
            f"Masking parameters {masking_value} for {self.__name__()} "
            "should be a tuple of (sampled_rate, masked_seq_len).")
        assert 0.0 < masking_value[0] < 1.0 and isinstance(
            masking_value[0], float), (
                f"Masking parameters {masking_value} for {self.__name__()} "
                "should be between 0.0 and 1.0.")
        assert masking_value[1] > 0 and isinstance(masking_value[1], int), (
            f"Masking parameters {masking_value} for {self.__name__()} "
            "should be an integer greater than 0.")

    def _select_masking_params(self) -> float:
        """
        The masking values are the masking ratio to apply.
        It can be a single ratio, a list of ratio that will be picked randomly or
        a tuple of (min, max, step_size) that will be used to create a list of ratios
        from which to sample randomly.
        """
        return self.sampled_rate, self.masked_seq_len
