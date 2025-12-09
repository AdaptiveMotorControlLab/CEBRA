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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

__all__ = [
    "MaskedMixin", "Mask", "RandomNeuronMask", "RandomTimestepMask",
    "NeuronBlockMask", "TimeBlockMask", "SessionBlockMask"
]


class MaskedMixin:
    """A mixin class for applying masking to data.

    Note:
        This class is designed to be used as a mixin for other classes.
        It provides functionality to apply masking to data.
        The `set_masks` method should be called to set the masking types
        and their corresponding probabilities.
    """
    _masks = []  # a list of Mask instances

    def set_masks(
        self,
        masking: Optional[Union[
            Dict[str, Any],
            List[Union[Tuple[str, Any], Dict[str, Any]]],
        ]] = None,
        apply_multiple_masks: bool = False,
        fill_with_noise: bool = False,
    ) -> None:
        """Set the mask types and parameters for the dataset.

        Supports two input formats:
        - Dict[str, params]: One instance per mask type; replaces any previous
          instance of the same class (backward compatible).
        - List[ (name, params) | {name: params} ]: Adds one instance per entry,
          allowing multiple instances of the same mask with different params.

        Args:
            masking: Mask configuration. See formats above.
            apply_multiple_masks: When True, `apply_mask` will randomly pick a
                number of configured masks and apply them all in a single call
                (combined). When False, a single mask is sampled per call
                (default/legacy behavior).
            fill_with_noise: When True, masked positions are filled with Gaussian
                noise sampled from per-neuron mean and std (computed from unmasked
                values). When False, masked positions are zeroed (default/legacy behavior).

        Note:
            - By default, no masks are applied.
            - When using the dict format, existing instances of the same mask
              class are replaced. When using the list format, entries are
              appended, allowing duplicates with different parameters.
        """
        # Store strategy
        self._apply_multiple_masks = bool(apply_multiple_masks)
        self._fill_with_noise = bool(fill_with_noise)

        if masking is None:
            return

        # Helper to resolve and create a mask instance by name and params
        def _create_mask(mask_key: str, params: Any) -> None:
            # First try exact match
            if mask_key in globals():
                cls = globals()[mask_key]
                self._masks.append(cls(params))
                return

            # Try fuzzy match: find mask classes that start with mask_key
            # e.g., "SessionBlockMask2" -> match "SessionBlockMask"
            for available_mask in __all__:
                if available_mask == "MaskedMixin" or available_mask == "Mask":
                    continue
                if mask_key.startswith(available_mask):
                    cls = globals()[available_mask]
                    self._masks.append(cls(params))
                    return

            # No match found
            raise ValueError(
                f"Mask type {mask_key} not supported. Supported types are {__all__}"
            )

        # Dict mode: replace-by-class semantics (backward compatible)
        if isinstance(masking, dict):
            for mask_key, params in masking.items():
                # Resolve the actual mask class (supports fuzzy matching)
                cls = None
                if mask_key in globals():
                    cls = globals()[mask_key]
                else:
                    # Try fuzzy match
                    for available_mask in __all__:
                        if available_mask == "MaskedMixin" or available_mask == "Mask":
                            continue
                        if mask_key.startswith(available_mask):
                            cls = globals()[available_mask]
                            break

                if cls is None:
                    raise ValueError(
                        f"Mask type {mask_key} not supported. Supported types are {__all__}"
                    )

                # Remove previous instances of this class, then add one
                self._masks = [m for m in self._masks if not isinstance(m, cls)]
                _create_mask(mask_key, params)
            return

        # List mode: append semantics; allow duplicates of the same mask class
        if isinstance(masking, list):
            for entry in masking:
                if isinstance(entry, tuple) and len(entry) == 2:
                    mask_key, params = entry
                    _create_mask(mask_key, params)
                elif isinstance(entry, dict):
                    if len(entry) != 1:
                        raise ValueError(
                            "Each dict entry in the masking list must have exactly one key."
                        )
                    mask_key, params = next(iter(entry.items()))
                    _create_mask(mask_key, params)
                else:
                    raise ValueError(
                        "List masking entries must be either (name, params) tuples or {name: params} dicts."
                    )
            return

        # Unsupported type
        raise ValueError(
            "masking must be either a dict{name: params} or a list of (name, params)/{name: params} entries."
        )

    def apply_mask(self,
                   data: torch.Tensor,
                   chunk_size: int = 1000) -> torch.Tensor:
        """Apply masking to the input data.

        Note:
            - By default, no masking. Else apply masking on the input data.
            - Only one masking type can be applied at a time, but multiple
                masking types can be set so that it alternates between them
                across iterations.
            - Masking is applied to the data in chunks to avoid memory issues.

        Args:
            data (torch.Tensor): batch of size (batch_size, num_neurons, offset).
            chunk_size (int): Number of rows to process at a time.

        Returns:
            torch.Tensor: The masked data.
        """
        if data.dim() == 2:
            data = data.unsqueeze(0)

        if data.dim() != 3:
            raise ValueError(
                f"Data must be a 3D tensor, but got {data.dim()}D tensor.")
        if data.dtype != torch.float32:
            raise ValueError(
                f"Data must be a float32 tensor, but got {data.dtype}.")

        # If masks is empty, return the data as is
        if not self._masks:
            return data

        # Compute the mask to apply (single or combined)
        if getattr(self, "_apply_multiple_masks", False):
            # Randomly choose how many masks to apply (at least one, at most all)
            num_to_apply = random.randint(1, len(self._masks))
            selected_masks = random.sample(self._masks, num_to_apply)

            # Start with all-ones mask and combine multiplicatively (logical AND)
            mask = torch.ones_like(data, dtype=torch.int)
            for m in selected_masks:
                mask = mask.mul(m.apply_mask(data))
        else:
            sampled_mask = random.choice(self._masks)
            mask = sampled_mask.apply_mask(data)

        # Apply masking strategy: zeros or Gaussian noise
        if getattr(self, "_fill_with_noise", False):
            # Fill masked positions with per-neuron Gaussian noise
            # mask: (batch_size, n_neurons, offset) with 1=keep, 0=mask
            inverse_mask = (mask == 0)  # True where we need to fill

            # Compute per-neuron mean and std from unmasked values
            # Shape: (batch_size, n_neurons)
            masked_data = data * mask  # Zero out masked positions
            sum_unmasked = masked_data.sum(dim=2)  # Sum over time
            count_unmasked = mask.sum(dim=2).clamp(
                min=1)  # Count valid timesteps
            mean_per_neuron = sum_unmasked / count_unmasked  # (batch_size, n_neurons)

            # Compute std
            squared_diff = ((data - mean_per_neuron.unsqueeze(2))**2) * mask
            var_per_neuron = squared_diff.sum(dim=2) / count_unmasked
            std_per_neuron = var_per_neuron.sqrt().clamp(
                min=1e-6)  # Avoid div by zero

            # Generate Gaussian noise and fill masked positions
            noise = torch.randn_like(data) * std_per_neuron.unsqueeze(
                2) + mean_per_neuron.unsqueeze(2)
            data = torch.where(inverse_mask, noise, data)
        else:
            # Legacy behavior: multiply by mask (zeros out masked positions)
            num_chunks = (data.shape[0] + chunk_size -
                          1) // chunk_size  # Compute number of chunks

            for i in range(num_chunks):
                start, end = i * chunk_size, min((i + 1) * chunk_size,
                                                 data.shape[0])
                data[start:end].mul_(mask[start:end])  # in-place for memory

        return data


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

        # Random mask: shape [batch_idx, offset_length], different per batch and timestamp
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


class SessionBlockMask(Mask):

    def __init__(self, masking_value: Union[List[int], Tuple[List[int], int]]):
        super().__init__(masking_value)
        # Handle both list and tuple (list, num_sessions) formats
        if isinstance(masking_value, tuple):
            self.num_neurons = masking_value[0]
            self.fixed_num_sessions = masking_value[1]
        else:
            self.num_neurons = masking_value
            self.fixed_num_sessions = None

    def __name__(self):
        return "SessionBlockMask"

    def apply_mask(self, data: torch.Tensor) -> torch.Tensor:
        """ Apply masking to contiguous blocks of neurons corresponding to sessions.

        Args:
            data: batch of size (batch_size, n_neurons, offset).
            self.mask_prop: Proportion of neurons to mask. The neurons are masked in a
                contiguous block.

        Returns:
            torch.Tensor: The mask, a tensor of the same size as the input data with the
                masked neurons set to 1.
        """
        batch_size, n_neurons, offset_length = data.shape

        if sum(self.num_neurons) != n_neurons:
            raise ValueError(
                f"Sum of num_neurons {sum(self.num_neurons)} defined at init does not match "
                f"the number of neurons in the data {n_neurons}.")

        num_mask, start_idx = self._select_masking_params()
        mask = torch.ones((batch_size, n_neurons),
                          dtype=torch.int,
                          device=data.device)

        for i in range(len(num_mask)):
            end_idx = min(start_idx[i] + num_mask[i], n_neurons)
            mask[:, start_idx[i]:end_idx] = 0  # set masked neurons to 0

        return mask.unsqueeze(2).expand(
            -1, -1, offset_length)  # Expand to all timesteps

    def _select_masking_params(self) -> Tuple[List[int], List[int]]:
        """
        Select which sessions to mask.
        If fixed_num_sessions is set (from tuple input), use that exact number.
        Otherwise, randomly pick between 1 and len(num_neurons)-1 sessions.
        """
        if isinstance(self.num_neurons, list):
            # Determine number of sessions to mask
            if self.fixed_num_sessions is not None:
                select_num_sessions = self.fixed_num_sessions
            else:
                # Select the # of sessions to mask randomly
                select_num_sessions = random.randint(1,
                                                     len(self.num_neurons) - 1)

            # Select session indices to mask
            select_idxs = random.sample(range(len(self.num_neurons)),
                                        select_num_sessions)
            num_mask, start_idx = [], []
            for select_idx in select_idxs:
                num_mask.append(self.num_neurons[select_idx])
                start_idx.append(sum(self.num_neurons[:select_idx]))
        else:
            raise ValueError(
                f"Number of neurons {self.num_neurons} for {self.__name__()} "
                "should be a list of the number of neurons in each session.")

        return num_mask, start_idx

    def _check_masking_parameters(self, masking_value: Union[List[int],
                                                             Tuple[List[int],
                                                                   int]]):
        """
        The masking values are the number of neurons per session to mask.
        It can be:
        - A list of positive integers (one per session)
        - A tuple of (list of positive integers, int) where the int is the fixed number of sessions to mask
        """
        if isinstance(masking_value, tuple):
            # Validate tuple format: (list, int)
            if len(masking_value) != 2:
                raise ValueError(
                    f"Tuple format for {self.__name__()} should be (list, int), "
                    f"got tuple of length {len(masking_value)}.")
            neuron_list, num_sessions = masking_value

            # Validate the list part
            if not isinstance(neuron_list, list):
                raise ValueError(
                    f"First element of tuple for {self.__name__()} should be a list, "
                    f"got {type(neuron_list)}.")
            if not all(isinstance(n, int) for n in neuron_list) or any(
                    n <= 0 for n in neuron_list):
                raise ValueError(
                    f"Neuron list {neuron_list} for {self.__name__()} "
                    "should be a list of positive integers.")

            # Validate the int part
            if not isinstance(num_sessions, int) or num_sessions <= 0:
                raise ValueError(
                    f"Second element of tuple for {self.__name__()} should be a positive integer, "
                    f"got {num_sessions}.")
            if num_sessions >= len(neuron_list):
                raise ValueError(
                    f"Number of sessions to mask ({num_sessions}) must be less than "
                    f"total number of sessions ({len(neuron_list)}).")

        elif isinstance(masking_value, list):
            # Validate list format
            if not all(isinstance(n, int) for n in masking_value) or any(
                    n <= 0 for n in masking_value):
                raise ValueError(
                    f"Number of neurons {masking_value} for {self.__name__()} "
                    "should be a list of positive integers.")
        else:
            raise ValueError(
                f"Masking value for {self.__name__()} should be either a list of positive integers "
                f"or a tuple of (list, int), got {type(masking_value)}.")


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
