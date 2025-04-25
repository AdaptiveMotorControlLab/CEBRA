import random
from typing import Dict, Optional

import torch

import cebra.data.mask as mask


class MaskedMixin:

    # Initialization so that no maskins is applied
    masks = []  # a list of Mask instances

    def set_masks(self, masking: Optional[Dict[str, float]] = None) -> None:
        """Set the mask type and probability for the dataset.

        Args:
            masking (Dict[str, float]): A dictionary of masking types and their
                corresponding required masking values. The keys are the names
                of the Mask instances.

        Note:
            By default, no masks are applied.
        """
        if masking is not None:
            for mask_key in self.masking:
                if mask_key in mask.__all__:
                    cls = getattr(mask, mask_key)
                    self.masks.append(cls(masking[mask_key]))
                else:
                    raise ValueError(
                        f"Mask type {mask_key} not supported. Supported types are {self.masking.keys()}"
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

        # If masks is empty, return the data as is
        if not self.masks:
            return data

        sampled_mask = random.choice(self.masks)
        mask = sampled_mask.apply_mask(data)

        num_chunks = (data.shape[0] + chunk_size -
                      1) // chunk_size  # Compute number of chunks

        for i in range(num_chunks):
            start, end = i * chunk_size, min((i + 1) * chunk_size,
                                             data.shape[0])
            data[start:end].mul_(
                mask[start:end])  # apply mask in-place to save memory

        return data
