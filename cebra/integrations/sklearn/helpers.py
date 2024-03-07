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
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch


def _get_min_max(
        labels: List[Union[npt.NDArray, torch.Tensor]]) -> Tuple[float, float]:
    """Get the min and max values in the list of labels.

    Args:
        labels: A list of arrays corresponding to the labels associated to each embeddings.

    Returns:
        The minimum value (first returns) and the maximum value (second returns).

    """
    min = float("inf")
    max = float("-inf")
    for label in labels:
        if any(isinstance(l, str) for l in label):
            raise ValueError(
                f"Invalid labels dtype, expect floats or integers, got string")
        min = np.min(label) if min > np.min(label) else min
        max = np.max(label) if max < np.max(label) else max
    return min, max


def _mean_bin(data: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """Compute the mean across samples in `data` for the given mask.

    Args:
        data: Array of data to compute mean over.
        mask: Boolean array indicating which elements to include.

    Returns:
        Mean of the data across samples over the specified mask.
    """
    if mask.sum() == 0:
        return None
    else:
        return data[mask].mean(axis=0)


def _coarse_to_fine(
    data: Union[npt.NDArray, torch.Tensor],
    digitized_labels: Union[npt.NDArray, torch.Tensor],
    bin_idx: int,
) -> npt.NDArray:
    """Compute a quantized feature from the data using the given digitized labels.

    Args:
        data: Array of data to compute quantized feature over.
        digitized: Array of digitized labels to find the indexes in the data for quantization.
        bin_idx: Integer to which values from the labels that are close enough to are grouped
            to be averaged.

    Returns:
        Quantized data for the specified bin.
    """
    for difference in [0, 1, 2]:
        mask = abs(digitized_labels - bin_idx) <= difference
        quantized_data = _mean_bin(data, mask)
        if quantized_data is not None:
            return quantized_data
    raise ValueError(
        f"Digitalized labels does not have elements close enough to bin index {bin_idx}. "
        f"The bin index should be in the range of the labels values.")


def align_embeddings(
    embeddings: List[Union[npt.NDArray, torch.Tensor]],
    labels: List[Union[npt.NDArray, torch.Tensor]],
    normalize: bool = True,
    n_bins: int = 100,
) -> List[Union[npt.NDArray, torch.Tensor]]:
    """Align the embeddings in the ``embeddings`` list to the ``labels``.

    Each embedding has an associated set of labels. During alignment, the labels are
    digitalized so that all sets of digitalized labels contain the same set of values.
    Then the embeddings are quantized based on the new digitalized labels.

    Args:
        embeddings: List of embeddings to align on the labels.
        labels: List of labels corresponding to each embedding and to use for alignment
            between them.
        normalize: If True, samples of the embeddings are normalized across dimensions.
        n_bins: Number of values for the digitalized common labels.

    Returns:
        The embeddings aligned on the labels.
    """
    if n_bins < 1:
        raise ValueError(
            f"Invalid value for n_bins, expect a value at least equals to 1, got {n_bins}."
        )

    quantized_embeddings = []

    # get min and max label values to set the digitalization bins
    min_labels_value, max_labels_value = _get_min_max(labels)

    for embedding, label in zip(embeddings, labels):
        if len(embedding) != len(label):
            raise ValueError(
                "Labels invalid, the labels associated to an embedding should "
                f"contain the same number of samples as its embedding, got "
                f"{len(embeddings)} samples in embedding and {len(label)} samples in label"
            )
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        # Only keep non-NaNs values
        valid_ = np.isnan(embedding).any(axis=1)
        valid_embedding = embedding[~valid_]
        valid_labels = label[~valid_]

        # Digitize the labels so that values are contained into a common set of values
        digitized_labels = np.digitize(
            valid_labels, np.linspace(min_labels_value, max_labels_value,
                                      n_bins))

        # quantize embedding based on the new labels
        quantized_embedding = [
            _coarse_to_fine(valid_embedding, digitized_labels, bin_idx)
            for bin_idx in range(n_bins)[1:]
        ]

        if normalize:  # normalize across dimensions
            quantized_embedding_norm = [
                quantized_sample / np.linalg.norm(quantized_sample, axis=0)
                for quantized_sample in quantized_embedding
            ]

        quantized_embeddings.append(quantized_embedding_norm)
    return quantized_embeddings
