#
# Regularized contrastive learning implementation.
#
# Not licensed yet. Distribution for review.
# Code will be open-sourced upon publication.
#
import itertools
from typing import List

import torch
from torch import nn

import cebra.models
import cebra.models.model as cebra_models_base


def create_multiobjective_model(module, **kwargs) -> "MultiobjectiveModel":
    assert isinstance(module, cebra_models_base.Model)
    if isinstance(module, cebra.models.ConvolutionalModelMixin):
        return MultiobjectiveConvolutionalModel(module=module, **kwargs)
    else:
        return MultiobjectiveModel(module=module, **kwargs)


def check_slices_for_gaps(slice_list):
    slice_list = sorted(slice_list, key=lambda s: s.start)
    for i in range(1, len(slice_list)):
        if slice_list[i - 1].stop < slice_list[i].start:
            raise ValueError(
                f"There is a gap in the slices {slice_list[i-1]} and {slice_list[i]}"
            )


def check_overlapping_feature_ranges(slice_list):
    for slice1, slice2 in itertools.combinations(slice_list, 2):
        if slice1.start < slice2.stop and slice1.stop > slice2.start:
            return True
    return False


def compute_renormalize_ranges(feature_ranges, sort=True):

    max_slice_dim = max(s.stop for s in feature_ranges)
    min_slice_dim = min(s.start for s in feature_ranges)
    full_emb_slice = slice(min_slice_dim, max_slice_dim)

    n_full_emb_slices = sum(1 for s in feature_ranges if s == full_emb_slice)

    if n_full_emb_slices > 1:
        raise ValueError(
            "There are more than one slice that cover the full embedding.")

    if n_full_emb_slices == 0:
        raise ValueError(
            "There are overlapping slices but none of them cover the full embedding."
        )

    rest_of_slices = [s for s in feature_ranges if s != full_emb_slice]
    max_slice_dim_rest = max(s.stop for s in rest_of_slices)
    min_slice_dim_rest = min(s.start for s in rest_of_slices)

    remaining_slices = []
    if full_emb_slice.start < min_slice_dim_rest:
        remaining_slices.append(slice(full_emb_slice.start, min_slice_dim_rest))

    if full_emb_slice.stop > max_slice_dim_rest:
        remaining_slices.append(slice(max_slice_dim_rest, full_emb_slice.stop))

    if len(remaining_slices) == 0:
        raise ValueError(
            "The behavior slices and the time slices coincide completely.")

    final_slices = remaining_slices + rest_of_slices

    if sort:
        final_slices = sorted(final_slices, key=lambda s: s.start)

    check_slices_for_gaps(final_slices)
    return final_slices


class _Norm(nn.Module):
    """Normalize the input tensor across its first dimension.

    TODO:
        * Move this class to ``cebra.models.layers``.
    """

    def forward(self, inp):
        """Normalize the input tensor across its first dimension."""
        return inp / torch.norm(inp, dim=1, keepdim=True)


class MultiobjectiveModel(nn.Module):
    """Wrapper around contrastive learning models to all training with multiple objectives

    Multi-objective training splits the last layer's feature representation into multiple
    chunks, which are then used for individual training objectives.

    Args:
        module: The module to wrap
        dimensions: A tuple of dimension values to extract from the model's feature embedding.
        renormalize: If True, the individual feature slices will be re-normalized before
            getting returned---this option only makes sense in conjunction with a loss based
            on the cosine distance or dot product.
    TODO:
        - Update nn.Module type annotation for ``module`` to cebra.models.Model
    """

    def __init__(self,
                 module: nn.Module,
                 feature_ranges: List[slice],
                 renormalize: bool,
                 split_outputs: bool = True):
        super().__init__()

        if not isinstance(module, cebra.models.Model):
            raise ValueError("Can only wrap models that are subclassing the "
                             "cebra.models.Model abstract base class. "
                             f"Got a model of type {type(module)}.")

        self.module = module
        self.renormalize = renormalize
        self._norm = _Norm()
        self.feature_ranges = feature_ranges
        self.split_outputs = split_outputs

        max_slice_dim = max(s.stop for s in self.feature_ranges)
        min_slice_dim = min(s.start for s in self.feature_ranges)
        if min_slice_dim != 0:
            raise ValueError(
                f"The first slice should start at 0, but it starts at {min_slice_dim}."
            )

        if max_slice_dim != self.num_output:
            raise ValueError(
                f"The dimension of ouput {self.num_output} is different than the highest dimension of slices {max_slice_dim}."
                f"They need to have the same dimension.")

        check_slices_for_gaps(self.feature_ranges)

        if check_overlapping_feature_ranges(self.feature_ranges):
            print("Computing renormalize ranges...")
            self.renormalize_ranges = compute_renormalize_ranges(
                self.feature_ranges, sort=True)
            print("New ranges:", self.renormalize_ranges)

    def set_split_outputs(self, val):
        assert isinstance(val, bool)
        self.split_outputs = val

    @property
    def get_offset(self):
        """See :py:meth:`cebra.models.model.Model.get_offset`."""
        return self.module.get_offset

    @property
    def num_output(self):
        """See :py:attr:`cebra.models.model.Model.num_output`."""
        return self.module.num_output

    def forward(self, inputs):
        """Compute multiple embeddings for a single signal input.

        Args:
            inputs: The input tensor

        Returns:
            A tuple of tensors which are sliced according to `self.feature_ranges`
            if `renormalize` is set to true, each of the tensors will be normalized
            across the first (feature) dimension.
        """

        output = self.module(inputs)

        if (not self.renormalize) and (not self.split_outputs):
            return output

        if self.renormalize:
            if hasattr(self, "renormalize_ranges"):
                #TODO: does the order of the renormalize ranges matter??
                # I think it does, imagine that the renormalize ranges are (5, 10), (0, 5), then
                # when we do torch.cat() output will be wrong --> Renormalize ranges need to be ordered.
                output = [
                    self._norm(output[:, slice_features])
                    for slice_features in self.renormalize_ranges
                ]
            else:
                output = [
                    self._norm(output[:, slice_features])
                    for slice_features in self.feature_ranges
                ]

            output = torch.cat(output, dim=1)

        if self.split_outputs:
            return tuple(output[:, slice_features]
                         for slice_features in self.feature_ranges)
        else:
            assert isinstance(output, torch.Tensor)
            return output


class MultiobjectiveConvolutionalModel(MultiobjectiveModel,
                                       cebra_models_base.ConvolutionalModelMixin
                                      ):
    pass
