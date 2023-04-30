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
import numpy as np
import pytest
import torch
from torch import nn

import cebra.models.criterions as cebra_criterions


@torch.jit.script
def ref_dot_similarity(ref: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                       temperature: float):
    pos_dist = torch.einsum("ni,ni->n", ref, pos) / temperature
    neg_dist = torch.einsum("ni,mi->nm", ref, neg) / temperature
    return pos_dist, neg_dist


@torch.jit.script
def ref_euclidean_similarity(ref: torch.Tensor, pos: torch.Tensor,
                             neg: torch.Tensor, temperature: float):
    ref_sq = torch.einsum("ni->n", ref**2) / temperature
    pos_sq = torch.einsum("ni->n", pos**2) / temperature
    neg_sq = torch.einsum("ni->n", neg**2) / temperature

    pos_cosine, neg_cosine = ref_dot_similarity(ref, pos, neg, temperature)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)

    return pos_dist, neg_dist


@torch.jit.script
def ref_infonce(pos_dist: torch.Tensor, neg_dist: torch.Tensor):
    with torch.no_grad():
        c, _ = neg_dist.max(dim=1)
    c = c.detach()
    pos_dist = pos_dist - c
    neg_dist = neg_dist - c

    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    return align + uniform, align, uniform


class ReferenceInfoNCE(nn.Module):
    """The InfoNCE loss.
    Attributes:
        temperature (float): The softmax temperature
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def _distance(self, ref, pos, neg):
        return ref_dot_similarity(ref, pos, neg, self.temperature)

    def forward(self, ref, pos, neg):
        pos_dist, neg_dist = self._distance(ref, pos, neg)
        return ref_infonce(pos_dist, neg_dist)


class ReferenceInfoMSE(ReferenceInfoNCE):
    """A variant of the InfoNCE loss using a MSE error.
    Attributes:
        temperature (float): The softmax temperature
    """

    def _distance(self, ref, pos, neg):
        return ref_euclidean_similarity(ref, pos, neg, self.temperature)


def setup():
    ref = torch.randn(100).float().unsqueeze(1)
    pos = torch.randn(100).float().unsqueeze(1)
    neg = torch.randn(100).float().unsqueeze(1)
    return ref, pos, neg


@pytest.mark.parametrize("temperature", [0.1, 1.0, 5.0])
@pytest.mark.parametrize(
    "criterion",
    [
        ReferenceInfoNCE,
        ReferenceInfoMSE,
        cebra_criterions.InfoNCE,
        cebra_criterions.InfoMSE,
        cebra_criterions.FixedCosineInfoNCE,
        cebra_criterions.FixedEuclideanInfoNCE,
        cebra_criterions.LearnableCosineInfoNCE,
        cebra_criterions.LearnableEuclideanInfoNCE,
    ],
)
def test_infonce(temperature, criterion):
    """Test infonce loss is computed correctly."""
    ref, pos, neg = setup()
    infonce = criterion(temperature=temperature)
    # perfect alignment
    loss1, _, _ = infonce(ref, ref, neg)
    # random alignment
    loss2, _, _ = infonce(ref, pos, neg)
    assert loss1 < loss2

    infonce_lower_temp = criterion(temperature=temperature / 2.0)
    infonce_higher_temp = criterion(temperature=temperature * 2.0)
    loss, _, _ = infonce(ref, pos, neg)
    loss_low, _, _ = infonce_lower_temp(ref, pos, neg)
    loss_high, _, _ = infonce_higher_temp(ref, pos, neg)
    assert not torch.allclose(loss, loss_high)
    assert not torch.allclose(loss, loss_low)


@pytest.mark.parametrize("temperature", [0.1, 1.0, 5.0])
@pytest.mark.parametrize(
    "criterion",
    [
        ReferenceInfoNCE,
        cebra_criterions.FixedCosineInfoNCE,
        cebra_criterions.FixedEuclideanInfoNCE,
    ],
)
def test_fixed(temperature, criterion):
    # Check that temperature is frozen when trainable option is False
    ref, pos, neg = setup()
    infonce_fixed = criterion(temperature=temperature)
    assert len(list(infonce_fixed.parameters())) == 0


@pytest.mark.parametrize("temperature", [0.1, 1.0, 5.0])
@pytest.mark.parametrize(
    "criterion",
    [
        cebra_criterions.LearnableCosineInfoNCE,
        cebra_criterions.LearnableEuclideanInfoNCE,
    ],
)
def test_trainable(temperature, criterion):
    # Check that temperature is trainable
    ref, pos, neg = setup()
    infonce_learnable = criterion(temperature=temperature)
    (temp,) = list(iter(infonce_learnable.parameters()))
    loss, _, _ = infonce_learnable(ref, pos, neg)
    loss.backward()
    assert temp.grad is not None


def test_clipping():
    # Check that clipping works
    ref, pos, neg = setup()
    clipped_infonce = cebra_criterions.LearnableCosineInfoNCE(
        temperature=100, min_temperature=0.1)
    nonclipped_infonce = cebra_criterions.LearnableCosineInfoNCE(
        temperature=100, min_temperature=None)
    assert torch.allclose(
        clipped_infonce(ref, pos, neg)[0],
        nonclipped_infonce(ref, pos, neg)[0])


@pytest.mark.parametrize("temperature", [0.1, 1.0, 5.0])
def test_infonce_equivalence(temperature):
    # Check if InfoNCE with non trainable option is equivalent to old implementation
    ref, pos, neg = setup()
    infonce_old = cebra_criterions.InfoNCE(temperature=temperature)
    infonce_learnable = cebra_criterions.LearnableCosineInfoNCE(
        temperature=temperature)
    loss_old, _, _ = infonce_old(ref, pos, neg)
    loss_learnable, _, _ = infonce_learnable(ref, pos, neg)
    assert torch.allclose(loss_old,
                          loss_learnable.detach(),
                          rtol=1e-04,
                          atol=1e-05)
    assert len(list(infonce_old.parameters())) == 0
    assert len(list(infonce_learnable.parameters())) == 1


@pytest.mark.parametrize("temperature", [0.1, 1.0, 5.0])
def test_infonce_reference_new_equivalence(temperature):
    # Check if InfoNCE with non trainable option is equivalent to reference implementation
    ref, pos, neg = setup()
    ref_infonce = ReferenceInfoNCE(temperature=temperature)
    new_infonce = cebra_criterions.InfoNCE(temperature=temperature)
    ref_infomse = ReferenceInfoMSE(temperature=temperature)
    new_infomse = cebra_criterions.InfoMSE(temperature=temperature)
    cosine_loss_old, _, _ = ref_infonce(ref, pos, neg)
    euclidean_loss_old, _, _ = ref_infomse(ref, pos, neg)
    cosine_loss_new, _, _ = new_infonce(ref, pos, neg)
    euclidean_loss_new, _, _ = new_infomse(ref, pos, neg)
    assert torch.allclose(cosine_loss_old,
                          cosine_loss_new,
                          rtol=1e-04,
                          atol=1e-05)
    assert torch.allclose(euclidean_loss_old,
                          euclidean_loss_new,
                          rtol=1e-04,
                          atol=1e-05)


def test_alias():
    assert cebra_criterions.InfoNCE == cebra_criterions.FixedCosineInfoNCE
    assert cebra_criterions.InfoMSE == cebra_criterions.FixedEuclideanInfoNCE
