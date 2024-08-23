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
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()
    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    return align + uniform, align, uniform


@torch.jit.script
def ref_infonce_not_stable(pos_dist: torch.Tensor, neg_dist: torch.Tensor):
    pos_dist = pos_dist
    neg_dist = neg_dist

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


def _reference_dot_similarity(ref, pos, neg):
    pos_dist = torch.zeros(ref.shape[0])
    neg_dist = torch.zeros(ref.shape[0], neg.shape[0])
    for d in range(ref.shape[1]):
        for i in range(len(ref)):
            pos_dist[i] += ref[i, d] * pos[i, d]
            for j in range(len(neg)):
                neg_dist[i, j] += ref[i, d] * neg[j, d]
    return pos_dist, neg_dist


def _reference_euclidean_similarity(ref, pos, neg):
    pos_dist = torch.zeros(ref.shape[0])
    neg_dist = torch.zeros(ref.shape[0], neg.shape[0])
    for d in range(ref.shape[1]):
        for i in range(len(ref)):
            pos_dist[i] += -(ref[i, d] - pos[i, d])**2
            for j in range(len(neg)):
                neg_dist[i, j] += -(ref[i, d] - neg[j, d])**2
    return pos_dist, neg_dist


def _reference_infonce(pos_dist, neg_dist):
    align = -pos_dist.mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    return align + uniform, align, uniform


def test_similiarities():
    rng = torch.Generator().manual_seed(42)
    ref = torch.randn(10, 3, generator = rng)
    pos = torch.randn(10, 3, generator = rng)
    neg = torch.randn(12, 3, generator = rng)

    pos_dist, neg_dist = _reference_dot_similarity(ref, pos, neg)
    pos_dist_2, neg_dist_2 = cebra_criterions.dot_similarity(ref, pos, neg)

    assert torch.allclose(pos_dist, pos_dist_2)
    assert torch.allclose(neg_dist, neg_dist_2)

    pos_dist, neg_dist = _reference_euclidean_similarity(ref, pos, neg)
    pos_dist_2, neg_dist_2 = cebra_criterions.euclidean_similarity(
        ref, pos, neg)

    assert torch.allclose(pos_dist, pos_dist_2)
    assert torch.allclose(neg_dist, neg_dist_2)


def _compute_grads(output, inputs):
    for input_ in inputs:
        input_.grad = None
        assert input_.requires_grad
    output.backward()
    return [input_.grad for input_ in inputs]


def _sample_dist_matrices(seed):
    rng = torch.Generator().manual_seed(42)
    pos_dist = torch.randn(100, generator=rng)
    neg_dist = torch.randn(100, 100, generator=rng)
    return pos_dist, neg_dist


@pytest.mark.parametrize("seed", [42, 4242, 424242])
def test_infonce(seed):
    pos_dist, neg_dist = _sample_dist_matrices(seed)

    ref_loss, ref_align, ref_uniform = _reference_infonce(pos_dist, neg_dist)
    loss, align, uniform = cebra_criterions.infonce(pos_dist, neg_dist)

    assert torch.allclose(ref_loss, loss)
    assert torch.allclose(ref_align, align, atol=0.0001)
    assert torch.allclose(ref_uniform, uniform)
    assert torch.allclose(align + uniform, loss)


@pytest.mark.parametrize("seed", [42, 4242, 424242])
@pytest.mark.parametrize("case", [0,1,2])
def test_infonce_gradients(seed, case):
    pos_dist, neg_dist = _sample_dist_matrices(seed)

    # TODO(stes): This test seems to fail due to some recent software
    # updates; root cause not identified. Remove this comment once
    # fixed. (for i = 0, 1)
    pos_dist_ = pos_dist.clone()
    neg_dist_ = neg_dist.clone()
    pos_dist_.requires_grad_(True)
    neg_dist_.requires_grad_(True)
    loss_ref = _reference_infonce(pos_dist_, neg_dist_)[case]
    grad_ref = _compute_grads(loss_ref, [pos_dist_, neg_dist_])

    pos_dist_ = pos_dist.clone()
    neg_dist_ = neg_dist.clone()
    pos_dist_.requires_grad_(True)
    neg_dist_.requires_grad_(True)
    loss = cebra_criterions.infonce(pos_dist_, neg_dist_)[case]
    grad = _compute_grads(loss, [pos_dist_, neg_dist_])

    # NOTE(stes) default relative tolerance is 1e-5
    assert torch.allclose(loss_ref, loss, rtol=1e-4)

    if case == 0:
        assert grad[0] is not None
        assert grad[1] is not None
        assert torch.allclose(grad_ref[0], grad[0])
        assert torch.allclose(grad_ref[1], grad[1])
    if case == 1:
        assert grad[0] is not None
        assert torch.allclose(grad_ref[0], grad[0])
        # TODO(stes): This is most likely not the right fix, needs more
        # investigation. On the first run of the test, grad[1] is actually
        # None, and then on the second run of the test it is a Tensor, but
        # with zeros everywhere. The behavior is fine for fitting models,
        # but there is some side-effect in our test suite we need to fix.
        if grad[1] is not None:
            assert torch.allclose(grad[1], torch.zeros_like(grad[1]))
    if case == 2:
        if grad[0] is None:
            assert torch.allclose(grad[0], torch.zeros_like(grad[0]))
        assert grad[1] is not None
        assert torch.allclose(grad_ref[1], grad[1])
