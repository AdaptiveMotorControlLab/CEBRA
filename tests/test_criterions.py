import numpy as np
import pytest
import torch
from torch import nn

import cebra.models.criterions as cebra_criterions

def setup():
    ref = torch.randn(100).float().unsqueeze(1)
    pos = torch.randn(100).float().unsqueeze(1)
    neg = torch.randn(100).float().unsqueeze(1)
    return ref, pos, neg

def test_infonce(temperature, criterion):
    """Test infonce loss is computed correctly."""
    ref, pos, neg = setup()
    infonce = criterion(temperature=temperature)
    # perfect alignment
    loss1, _, _ = infonce(ref, ref, neg)
    # random alignment
    loss2, _, _ = infonce(ref, pos, neg)
    assert loss1 < loss2

    loss, _, _ = infonce(ref, pos, neg)
    loss_low, _, _ = infonce_lower_temp(ref, pos, neg)
    loss_high, _, _ = infonce_higher_temp(ref, pos, neg)
    assert not torch.allclose(loss, loss_high)
    assert not torch.allclose(loss, loss_low)


    ref, pos, neg = setup()

    # Check that temperature is trainable
    ref, pos, neg = setup()
    loss.backward()
    assert temp.grad is not None

def test_clipping():
    # Check that clipping works
    ref, pos, neg = setup()
def test_infonce_equivalence(temperature):
    ref, pos, neg = setup()
    infonce_old = cebra_criterions.InfoNCE(temperature=temperature)
                          loss_learnable.detach(),
    assert len(list(infonce_old.parameters())) == 0
    assert len(list(infonce_learnable.parameters())) == 1


def test_alias():
    assert cebra_criterions.InfoNCE == cebra_criterions.FixedCosineInfoNCE
    assert cebra_criterions.InfoMSE == cebra_criterions.FixedEuclideanInfoNCE
