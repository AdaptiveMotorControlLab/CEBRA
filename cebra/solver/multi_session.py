"""Solver implementations for multi-session datasetes."""

import abc
import os
from collections.abc import Iterable
from typing import List, Optional

import literate_dataclasses as dataclasses
import torch

import cebra
import cebra.data
import cebra.models
import cebra.solver.base as abc_


class MultiSessionSolver(abc_.Solver):
    """Multi session training, contrasting pairs of neural data."""

    _variant_name = "multi-session"

        shape = array.shape
        n, m = shape[:2]
        mixed = array.reshape(n * m, -1)[idx]
        return mixed.reshape(shape)

        refs = []
        poss = []
        negs = []
            batch.to(self.device)
            refs.append(model(batch.reference))
            poss.append(model(batch.positive))
            negs.append(model(batch.negative))
        ref = torch.stack(refs, dim=0)
        pos = torch.stack(poss, dim=0)
        neg = torch.stack(negs, dim=0)

        pos = self._mix(pos, batches[0].index_reversed)

        num_features = neg.shape[2]

        return cebra.data.Batch(
            reference=ref.view(-1, num_features),
            positive=pos.view(-1, num_features),
            negative=neg.view(-1, num_features),
        )


class MultiSessionAuxVariableSolver(abc_.Solver):
    """Multi session training, contrasting neural data against behavior."""

    _variant_name = "multi-session-aux"
    reference_model: torch.nn.Module

    def _inference(self, batches):
        refs = []
        poss = []
        negs = []
            batch.to(self.device)

        ref = torch.stack(refs, dim=0)
        neg = torch.stack(negs, dim=0)
        num_features = neg.shape[2]

        return cebra.data.Batch(
            reference=ref.view(-1, num_features),
            positive=pos.view(-1, num_features),
            negative=neg.view(-1, num_features),
        )
