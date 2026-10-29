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
"""Trajectory contrastive learning using CEBRA's InfoNCE reduction."""

import math
from typing import Optional

import torch
from torch import nn

from cebra.models.criterions import ContrastiveLoss
from cebra.models.criterions import euclidean_similarity
from cebra.models.criterions import infonce_full_denominator


class KDEPotential(nn.Module):
    """Log kernel density on detached encoded support, with Scott bandwidth.

    Args:
        epsilon: Positive bandwidth, or ``None`` for Scott's isotropic rule.
        min_epsilon: Lower bound for an estimated bandwidth.
    """

    def __init__(self,
                 epsilon: Optional[float] = None,
                 min_epsilon: float = 1e-6):
        super().__init__()
        if epsilon is not None and (not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError("epsilon must be finite and positive.")
        if not math.isfinite(min_epsilon) or min_epsilon <= 0:
            raise ValueError("min_epsilon must be finite and positive.")
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.register_buffer(
            "bandwidth",
            torch.tensor(1.0 if epsilon is None else epsilon,
                         dtype=torch.float32))

    def forward(self, query, support):
        """Evaluate log density; gradients flow through queries only."""
        support = support.detach()
        count, dimension = support.shape
        if self.epsilon is None:
            epsilon = (2 * count**(-2 / (dimension + 4)) *
                       support.var(dim=0, unbiased=False).mean()).clamp_min(
                           self.min_epsilon)
        else:
            epsilon = self.bandwidth.to(query)
        _, distances = euclidean_similarity(query, query, support)
        return (torch.logsumexp(distances / epsilon, dim=1, keepdim=True) -
                math.log(count) - 0.5 * dimension * epsilon.log())


class MLPPotential(nn.Module):
    """Centered scalar potential owned by the contrastive criterion.

    Args:
        latent_dim: Number of encoded coordinates.
        num_hidden_units: Width of one hidden GELU layer.
    """

    def __init__(self, latent_dim: int, num_hidden_units: int = 32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(latent_dim, num_hidden_units),
                                     nn.GELU(), nn.Linear(num_hidden_units, 1))
        self.offset = nn.Parameter(torch.zeros(1))

    def forward(self, query, support=None):
        """Evaluate and center the potential over the query batch."""
        value = self.network(query)
        return value - value.mean() + self.offset


class DYSCOInfoNCE(ContrastiveLoss):
    """Full-denominator trajectory InfoNCE with candidate potential correction.

    The three comparison blocks are forward/encoded, backward/encoded and
    forward/backward. Every block uses the concatenated encoded, forward and
    backward negative pools. Only the first two blocks receive the potential.
    The backward/encoded block deliberately uses the reference time index in
    its temperature, matching the submitted DYSCO objective.

    Args:
        sample_interval: Positive time between observation nodes.
        temperature: Positive noise scale; squared before horizon scaling.
        negative_temperature: Negative noise scale; defaults to temperature.
        num_positive: Events sampled per reference window.
        num_negatives: Number of sampled negative events; defaults to batch size.
        candidate_mode: ``in_batch`` matches submission sampling, excluding the
            reference window row for each negative draw. ``independent`` uses
            the loader's separately sampled windows and independent times.
        bidirectional: Include all three blocks; False uses forward/encoded.
        potential: Optional module accepting query and detached support.
        seed: Seed for checkpointed, private event sampling.
    """

    def __init__(self,
                 sample_interval=0.02,
                 temperature=1.0,
                 negative_temperature=None,
                 num_positive=1,
                 num_negatives=None,
                 candidate_mode="in_batch",
                 bidirectional=True,
                 potential=None,
                 seed=42):
        super().__init__()
        negative_temperature = temperature if negative_temperature is None else negative_temperature
        for name, value in [("sample_interval", sample_interval),
                            ("temperature", temperature),
                            ("negative_temperature", negative_temperature)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive.")
        if num_positive is None:
            raise ValueError("num_positive must be a positive integer.")
        for name, value in [("num_positive", num_positive),
                            ("num_negatives", num_negatives)]:
            if value is not None and (isinstance(value, bool) or
                                      not isinstance(value, int) or value < 1):
                raise ValueError(name + " must be a positive integer.")
        if candidate_mode not in ("in_batch", "independent"):
            raise ValueError("candidate_mode must be in_batch or independent.")
        self.sample_interval = float(sample_interval)
        self.temperature = float(temperature)
        self.negative_temperature = float(negative_temperature)
        self.num_positive = num_positive
        self.num_negatives = num_negatives
        self.candidate_mode = candidate_mode
        self.bidirectional = bidirectional
        self.potential = potential
        self.register_buffer("_sampling_state",
                             torch.Generator().manual_seed(seed).get_state())

    def sample_indices(self, encoded, num_candidates):
        """Draw event indices without consuming the global Torch generator."""
        batch, views, nodes, _ = encoded.shape
        if batch < 2 and self.candidate_mode == "in_batch":
            raise ValueError(
                "in_batch sampling requires at least two reference windows.")
        generator = torch.Generator().set_state(self._sampling_state.cpu())

        def draw(low, high, count):
            return torch.randint(low, high, (count,), generator=generator)

        count = self.num_negatives or batch
        references = torch.arange(batch).repeat_interleave(self.num_positive)
        if self.candidate_mode == "in_batch":
            offsets = draw(1, batch, count)
            negative_rows = (torch.arange(count) * batch // count +
                             offsets) % batch
        else:
            negative_rows = draw(0, num_candidates, count)
        view_offsets = draw(0, views, len(references))
        reference_views = draw(0, views, len(references))
        negative_views = draw(0,
                              views if self.candidate_mode == "in_batch" else 1,
                              count)
        times = draw(0, nodes, len(references))
        negative_times = draw(0, nodes, count)
        support = draw(0, batch * views * nodes, 2 * views * batch)
        self._sampling_state.copy_(generator.get_state().to(
            self._sampling_state.device))
        result = dict(reference=torch.stack(
            (references, reference_views, times)),
                      positive=torch.stack(
                          (references, (reference_views + view_offsets) % views,
                           times)),
                      negative=torch.stack(
                          (negative_rows, negative_views, negative_times)),
                      support=support)
        return {key: value.to(encoded.device) for key, value in result.items()}

    def forward(self, ref, pos, neg, indices=None):
        """Reduce trajectory comparisons to CEBRA's standard loss/statistics tuple.

        Args:
            ref: Encoded windows, shape (batch, view, time, latent).
            pos: Forward and optional backward trajectories, shape
                (direction, batch, view, time, latent).
            neg: Candidate trajectories, shape (estimate, window, time, latent),
                ordered encoded, forward, backward. In in_batch mode the window
                dimension is the flattened batch/view axis of ref.
            indices: Optional explicit event/support indices for numerical audits.
        """
        directions = 2 if self.bidirectional else 1
        if ref.ndim != 4 or ref.shape[-2] < 2 or pos.shape != (directions,
                                                               *ref.shape):
            raise ValueError(
                "Expected aligned encoded and predicted trajectory tensors.")
        if neg.ndim != 4 or neg.shape[0] != directions + 1 or neg.shape[
                2:] != ref.shape[2:] or neg.shape[1] < 1:
            raise ValueError(
                "Candidate trajectories must match time and latent dimensions.")
        if self.candidate_mode == "in_batch" and neg.shape[
                1] != ref.shape[0] * ref.shape[1]:
            raise ValueError(
                "in_batch candidates must contain every reference view.")
        indices = self.sample_indices(
            ref, neg.shape[1]) if indices is None else indices
        r, p, n = (tuple(indices[key])
                   for key in ("reference", "positive", "negative"))
        negative_windows = n[0] * ref.shape[1] + n[
            1] if self.candidate_mode == "in_batch" else n[0]
        encoded, forward = ref, pos[0]
        references, positives = [forward[r]], [encoded[p]]
        if self.bidirectional:
            backward = pos[1]
            references.extend([backward[r], forward[r]])
            positives.extend([encoded[p], backward[p]])
        candidates = torch.cat(
            [estimate[negative_windows, n[2]] for estimate in neg], dim=0)
        references, positives = torch.cat(references), torch.cat(positives)
        positive_scores, negative_scores = euclidean_similarity(
            references, positives, candidates)
        horizon = r[2].clamp_min(1)
        horizons = torch.cat(
            [horizon, horizon,
             torch.full_like(horizon, ref.shape[2] -
                             1)]) if self.bidirectional else horizon
        scale = 2 * self.sample_interval * horizons
        positive_scores = positive_scores / (self.temperature**2 * scale)
        negative_scores = negative_scores / (self.negative_temperature**2 *
                                             scale[:, None])
        if self.potential is not None:
            support = ref.reshape(-1,
                                  ref.shape[-1])[indices["support"]].detach()
            corrected = len(r[0]) * (2 if self.bidirectional else 1)
            positive_scores = torch.cat(
                (positive_scores[:corrected] - self.potential(
                    positives[:corrected], support=support).flatten(),
                 positive_scores[corrected:]))
            negative_scores = torch.cat(
                (negative_scores[:corrected] -
                 self.potential(candidates, support=support).T,
                 negative_scores[corrected:]))
        return infonce_full_denominator(positive_scores, negative_scores)
