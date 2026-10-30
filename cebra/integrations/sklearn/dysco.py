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
"""Scikit-learn entry point for aligned-trial dynamics learning."""

import copy
import itertools
import math

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted

import cebra.data
import cebra.models
import cebra.solver
from cebra.dynamics.polynomial import PolynomialFlow
from cebra.integrations.sklearn.cebra import CEBRA
from cebra.integrations.sklearn.utils import check_device
from cebra.models.dysco import DYSCOInfoNCE
from cebra.models.dysco import KDEPotential
from cebra.models.dysco import MLPPotential


class DYSCO(CEBRA):
    """Learn an encoder and polynomial dynamics from explicitly aligned trials.

    Trials share an observation space and evenly sampled time. Alignment IDs
    group repetitions whose time indices already correspond. Optional controls
    are observed at every node; no controls are inferred. Heterogeneous sessions
    and irregular sampling are outside this interface.

    Args:
        model_architecture: Encoder name from the existing CEBRA registry.
            Defaults to an unnormalized Euclidean encoder.
        output_dimension: Number of latent coordinates.
        num_hidden_units: Encoder and optional potential hidden width.
        polynomial_degree: Maximum total state/control polynomial degree.
        sample_interval: Positive observation interval, in field time units.
        rollout_steps: Number of transitions per training window.
        integration_substeps: RK4 substeps per observation interval.
        num_views: Distinct aligned repetitions per window; one for singletons.
        num_positive: Events sampled per reference window.
        candidate_mode: ``in_batch`` for submitted candidate sampling, or
            ``independent`` for separately sampled negative windows.
        bidirectional: Full three-block objective, or forward-only ablation.
        potential: ``kde``, ``mlp``, or ``None``.
        potential_bandwidth: Fixed positive KDE bandwidth, or automatic if None.
        temperature: Positive noise scale, squared before horizon scaling.
        negative_temperature: Negative noise scale, defaults to temperature.
        batch_size: Reference window count; at least two for in_batch candidates.
        batch_size_negatives: Negative event count, defaults to batch_size.
        max_iterations: Optimizer updates per fit or partial_fit call.
        learning_rate: Adam learning rate.
        dynamics_learning_rate: Optional finite, nonnegative Adam rate for
            polynomial coefficients; defaults to learning_rate. Encoder and
            potential keep learning_rate.
        optimizer_kwargs: Adam options as key/value pairs, following CEBRA.
        device: CEBRA device string.
        random_state: Private initialization and sampling seed.
        verbose: Show the common solver progress bar.
        pad_before_transform: Preserve input length during embedding transforms.
    """

    def __init__(self,
                 model_architecture="offset1-model-mse",
                 output_dimension=3,
                 num_hidden_units=32,
                 polynomial_degree=2,
                 sample_interval=0.02,
                 rollout_steps=8,
                 integration_substeps=1,
                 num_views=2,
                 num_positive=1,
                 candidate_mode="in_batch",
                 bidirectional=True,
                 potential="kde",
                 potential_bandwidth=None,
                 temperature=1.0,
                 negative_temperature=None,
                 batch_size=64,
                 batch_size_negatives=None,
                 max_iterations=1000,
                 learning_rate=3e-4,
                 dynamics_learning_rate=None,
                 optimizer_kwargs=(("betas", (0.9, 0.999)), ("eps", 1e-8)),
                 device="cuda_if_available",
                 random_state=42,
                 verbose=False,
                 pad_before_transform=True):
        super().__init__(model_architecture=model_architecture,
                         output_dimension=output_dimension,
                         num_hidden_units=num_hidden_units,
                         batch_size=batch_size,
                         batch_size_negatives=batch_size_negatives,
                         max_iterations=max_iterations,
                         learning_rate=learning_rate,
                         optimizer_kwargs=optimizer_kwargs,
                         device=device,
                         verbose=verbose,
                         temperature=temperature,
                         distance="euclidean",
                         pad_before_transform=pad_before_transform)
        self.polynomial_degree = polynomial_degree
        self.sample_interval = sample_interval
        self.rollout_steps = rollout_steps
        self.integration_substeps = integration_substeps
        self.num_views = num_views
        self.num_positive = num_positive
        self.candidate_mode = candidate_mode
        self.bidirectional = bidirectional
        self.potential = potential
        self.potential_bandwidth = potential_bandwidth
        self.negative_temperature = negative_temperature
        self.random_state = random_state
        self.dynamics_learning_rate = dynamics_learning_rate

    def _prepare_criterion(self):
        if self.potential == "kde":
            potential = KDEPotential(self.potential_bandwidth)
        elif self.potential == "mlp":
            potential = MLPPotential(self.output_dimension,
                                     self.num_hidden_units)
        elif self.potential is None:
            potential = None
        else:
            raise ValueError("potential must be kde, mlp, or None.")
        return DYSCOInfoNCE(sample_interval=self.sample_interval,
                            temperature=self.temperature,
                            negative_temperature=self.negative_temperature,
                            num_positive=self.num_positive,
                            num_negatives=self.batch_size_negatives or
                            self.batch_size,
                            candidate_mode=self.candidate_mode,
                            bidirectional=self.bidirectional,
                            potential=potential,
                            seed=self.random_state)

    def _initialize(self, num_features, num_controls):
        if self.dynamics_learning_rate is not None and (
                not math.isfinite(self.dynamics_learning_rate) or
                self.dynamics_learning_rate < 0):
            raise ValueError(
                "dynamics_learning_rate must be finite and nonnegative.")
        self.device_ = check_device(self.device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.random_state)
            self.model_ = cebra.models.init(
                self.model_architecture,
                num_neurons=num_features,
                num_units=self.num_hidden_units,
                num_output=self.output_dimension).to(self.device_)
            if isinstance(self.model_, cebra.models.ResampleModelMixin):
                raise ValueError(
                    "DYSCO requires one embedding per observation node; resampling encoders are unsupported."
                )
            self.dynamics_model_ = PolynomialFlow(
                self.output_dimension,
                degree=self.polynomial_degree,
                dt=self.sample_interval,
                substeps=self.integration_substeps,
                control_dim=num_controls).to(self.device_)
            criterion = self._prepare_criterion().to(self.device_)
        optimizer = torch.optim.Adam([
            dict(params=itertools.chain(self.model_.parameters(),
                                        criterion.parameters())),
            dict(params=self.dynamics_model_.parameters(),
                 lr=self.learning_rate if self.dynamics_learning_rate is None
                 else self.dynamics_learning_rate)
        ],
                                     lr=self.learning_rate,
                                     **dict(self.optimizer_kwargs))
        self.solver_name_ = "aligned-trial-dysco"
        self.solver_ = cebra.solver.init(self.solver_name_,
                                         model=self.model_,
                                         dynamics_model=self.dynamics_model_,
                                         criterion=criterion,
                                         optimizer=optimizer,
                                         tqdm_on=self.verbose).to(self.device_)
        self.offset_ = self.model_.get_offset()
        self.control_dimension_ = num_controls
        self.label_types_ = []
        self.sampling_state_ = torch.Generator().manual_seed(
            self.random_state).get_state()
        self._fit_configuration = copy.deepcopy(self.get_params())

    def _fit_trials(self, X, alignment_ids, controls, reset, callback,
                    callback_frequency):
        if not reset:
            self._check_fit_configuration()
        device = check_device(self.device) if reset else self.device_
        dataset = cebra.data.AlignedTrialDataset(X,
                                                 alignment_ids,
                                                 self.sample_interval,
                                                 controls=controls,
                                                 device=device)
        num_controls = 0 if dataset.controls is None else dataset.controls.shape[
            -1]
        if reset:
            self._initialize(dataset.input_dimension, num_controls)
        elif dataset.input_dimension != self.n_features_in_ or num_controls != self.control_dimension_:
            raise ValueError(
                "Observation and control dimensions must match the fitted model."
            )
        self._configure_for_all(dataset, self.model_, is_multisession=False)
        loader = cebra.data.AlignedTrialLoader(
            dataset=dataset,
            num_steps=self.max_iterations,
            batch_size=self.batch_size,
            batch_size_negatives=self.batch_size_negatives,
            rollout_steps=self.rollout_steps,
            num_views=self.num_views,
            candidate_mode=self.candidate_mode,
            seed=self.random_state)
        loader.generator.set_state(self.sampling_state_.cpu())
        self._active_window_generator = loader.generator
        # Establish metadata before training so callbacks may save even on
        # the first fit call, through the standard checkpoint envelope.
        self.n_features_ = dataset.input_dimension
        self.n_features_in_ = dataset.input_dimension
        self.num_sessions_ = None
        self.solver_.tqdm_on = self.verbose
        try:
            self._partial_fit(self.solver_,
                              self.model_,
                              loader,
                              False,
                              callback=callback,
                              callback_frequency=callback_frequency)
        finally:
            self.sampling_state_ = loader.generator.get_state()
            del self._active_window_generator
        return self

    def _check_fit_configuration(self):
        """Reject changes that would reconstruct a different training model."""
        mutable = {"max_iterations", "verbose", "pad_before_transform"}
        changed = [
            name for name, value in self.get_params().items()
            if name not in mutable and value != self._fit_configuration[name]
        ]
        if changed:
            raise ValueError(
                "Fitted DYSCO configuration changed: " + ", ".join(changed) +
                ". Call fit to apply new parameters; use to() to move devices.")

    def fit(self,
            X,
            y=None,
            *,
            alignment_ids=None,
            controls=None,
            callback=None,
            callback_frequency=None):
        """Fit on trial arrays with explicit temporal alignment metadata.

        Args:
            X: Sequence of (time, observed_feature) arrays.
            y: Unused sklearn target; use alignment_ids explicitly.
            alignment_ids: Group label per trial; members must already align.
            controls: Optional sequence of (time, control_feature) arrays.
            callback: Common CEBRA solver callback.
            callback_frequency: Callback interval in optimizer steps.
        """
        if y is not None:
            raise ValueError(
                "Use alignment_ids for explicit aligned-trial metadata.")
        if alignment_ids is None:
            raise ValueError("alignment_ids must be supplied explicitly.")
        return self._fit_trials(X, alignment_ids, controls, True, callback,
                                callback_frequency)

    def partial_fit(self,
                    X,
                    y=None,
                    *,
                    alignment_ids=None,
                    controls=None,
                    callback=None,
                    callback_frequency=None):
        """Continue training, preserving optimizer and private sampler states."""
        if not hasattr(self, "n_features_"):
            return self.fit(X,
                            y,
                            alignment_ids=alignment_ids,
                            controls=controls,
                            callback=callback,
                            callback_frequency=callback_frequency)
        if y is not None or alignment_ids is None:
            raise ValueError(
                "Supply alignment_ids explicitly and leave y unset.")
        return self._fit_trials(X, alignment_ids, controls, False, callback,
                                callback_frequency)

    def fit_transform(self, X, y=None, *, alignment_ids=None, controls=None):
        """Fit aligned trials and return one embedding array per trial."""
        self.fit(X, y, alignment_ids=alignment_ids, controls=controls)
        return self.transform(X)

    def transform(self, X, batch_size=None, session_id=None):
        """Embed one recording or a sequence of trials with the shared encoder.

        A two-dimensional recording returns an array. A sequence of trial
        arrays, or a three-dimensional array, returns a list of arrays. Trial
        lengths may differ; all observations share the same feature space.
        """
        if getattr(X, "ndim", None) == 3 or (isinstance(X, (list, tuple)) and
                                             len(X) > 0 and
                                             getattr(X[0], "ndim", None) == 2):
            return [
                super(DYSCO, self).transform(trial,
                                             batch_size=batch_size,
                                             session_id=session_id)
                for trial in X
            ]
        return super().transform(X,
                                 batch_size=batch_size,
                                 session_id=session_id)

    def vector_field(self, states, controls=None):
        """Evaluate derivatives at latent states, returning a NumPy array."""
        check_is_fitted(self, "n_features_")
        reference = next(self.dynamics_model_.parameters())
        states = torch.as_tensor(states,
                                 dtype=reference.dtype,
                                 device=self.device_)
        controls = None if controls is None else torch.as_tensor(
            controls, dtype=reference.dtype, device=self.device_)
        with torch.no_grad():
            return self.dynamics_model_.field(states, controls).cpu().numpy()

    def equations(self,
                  precision=6,
                  threshold=0.0,
                  variable_names=None,
                  control_names=None):
        """Inspect the learned polynomial dynamics as a list of equation strings.

        ``print(*model.equations(), sep="\\n")`` prints the complete field.
        Names default to ``z0``, ``z1``, etc., and ``u0``, ``u1`` for controls.
        ``precision`` controls significant digits; ``threshold`` hides terms
        with absolute coefficients at most that value, without changing the
        model. Equations use learned latent coordinates and the time units of
        ``sample_interval``. They do not perform coordinate alignment or infer
        a sparse model. See
        :meth:`cebra.dynamics.polynomial.PolynomialField.equations` for arguments.
        """
        check_is_fitted(self, "n_features_")
        return self.dynamics_model_.field.equations(
            precision=precision,
            threshold=threshold,
            variable_names=variable_names,
            control_names=control_names)

    def rollout(self, states, num_steps, controls=None, direction="forward"):
        """Integrate latent initial states, including the initial node.

        Controls are supplied in chronological order. Backward integration starts
        at the final state and returns nodes in decreasing-time order.
        """
        check_is_fitted(self, "n_features_")
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward.")
        reference = next(self.dynamics_model_.parameters())
        states = torch.as_tensor(states,
                                 dtype=reference.dtype,
                                 device=self.device_)
        controls = None if controls is None else torch.as_tensor(
            controls, dtype=reference.dtype, device=self.device_)
        if direction == "backward" and controls is not None:
            controls = controls.flip(-2)
        step = self.sample_interval if direction == "forward" else -self.sample_interval
        with torch.no_grad():
            return self.dynamics_model_.rollout(states,
                                                num_steps,
                                                controls=controls,
                                                dt=step).cpu().numpy()

    def to(self, device):
        """Move every owned model and criterion through the common solver."""
        check_is_fitted(self, "n_features_")
        target = check_device(str(device))
        self.solver_.to(target)
        self.solver_.criterion.to(target)
        # Torch's optimizer loader casts moment tensors to their parameter
        # devices while retaining the appropriate device for scalar steps.
        optimizer = self.solver_.optimizer
        optimizer.load_state_dict(optimizer.state_dict())
        self.device = self.device_ = target
        self._fit_configuration["device"] = target
        return self

    def _get_state(self):
        self._check_fit_configuration()
        state = super()._get_state()
        sampling_state = self.sampling_state_
        if hasattr(self, "_active_window_generator"):
            sampling_state = self._active_window_generator.get_state()
        state.update(control_dimension_=self.control_dimension_,
                     sampling_state_=sampling_state)
        return state

    def _get_state_dict(self):
        checkpoint = super()._get_state_dict()
        checkpoint["metadata"].update(estimator="DYSCO", dysco_version=1)
        return checkpoint

    @classmethod
    def _from_checkpoint(cls, checkpoint, map_location=None):
        if checkpoint.get("metadata", {}).get("dysco_version") != 1:
            raise ValueError(
                "Unsupported DYSCO checkpoint version. Unversioned development "
                "checkpoints require the matching implementation revision.")
        args, state = dict(checkpoint["args"]), checkpoint["state"]
        if isinstance(map_location, (str, torch.device)):
            args["device"] = str(map_location)
        estimator = cls(**args)
        estimator._initialize(state["n_features_in_"],
                              state["control_dimension_"])
        for key, value in state.items():
            if key != "device_":
                setattr(estimator, key, value)
        estimator.solver_.load_state_dict(checkpoint["state_dict"])
        return estimator
