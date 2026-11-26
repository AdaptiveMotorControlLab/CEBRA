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
"""Offline aligned-trial example and bounded DYSCO diagnostics.

Run from the CEBRA checkout with ``python tests/examples/dysco_sklearn_example.py``.
The example generates its own observations and optional controls.
"""

import argparse
import json
import time

import numpy as np

import cebra


def make_trials(controlled=False, seed=17):
    """Generate repeated noisy observations of two linear oscillator trials."""
    rng = np.random.default_rng(seed)
    dt = 0.05
    times = np.arange(100) * dt
    mixing, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    trials, states, controls, groups, derivatives = [], [], [], [], []
    for group, initial in enumerate(([1., 0.], [0.2, 0.8])):
        u = (0.3 *
             np.sin(0.7 * times + group))[:, None] if controlled else np.zeros(
                 (len(times), 1))

        def field(z, value):
            return np.array([-z[1] + value, z[0]])

        z = np.asarray(initial, dtype=float)
        trajectory = [z.copy()]
        for k in range(len(times) - 1):
            start, end = u[k, 0], u[k + 1, 0]
            middle = (start + end) / 2
            k1 = field(z, start)
            k2 = field(z + dt * k1 / 2, middle)
            k3 = field(z + dt * k2 / 2, middle)
            k4 = field(z + dt * k3, end)
            z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            trajectory.append(z.copy())
        trajectory = np.asarray(trajectory)
        derivative = np.stack(
            [field(z, value) for z, value in zip(trajectory, u[:, 0])])
        for repetition in range(6):
            trials.append(
                (trajectory @ mixing.T +
                 rng.normal(scale=0.02, size=(100, 5))).astype("float32"))
            states.append(trajectory)
            controls.append(u.astype("float32"))
            groups.append(group)
            derivatives.append(derivative)
    return trials, states, controls if controlled else None, groups, derivatives, dt


def run(seed=42, iterations=500, controlled=False):
    """Fit one seed and report held-out observation and dynamics diagnostics."""
    trials, states, controls, groups, derivatives, dt = make_trials(controlled)
    training = [i for i in range(len(trials)) if i % 6 < 4]
    testing = [i for i in range(len(trials)) if i % 6 >= 4]
    model = cebra.DYSCO(model_architecture="offset1-model-mse",
                        output_dimension=2,
                        num_hidden_units=16,
                        sample_interval=dt,
                        polynomial_degree=1,
                        rollout_steps=8,
                        integration_substeps=1,
                        num_views=2,
                        batch_size=16,
                        batch_size_negatives=32,
                        max_iterations=iterations,
                        learning_rate=1e-3,
                        temperature=0.1,
                        potential="kde",
                        random_state=seed,
                        device="cpu")
    start = time.perf_counter()
    model.fit(
        [trials[i] for i in training],
        alignment_ids=[groups[i] for i in training],
        controls=None if controls is None else [controls[i] for i in training])
    elapsed = time.perf_counter() - start
    encoded_train = np.concatenate(
        [model.transform(trials[i]) for i in training])
    target_train = np.concatenate([states[i] for i in training])
    mapping = np.linalg.lstsq(np.c_[encoded_train,
                                    np.ones(len(encoded_train))],
                              target_train,
                              rcond=None)[0]
    predictions, targets, field_predictions, field_targets = [], [], [], []
    errors, persistence = [], []
    for i in testing:
        encoded = model.transform(trials[i])
        predictions.append(np.c_[encoded, np.ones(len(encoded))] @ mapping)
        targets.append(states[i])
        u = None if controls is None else controls[i]
        field_predictions.append(model.vector_field(encoded, u) @ mapping[:-1])
        field_targets.append(derivatives[i])
        for k in range(0, len(encoded) - 4, 4):
            segment = None if u is None else u[k:k + 5]
            trajectory = model.rollout(encoded[k], 4, segment)
            errors.append(np.square(trajectory[-1] - encoded[k + 4]).mean())
            persistence.append(np.square(encoded[k] - encoded[k + 4]).mean())

    def score(predicted, actual):
        predicted, actual = np.concatenate(predicted), np.concatenate(actual)
        return float(1 - np.square(predicted - actual).sum() /
                     np.square(actual - actual.mean(0)).sum())

    return dict(seed=seed,
                controlled=controlled,
                iterations=iterations,
                seconds=elapsed,
                last_loss=model.solver_.history[-1],
                embedding_r2=score(predictions, targets),
                transported_field_r2=score(field_predictions, field_targets),
                rollout_mse=float(np.mean(errors)),
                persistence_mse=float(np.mean(persistence)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controlled", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.iterations, args.controlled),
                     indent=2))
