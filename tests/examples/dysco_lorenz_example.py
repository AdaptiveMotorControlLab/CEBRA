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
"""Learn polynomial dynamics from noisy, repeated Lorenz observations.

Run ``python tests/examples/dysco_lorenz_example.py --output lorenz-example``.
The script generates all data locally and writes equations, a checkpoint,
held-out diagnostics and a figure. True states are used only for evaluation.
This small example uses linear observation mixing and deterministic dynamics;
it does not reproduce the paper's noisy nonlinear observation experiments.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import cebra


def lorenz(states):
    """Evaluate the standard Lorenz field in physical state and time units."""
    x, y, z = np.moveaxis(np.asarray(states), -1, 0)
    return np.stack((10 * (y - x), x * (28 - z) - y, x * y - 8 * z / 3),
                    axis=-1)


def simulate(initial, length=1200, dt=0.01, burn_in=300):
    """Integrate the field with two RK4 substeps per recorded interval."""
    state = np.asarray(initial, dtype=float)
    trajectory = []
    for index in range(length + burn_in):
        if index >= burn_in:
            trajectory.append(state.copy())
        for _ in range(2):
            h = dt / 2
            k1 = lorenz(state)
            k2 = lorenz(state + h * k1 / 2)
            k3 = lorenz(state + h * k2 / 2)
            k4 = lorenz(state + h * k3)
            state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return np.asarray(trajectory)


def make_trials(seed=17):
    """Create four training conditions and one unseen initial condition.

    Each training condition has four aligned noisy repetitions, three for
    training and one for evaluation. The same observation map is used for all
    conditions. No windows from the fifth condition enter training.
    """
    rng = np.random.default_rng(seed)
    mixing, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    trials, states, groups, training, testing = [], [], [], [], []
    initials = ([1., 1., 1.], [-2., 3., 5.], [5., -3.,
                                              12.], [-7., -4.,
                                                     20.], [2., -1., 8.])
    for group, initial in enumerate(initials):
        trajectory = simulate(initial)
        scaled = (trajectory - [0., 0., 25.]) / 20
        for repetition in range(4 if group < 4 else 1):
            index = len(trials)
            trials.append((scaled @ mixing.T + rng.normal(
                scale=0.02, size=(len(scaled), 12))).astype("float32"))
            states.append(trajectory)
            groups.append(group)
            (training
             if group < 4 and repetition < 3 else testing).append(index)
    return trials, states, groups, training, testing


def r2(prediction, target):
    """Compute variance-weighted R-squared over all coordinates."""
    return float(1 - np.square(prediction - target).sum() /
                 np.square(target - target.mean(axis=0)).sum())


def diagnose(model, trials, states, training, testing):
    """Fit a training-only affine map, then score held-out states and fields."""
    encoded = [model.transform(trial) for trial in trials]
    train_embedding = np.concatenate([encoded[i] for i in training])
    train_state = np.concatenate([states[i] for i in training])
    affine = np.linalg.lstsq(np.c_[train_embedding,
                                   np.ones(len(train_embedding))],
                             train_state,
                             rcond=None)[0]
    metrics = {}
    for name, indices in (("held_out_repetitions", testing[:-1]),
                          ("unseen_initial_condition", testing[-1:])):
        embedding = np.concatenate([encoded[i] for i in indices])
        target = np.concatenate([states[i] for i in indices])
        predicted = np.c_[embedding, np.ones(len(embedding))] @ affine
        field = model.vector_field(embedding) @ affine[:-1]
        horizons = {}
        for horizon in (1, 5, 10, 20):
            initial, future = [], []
            for i in indices:
                starts = np.arange(0, len(encoded[i]) - horizon, 20)
                initial.append(encoded[i][starts])
                future.append(states[i][starts + horizon])
            initial, future = np.concatenate(initial), np.concatenate(future)
            flow = model.rollout(initial, horizon)[:, -1]
            forecast = np.c_[flow, np.ones(len(flow))] @ affine
            persistence = np.c_[initial, np.ones(len(initial))] @ affine
            error = float(np.square(forecast - future).mean())
            baseline = float(np.square(persistence - future).mean())
            horizons[str(horizon)] = dict(mse=error,
                                          persistence_mse=baseline,
                                          relative_mse=error / baseline)
        metrics[name] = dict(
            embedding_r2=r2(predicted, target),
            transported_field_r2=r2(field, lorenz(target)),
            latent_standard_deviation=embedding.std(0).tolist(),
            forecasts=horizons)
    return metrics, encoded, affine


def plot_example(model, states, encoded, affine, output):
    """Plot unseen trajectory geometry and a short autonomous forecast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = states[-1]
    embedding = encoded[-1]
    aligned = np.c_[embedding, np.ones(len(embedding))] @ affine
    figure = plt.figure(figsize=(12, 3.8), constrained_layout=True)
    for position, values, title in ((1, target, "Held-out Lorenz trajectory"),
                                    (2, aligned, "Affine-aligned embedding")):
        axis = figure.add_subplot(1, 3, position, projection="3d")
        axis.plot(*values.T, linewidth=0.6, color="tab:blue")
        axis.set(xlabel="x", ylabel="y", zlabel="z", title=title)
    axis = figure.add_subplot(1, 3, 3)
    start, horizon = 200, 50
    forecast = model.rollout(embedding[start], horizon)
    forecast = np.c_[forecast, np.ones(len(forecast))] @ affine
    times = np.arange(horizon + 1) * model.sample_interval
    for dimension, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
        axis.plot(times,
                  target[start:start + horizon + 1, dimension],
                  color=color,
                  label="True state" if dimension == 0 else None)
        axis.plot(times,
                  forecast[:, dimension],
                  "--",
                  color=color,
                  label="DYSCO rollout" if dimension == 0 else None)
    axis.set(xlabel="Time from initial state",
             ylabel="State",
             title="Autonomous forecast")
    axis.legend(frameon=False)
    figure.savefig(output / "lorenz.png", dpi=180)
    plt.close(figure)


def run(output, seed=42, iterations=3000):
    """Train, inspect equations, save and evaluate one reproducible fit."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trials, states, groups, training, testing = make_trials()
    model = cebra.DYSCO(model_architecture="offset1-model-mse",
                        output_dimension=3,
                        num_hidden_units=32,
                        polynomial_degree=2,
                        sample_interval=0.01,
                        rollout_steps=8,
                        num_views=2,
                        batch_size=32,
                        batch_size_negatives=128,
                        temperature=0.1,
                        potential=None,
                        learning_rate=3e-3,
                        dynamics_learning_rate=0.05,
                        max_iterations=iterations,
                        random_state=seed,
                        device="cpu")
    start = time.perf_counter()

    def report_progress(step, solver):
        print(f"Update {step + 1}: loss={solver.history[-1]:.4f}", flush=True)

    model.fit([trials[i] for i in training],
              alignment_ids=[groups[i] for i in training],
              callback=report_progress,
              callback_frequency=500)
    elapsed = time.perf_counter() - start
    equations = model.equations(precision=6)
    print(*equations, sep="\n", flush=True)
    model.save(output / "dysco.pt")
    (output / "equations.txt").write_text("\n".join(equations) + "\n")
    metrics, encoded, affine = diagnose(model, trials, states, training,
                                        testing)
    metrics.update(seed=seed,
                   iterations=iterations,
                   training_seconds=elapsed,
                   last_loss=model.solver_.history[-1],
                   affine_map_condition=float(np.linalg.cond(affine[:-1])))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(output / "diagnostics.npz",
             true_state=states[-1],
             embedding=encoded[-1],
             affine=affine,
             history=np.asarray(model.solver_.history))
    plot_example(model, states, encoded, affine, output)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("lorenz-example"))
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.seed, args.iterations)
