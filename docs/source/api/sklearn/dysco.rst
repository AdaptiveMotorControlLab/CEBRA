DYSCO: aligned-trial dynamics
=============================

DYSCO jointly trains a CEBRA encoder and a continuous-time polynomial vector
field. Training uses aligned repeated trials in a shared observation space,
with evenly sampled time and optional observed controls. Time alignment is
supplied by the caller: giving two recordings the same label does not align
them automatically.

Quick start
-----------

The following example generates its own observations:

.. code-block:: python

    import numpy as np
    import cebra

    rng = np.random.default_rng(4)
    time = np.arange(100) * 0.05
    signal = np.column_stack([np.cos(time), np.sin(time), np.cos(2 * time)])
    trials = [(signal + 0.02 * rng.normal(size=signal.shape)).astype("float32")
              for _ in range(4)]
    model = cebra.DYSCO(model_architecture="offset1-model-mse",
                        output_dimension=2, temperature=0.1, sample_interval=0.05,
                        rollout_steps=4, batch_size=16, max_iterations=100,
                        device="cpu")
    model.fit(trials, alignment_ids=[0, 0, 0, 0])
    embedding = model.transform(trials[0])
    derivative = model.vector_field(embedding)
    forecast = model.rollout(embedding[0], num_steps=10)
    print(*model.equations(precision=4, threshold=0.01), sep="\n")
    model.save("dysco.pt")
    restored = cebra.DYSCO.load("dysco.pt", map_location="cpu")

``trials`` contains arrays of shape ``(time, observed_feature)``. Repetitions
within each alignment group must have equal time support; different groups
can have different lengths. Set ``num_views=1`` for singleton groups. Controls,
when present, are supplied as a sequence of ``(time, control_feature)`` arrays
via ``fit(..., controls=controls)``. Field evaluation requires a control value
per state; rollout requires all control nodes including both endpoints.
Backward rollout starts at the final state, takes chronologically ordered
controls, and returns nodes in decreasing-time order.
``transform(trials)`` returns one embedding array per trial, using the same
encoder as ``transform(trials[0])``.

Views and the shared encoder
----------------------------

The aligned loader retains observations as ``(batch, view, raw_time, feature)``.
Each row chooses one alignment group and one start time, then draws distinct
repetitions from that group. Every view in a row has corresponding time nodes;
views are repetitions in one observation space, with one shared encoder.

For an encoder receptive width ``R`` and ``K`` rollout transitions, the raw
window has ``K + R`` samples. Unfolding creates ``K + 1`` receptive windows.
Only the encoder call flattens batch, view and core time into a single sample
axis: ``(B*V*(K+1), feature, R)``. Its output is immediately reshaped to
``(B, V, K+1, latent)``. The field and RK4 preserve the leading batch/view
axes; forward and backward predictions add a direction axis. Positive-event
sampling can then select corresponding times in different views explicitly.

The shared solver loop calls a three-argument criterion through ``Batch``.
The DYSCO inference adapter places encoded trajectories in ``reference``,
predicted trajectories in ``positive``, and candidate trajectories in
``negative``. DYSCOInfoNCE constructs actual positive/negative events from these
arrays. These slots therefore carry trajectory inputs, rather than already
selected positive/negative pairs. This adapter keeps optimizer updates,
callbacks, logging and validation in the existing solver. Generic losses and
batch-inspection code cannot assume the ordinary two-dimensional embeddings.

Training and sampling conventions
---------------------------------

The default bidirectional loss compares forward predictions with encoded
states, backward predictions with encoded states, and forward with backward
predictions. Every comparison sees the concatenation of encoded, forward and
backward negative candidates. CEBRA's existing squared Euclidean similarity
and stabilized full-denominator InfoNCE reduction are reused.

``temperature`` and ``negative_temperature`` are noise scales. Their squares
are multiplied by twice the physical reference horizon before dividing the
scores. The submitted convention uses the sampled reference index for both
encoded comparison blocks, including the backward block, and the complete
window horizon for forward/backward comparison. Index zero uses one time step.
This convention is explicit to preserve numerical compatibility with the
submitted objective. ``bidirectional=False`` selects the forward-only ablation.

``candidate_mode="in_batch"`` samples events from the encoded reference windows
and preserves the submitted row-offset sampling rule, including within-view
positives and independently sampled negative times. The negative pool is
shared across score rows. Duplicate physical windows and positive/negative
collisions are retained; there is no implicit collision mask.
``batch_size_negatives`` controls the candidate event count independently of
``batch_size``. For integer multiples of the batch size, event draws match the
submitted row-offset rule. Other counts use uniform row draws, preserving the
candidate marginal without assigning unequal sampling weights to windows.
No separate negative windows are materialized in this mode.
``candidate_mode="independent"`` instead samples additional windows through
the aligned-trial loader; this is an explicitly different sampling policy.

Choose noise scales and horizons for the time and state units of the dataset.
The default ``offset1-model-mse`` encoder allows unconstrained Euclidean latent
coordinates. A normalized encoder constrains coordinates to a unit sphere and
changes the representational assumptions of the polynomial state model.

The default KDE potential corrects candidate scores in the first two blocks.
Its encoded support is detached, while query and rollout gradients remain
active. ``potential="mlp"`` selects a learned centered potential; ``None``
disables the correction. Positive and negative queries share one potential
evaluation and centering constant. This corrects the separately centered MLP
queries in the submitted implementation; the frozen uncentered-potential and
KDE parity tests remain unchanged. The criterion owns potential parameters.

Lifecycle and scope
-------------------

The estimator reuses CEBRA's encoder registry, model configuration, solver
training loop, callbacks, embedding transform and checkpoint envelope. Its
solver inherits reference-model persistence: the encoder and dynamics are
registered in ``reference_model``. Potential parameters and the private event
sampler state belong to ``criterion``. Checkpoints also preserve optimizer
state, window-sampler state and constructor configuration. ``partial_fit``
continues from these states, while ``fit`` initializes a new model. The default
encoder's subsequent updates are tested to match after save/load. Stochastic
encoder layers can also depend on the external Torch random state.
Snapshots taken inside training callbacks read the live window sampler.
``to(device)`` also migrates Adam moments before continued training.
Changing construction or objective parameters with ``set_params`` requires
``fit``; ``partial_fit`` rejects such changes. Iteration count, verbosity and
transform padding may change during continuation.
Checkpoints identify the DYSCO objective/format version. Unversioned local
development checkpoints require their matching code revision; loading them
into this corrected objective is rejected explicitly.

The vector field uses directly trainable polynomial coefficients. Numerical
loss and gradient tests use frozen fixtures from the submitted research code;
these tests do not reproduce the paper's recovery experiments. Direct
coefficient optimization is an intentional parameterization choice. Inspect
the coefficient tensor through ``model.dynamics_model_.field.coefficients`` or
print equations with ``print(*model.equations(), sep="\n")``. Equations use the
learned latent coordinates and the time units of ``sample_interval``. Custom
``variable_names`` and ``control_names`` label coordinates without transforming
them. ``threshold`` and ``precision`` affect display only; they do not fit a
sparse field or recover the physical coordinate system.
``dynamics_learning_rate`` optionally gives polynomial coefficients their own
Adam rate; encoder and potential use ``learning_rate``. This uses ordinary
optimizer parameter groups, preserving direct coefficient optimization.
The default ``None`` gives all parameters the same rate.

This first interface supports polynomial fields, observed controls, aligned
trials and one embedding per observation node. Irregular time, heterogeneous
observation spaces, resampling encoders, latent inferred controls, symbolic
library selection and affine-gauge sparsification are outside its scope.

An offline autonomous/controlled oscillator example with held-out diagnostics
is available in ``tests/examples/dysco_sklearn_example.py``. It reports
embedding and transported-field R-squared, short-rollout error and a persistence
comparison; these are bounded diagnostics rather than recovery guarantees.

The Lorenz example ``tests/examples/dysco_lorenz_example.py`` generates repeated
noisy observations through a shared linear mixing map, fits a quadratic field,
prints its equations and saves a checkpoint, diagnostics and a figure. Run it
with ``--output lorenz-example``. Ground-truth states are used only for a
training-set affine evaluation map and held-out diagnostics. Repeated-trial and
unseen-initial-condition results are reported separately, including field
agreement and forecasts against persistence. This is a demonstration with
linear observations and deterministic Lorenz trajectories.

The ordinary CEBRA ``infonce_loss`` and goodness-of-fit helpers reject this
aligned-trial objective: their data loaders and information baseline do not
apply. Raw DYSCO training loss is available through ``model.solver_.history``;
scientific validation should assess held-out embeddings, fields and rollouts.

API reference
-------------

.. autoclass:: cebra.DYSCO
   :members:
   :show-inheritance:

.. automodule:: cebra.models.dysco
   :members:
   :show-inheritance:

.. automodule:: cebra.solver.dysco
   :members:
   :show-inheritance:
