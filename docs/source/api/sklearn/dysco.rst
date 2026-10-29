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
``batch_size``. No separate negative windows are materialized in this mode.
``candidate_mode="independent"`` instead samples additional windows through
the aligned-trial loader; this is an explicitly different sampling policy.

Choose noise scales and horizons for the time and state units of the dataset.
An unnormalized encoder such as ``offset1-model-mse`` allows unconstrained
Euclidean latent coordinates; it is used by the oscillator example.

The default KDE potential corrects candidate scores in the first two blocks.
Its encoded support is detached, while query and rollout gradients remain
active. ``potential="mlp"`` selects a learned centered potential; ``None``
disables the correction. The criterion owns the potential and its parameters.

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

The vector field uses directly trainable polynomial coefficients. Numerical
loss and gradient tests use frozen fixtures from the submitted research code;
these tests do not establish optimization equivalence with its coefficient
network or reproduce the paper's recovery experiments. Polynomial coefficient
inspection is available through ``model.dynamics_model_.field.coefficients``.

This first interface supports polynomial fields, observed controls, aligned
trials and one embedding per observation node. Irregular time, heterogeneous
observation spaces, resampling encoders, latent inferred controls, symbolic
library selection and affine-gauge sparsification are outside its scope.

An offline autonomous/controlled oscillator example with held-out diagnostics
is available in ``tests/examples/dysco_sklearn_example.py``. It reports
embedding and transported-field R-squared, short-rollout error and a persistence
comparison; these are bounded diagnostics rather than recovery guarantees.

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
