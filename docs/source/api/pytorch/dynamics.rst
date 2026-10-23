Dynamics Models
---------------

Dynamics models used for Dynamics-aware Contrastive Learning (DCL). They are
applied to the reference samples during training and selected by name via
:py:attr:`cebra.CEBRA.dynamics_model_architecture`.

.. automodule:: cebra.dynamics
   :members:
   :show-inheritance:

Registration and initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: init

.. autofunction:: get_options

.. autofunction:: register

Models
~~~~~~

.. automodule:: cebra.dynamics.linear
   :members:
   :show-inheritance:

Polynomial vector fields and flow maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``polynomial-flow`` integrates a continuous-time polynomial vector field using
fixed-step RK4. Its forward method returns the state after one interval, so it
can be used by the existing DCL solver:

.. code-block:: python

    from cebra.integrations.sklearn.cebra import DCL

    model = DCL(dynamics_model_architecture="polynomial-flow",
                distance="euclidean", output_dimension=3)

The registered defaults are degree two, an interval of 0.02 and one RK4 step
per interval. The current DCL estimator constructs registry models with only
``latent_dim``. Configure other values through the low-level API:

.. code-block:: python

    import torch
    from cebra.dynamics.polynomial import PolynomialFlow

    flow = PolynomialFlow(latent_dim=3, degree=2, dt=0.01, substeps=4)
    z = torch.randn(8, 3)
    derivative = flow.field(z)
    next_state = flow(z)
    trajectory = flow.rollout(z, num_steps=20)  # (8, 21, 3)

``dt`` is the time interval represented by one map application. In DCL it must
be interpreted together with the sampler's temporal offset. Polynomial
coefficients have the corresponding time units. A stable numerical step does
not guarantee bounded long-term dynamics; check step refinement and rollout
behavior for the learned field.

For controlled fields, set ``control_dim`` and supply every control node to
``rollout``. Controls are linearly interpolated at intermediate RK4 stages.
The control array has shape ``(*batch, num_steps + 1, control_dim)``. Negative
``dt`` integrates backward; reverse the control-node order as well. DCL's
existing sampler does not pass control segments to the registered map.

This component supplies polynomial dynamics for DCL and a foundation for
aligned-trial DYSCO training. The multi-view DYSCO objective, candidate
potential and aligned-trial sampling are separate components under development.
Polynomial coefficients are directly trainable; this component does not
perform sparsification or select a symbolic library automatically.

Checkpoint behavior
~~~~~~~~~~~~~~~~~~~

Auxiliary solver checkpoints include ``reference_model``. For both minibatch
and full-batch DCL solvers, this contains the encoder and dynamics parameters.
Historical auxiliary solver checkpoints omitted the reference model and cannot
restore its unsaved parameters, including trained DCL dynamics. Strict loading
raises an explanatory error. Low-level
``solver.load_state_dict(checkpoint, strict=False)`` loads the available
components and warns that unsaved reference model parameters retain their
current values. Solvers without an auxiliary reference model are unchanged. For low-level custom
flows, reconstruct the same field/interval/substep configuration before loading
its state dictionary.

.. automodule:: cebra.dynamics.polynomial
   :members:
   :show-inheritance:
