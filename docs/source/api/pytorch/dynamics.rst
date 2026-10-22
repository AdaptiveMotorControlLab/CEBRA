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

Checkpoint behavior
~~~~~~~~~~~~~~~~~~~

Both minibatch and full-batch DCL solver checkpoints include the dynamics
parameters. Historical DCL checkpoints that omitted those parameters cannot
restore the trained dynamics. Strict loading raises an explanatory error.
Low-level ``solver.load_state_dict(checkpoint, strict=False)`` loads the
available components and warns that the dynamics retain their initialization.
Ordinary CEBRA checkpoints have no dynamics requirement.
