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

Auxiliary solver checkpoints include ``reference_model``. For both minibatch
and full-batch DCL solvers, this contains the encoder and dynamics parameters.
Historical auxiliary solver checkpoints omitted the reference model and cannot
restore its unsaved parameters, including trained DCL dynamics. Strict loading
raises an explanatory error. Low-level
``solver.load_state_dict(checkpoint, strict=False)`` loads the available
components and warns that unsaved reference model parameters retain their
current values. Solvers without an auxiliary reference model are unchanged.
