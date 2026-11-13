=======
Solvers
=======

.. automodule:: cebra.solver
   :members:
   :show-inheritance:

.. automodule:: cebra.solver.base
   :members:
   :private-members:
   :show-inheritance:

Single session training
-----------------------

.. automodule:: cebra.solver.single_session
   :members:
   :show-inheritance:

Multi session training
----------------------

.. automodule:: cebra.solver.multi_session
   :members:
   :show-inheritance:

Unified session training
------------------------

:py:class:`~cebra.solver.unified_session.UnifiedSolver` trains one model shared by all sessions;
:py:class:`~cebra.solver.unified_session.UnifiedSolverxCEBRA` adds the Jacobian regularization of
xCEBRA (registered as ``"unified-xcebra"``).

.. automodule:: cebra.solver.unified_session
   :members:
   :show-inheritance:

Multi-objective training
------------------------

The multi-objective solvers (:py:mod:`cebra.solver.multiobjective`) are documented under
:doc:`Multi-objective models </api/pytorch/multiobjective>`.

Training utilities
------------------

.. automodule:: cebra.solver.util
   :members:
   :show-inheritance:
