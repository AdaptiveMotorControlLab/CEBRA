============
Data Loading
============

.. automodule:: cebra.data
   :show-inheritance:


Base classes
----------------

.. automodule:: cebra.data.base
   :members:
   :show-inheritance:


General File Loading
--------------------------------

.. autofunction:: cebra.load_data


DeepLabCut File Loading
-----------------------

.. autofunction:: cebra.load_deeplabcut


Pre-defined Datasets
--------------------------------

.. automodule:: cebra.data.datasets
   :members:
   :show-inheritance:


Unified Datasets and Dataloaders
--------------------------------

:py:class:`cebra.data.datasets.UnifiedDataset` and :py:class:`cebra.data.datasets.DatasetxCEBRA`
are documented under *Pre-defined Datasets* above, and :py:class:`cebra.data.multi_session.UnifiedLoader`
under *Multi Session Dataloaders*.


Masking
--------------------------------

The masks (:py:mod:`cebra.data.masking`) are documented under :doc:`Helper functions </api/pytorch/helpers>`.


Single Session Dataloaders
--------------------------------

.. automodule:: cebra.data.single_session
   :members:
   :show-inheritance:


Multi Session Dataloaders
--------------------------------

.. automodule:: cebra.data.multi_session
   :members:
   :show-inheritance:


Datatypes
----------------

.. automodule:: cebra.data.datatypes
   :members:
   :show-inheritance:


Aligned repeated trials
-----------------------

``AlignedTrialDataset`` represents repeated trials in a shared observation
feature space. The caller supplies alignment groups and a sample interval;
trials within each group must already be temporally aligned and have equal
lengths. Groups may have different lengths. This is separate from heterogeneous
multi-session data.

.. code-block:: python

    import numpy as np
    from cebra.data import AlignedTrialDataset, AlignedTrialLoader, Offset

    trials = [np.zeros((100, 5), dtype="float32") for _ in range(4)]
    dataset = AlignedTrialDataset(trials, alignment_ids=[0, 0, 1, 1],
                                 sample_interval=0.02)
    dataset.offset = Offset(2, 3)  # Or dataset.configure_for(encoder).
    loader = AlignedTrialLoader(dataset, num_steps=10, batch_size=16,
                               batch_size_negatives=32, rollout_steps=8,
                               num_views=2, seed=42)
    batch = next(iter(loader))
    assert batch.observations.shape == (16, 2, 13, 5)
    assert batch.time.shape == (16, 2, 9)
    assert batch.negative.shape == (32, 13, 5)

A core window has ``rollout_steps + 1`` nodes. Raw observations include the
encoder's receptive context; controls and times cover the core nodes only.
Views are distinct trials from the same group at the same indices. Set
``num_views=1`` for singleton groups. Impossible view counts and windows are
rejected; no trial boundary is padded or crossed implicitly.

Reference groups are sampled uniformly, then valid starts and distinct trials
are selected within the group. Negative windows are sampled independently from
the same group-uniform marginal; ``batch_size_negatives`` counts windows. The
returned index metadata retains trial identities and starts, including any
candidate collisions, for the contrastive objective's exclusion policy. This
loader does not define the DYSCO loss or claim submitted-objective parity.

The private CPU random generator is independent of global Torch seeding and
advances across iterations. Its state can be captured with
``loader.generator.get_state()`` and restored with ``set_state``. Observation,
control, time and index tensors follow device transfers.

.. automodule:: cebra.data.aligned_trials
   :members:
   :show-inheritance:
