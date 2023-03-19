Demo Notebooks
==============

We provide a set of demo notebooks to get started with using CEBRA. To
run the notebooks, you need a working Jupyter notebook server, a CEBRA
installation, and the datasets required to run the notebooks, available on 
`FigShare <https://figshare.com/s/60adb075234c2cc51fa3>`_.


.. nbgallery::
   :maxdepth: 2

   Encoding of space, hippocampus (CA1) <demo_notebooks/Demo_hippocampus.ipynb>
   Decoding movie features from (V1) visual cortex <demo_notebooks/Demo_Allen.ipynb>
   Forelimb dynamics, somatosensory (S1) <demo_notebooks/Demo_primate_reaching.ipynb>
   Synthetic neural benchmarking <demo_notebooks/Demo_synthetic_exp.ipynb>
   Hypothesis-driven analysis <demo_notebooks/Demo_hypothesis_testing.ipynb>
   Consistency <demo_notebooks/Demo_consistency.ipynb>
   Decoding <demo_notebooks/Demo_decoding.ipynb>
   Topological data analysis <demo_notebooks/Demo_cohomology.ipynb>
   Technical: Training models across animals <demo_notebooks/Demo_hippocampus_multisession.ipynb>
   Technical: conv-piVAE <demo_notebooks/Demo_conv-pivae.ipynb>
   Technical: S1 training with MSE loss <demo_notebooks/Demo_primate_reaching_mse_loss.ipynb>
   Technical: Learning the temperature parameter <demo_notebooks/Demo_learnable_temperature.ipynb>
   

The demo notebooks can also be found in the ``demo_notebooks/`` subdirectory
in the CEBRA repository. 

Installation
------------

Before you can run these notebooks, you must have a working installation of CEBRA.
Please see the dedicated :doc:`Installation Guide </installation>` for information on installation options using ``conda``, ``pip`` and ``docker``.

Synthetic Experiment Demo (CEBRA, piVAE, tSNE, UMAP):
This demo requires several additional packages that have differing
requirements to CEBRA. Therefore, we recommend using the supplied
``docker`` container or ``conda`` cebra-full env.

Download Demo Data From FigureShare
-----------------------------------

We host prepackaged data on
`figshare <https://figshare.com/s/60adb075234c2cc51fa3>`__. Please
download them and check the loading directory is correct in the
notebook. By default we assume you have downloaded the data in the
``./data/`` directory in the repository root.

For different paths, you can specify the ``CEBRA_DATADIR=...``
environment variable. You can do this by placing
``import os; os.environ['CEBRA_DATADIR'] = "path/to/your/data"`` at the
**top** of your notebook.

For reference, the original data is available at:

- `Hippocampus dataset <https://crcns.org/data-sets/hc/hc-11/about-hc-11>`_, using a 
  `preprocessing script <https://github.com/zhd96/pi-vae/blob/main/code/rat_preprocess_data.py>`_.
- `Primate S1 dataset <https://gui.dandiarchive.org/#/dandiset/000127>`_.
- Allen Institute `Neuropixels dataset <https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html>`_ and `2P dataset  <https://allensdk.readthedocs.io/en/latest/>`_.


