    * Chowdhury, Raeed H., Joshua I. Glaser, and Lee E. Miller. "Area 2 of primary somatosensory cortex encodes kinematics of the whole arm." Elife 9 (2020).
    * Chowdhury, Raeed; Miller, Lee (2022) Area2 Bump: macaque somatosensory area 2 spiking activity during reaching with perturbations (Version 0.220113.0359) [Data set]. `DANDI archive <https://doi.org/10.48324/dandi.000127/0.220113.0359>`_
    * Pei, Felix, et al. "Neural Latents Benchmark'21: Evaluating latent variable models of neural population activity." arXiv preprint arXiv:2109.04463 (2021).
import hashlib
import os

import numpy as np
import scipy.io

try:
    from nlb_tools.nwb_interface import NWBDataset
except ImportError:
    import warnings
    warnings.warn(
        ("Could not import the nlb_tools package required for data loading "
         "of cebra.datasets.monkey_reaching. Dataset will not be available. "
         "If required, you can install the dataset by running "
         "pip install git+https://github.com/neurallatents/nlb_tools."))
from cebra.datasets import get_datapath
from cebra.datasets import register

            os.path.join(self.path, f"{self.load_session}_{split}.jl"))



            """


