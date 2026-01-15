<div align="center">


<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/6f3943a1-b76d-4674-9df9-87aebd33e517/cebralogo.png?format=2500w" width="95%">
</p>



[📚Documentation](https://cebra.ai/docs/) |
[💡DEMOS](https://cebra.ai/docs/demos.html) |
[🛠️ Installation](https://cebra.ai/docs/installation.html) |
[🌎 Home Page](https://www.cebra.ai) |
[🚨 News](https://cebra.ai/docs/index.html) |
[🪲 Reporting Issues](https://github.com/AdaptiveMotorControlLab/CEBRA)


[![Downloads](https://static.pepy.tech/badge/cebra)](https://pepy.tech/project/cebra)
[![Downloads](https://static.pepy.tech/badge/cebra/month)](https://pepy.tech/project/cebra)
[![PyPI version](https://badge.fury.io/py/cebra.svg)](https://badge.fury.io/py/cebra)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-red)
[![Twitter Follow](https://img.shields.io/twitter/follow/CEBRAAI.svg?label=CEBRAai&style=social)](https://twitter.com/CEBRAAI)



</div>

# Welcome! 👋

**CEBRA** is a library for estimating **C**onsistent **E**m**B**eddings of high-dimensional **R**ecordings using **A**uxiliary variables. It contains self-supervised learning algorithms implemented in PyTorch, and has support for a variety of different datasets common in biology and neuroscience.

To receive updates on code releases, please 👀 watch or ⭐️ star this repository!

``cebra`` is a patented self-supervised method for non-linear clustering that allows for label-informed time series analysis.
It can jointly use behavioral and neural data in a hypothesis- or discovery-driven manner to produce consistent, high-performance latent spaces. While it is not specific to neural and behavioral data, this is the first domain we used the tool in. This application case is to obtain a consistent representation of latent variables driving activity and behavior, improving decoding accuracy of behavioral variables over standard supervised learning, and obtaining embeddings which are robust to domain shifts.


# References

- 📄 **Publication April 2025**:
  [Time-series attribution maps with regularized contrastive learning.](https://arxiv.org/abs/2502.12977)
Steffen Schneider, Rodrigo González Laiz, Anastasiia Filipova, Markus Frey, Mackenzie Weygandt Mathis. AISTATS 2025.


- 📄 **Publication May 2023**:
  [Learnable latent embeddings for joint behavioural and neural analysis.](https://doi.org/10.1038/s41586-023-06031-6)
  Steffen Schneider*, Jin Hwa Lee* and Mackenzie Weygandt Mathis. Nature 2023.

- 📄 **Preprint April 2022**:
  [Learnable latent embeddings for joint behavioral and neural analysis.](https://arxiv.org/abs/2204.00673)
  Steffen Schneider*, Jin Hwa Lee* and Mackenzie Weygandt Mathis

# Patent Information 

- [Dimensionality reduction of time-series data, and systems and devices that use the resultant embeddings](https://patents.google.com/patent/US12499131B2/en).  Steffen Schneider* & Mackenzie Weygandt Mathis*. Awarded Dec 2025. Please contact the [TTO office](adam.swetloff@epfl.ch) at EPFL for licensing.  

# License

- Since version 0.4.0, CEBRA is open source software under an Apache 2.0 license.
- Prior versions 0.1.0 to 0.3.1 were released for academic use only (please read the license file).
