.. toctree::
   :hidden:

   Home <self>
   Installation <installation>
   Usage <usage>
   Demos <demos>
   Contributing <contributing>
   Figures <figures>
   API Docs <api>

.. image:: _static/img/logo_large.png
   :align: center
   :width: 480px

.. rst-class:: heading-center

Welcome to CEBRA's documentation!
=================================

CEBRA is a library for estimating Consistent Embeddings of high-dimensional Recordings using Auxiliary variables.
It contains self-supervised learning algorithms implemented in PyTorch, and has support for
a variety of different datasets common in biology and neuroscience.

Please support the development of CEBRA by starring and/or watching the project on Github_!

.. note::

   CEBRA is under active development and the API might include breaking changes
   between versions. If you use CEBRA in your work, we recommend to double check
   your current version. For writing reproducible analysis and experiment code, we recommend
   to use Docker.

Installation and Setup
----------------------

Please see the dedicated :doc:`Installation Guide </installation>` for information on installation options using ``conda``, ``pip`` and ``docker``.


Usage 
-----


Integrations
------------

CEBRA can be directly integrated with existing libraries commonly used in data analysis. The ``cebra.integrations`` module
possibility to compute CEBRA embeddings on DeepLabCut_ outputs directly.


Licensing
---------

© All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE, Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023. 


Contributing
------------

Please refer to the :doc:`Contributing </contributing>` tab to find our guidelines on contributions.
Code contributors
-----------------

The CEBRA code was originally developed by Steffen Schneider, Jin H. Lee, and Mackenzie Mathis (up to internal version 0.0.2). As of March 2023, it is being actively extended and maintained by `Steffen Schneider`_, `Célia Benquet`_, and `Mackenzie Mathis`_.

References
----------
.. code::

  @article{schneider2022cebra,
    author = {Schneider, Steffen and Lee, Jin H and Mathis, Mackenzie W},
    title = {Learnable latent embeddings for joint behavioral and neural analysis},
    journal = {CoRR},
    volume = {abs/2204.00673},
    doi = {10.48550/ARXIV.2204.00673},
    url = {https://arxiv.org/abs/2204.00673},
    year = {2022},
  }

This documentation is based on the `PyData Theme`_.


.. _`Twitter`: https://twitter.com/cebraAI
.. _`PyData Theme`: https://github.com/pydata/pydata-sphinx-theme
.. _`DeepLabCut`: https://deeplabcut.org
.. _`Github`: https://github.com/AdaptiveMotorControlLab/cebra
.. _`email`: mailto:mackenzie.mathis@epfl.ch
.. _`Steffen Schneider`: https://github.com/stes
.. _`Célia Benquet`: https://github.com/CeliaBenquet
.. _`Mackenzie Mathis`: https://github.com/MMathisLab
