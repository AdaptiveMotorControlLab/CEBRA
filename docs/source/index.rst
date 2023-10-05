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

CEBRA is a library for estimating Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables.
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

Have fun! 😁

Usage
-----

Please head over to the :doc:`Usage </usage>` tab to find step-by-step instructions to use CEBRA on your data. For example use cases, see the :doc:`Demos </demos>` tab.

Integrations
------------

CEBRA can be directly integrated with existing libraries commonly used in data analysis. The ``cebra.integrations`` module
is getting actively extended. Right now, we offer integrations for ``scikit-learn``-like usage of CEBRA, a package making use of ``matplotlib`` to plot the CEBRA model results, as well as the
possibility to compute CEBRA embeddings on DeepLabCut_ outputs directly.


Licensing
---------

© All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE, Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
It is made available for non-commercial research use only. It comes without any warranty or guarantee.
Please see the full license file on Github_, and if it is not suitable to your project, please email_ Mackenzie Mathis for a commercial license.


Contributing
------------

Please refer to the :doc:`Contributing </contributing>` tab to find our guidelines on contributions.

Code contributors
-----------------

The CEBRA code was originally developed by Steffen Schneider, Jin H. Lee, and Mackenzie Mathis (up to internal version 0.0.2). As of March 2023, it is being actively extended and maintained by `Steffen Schneider`_, `Célia Benquet`_, and `Mackenzie Mathis`_.

References
----------
.. code::

  @article{schneider2023cebra,
    author = {Schneider, Steffen and Lee, Jin H and Mathis, Mackenzie W},
    title = {Learnable latent embeddings for joint behavioural and neural analysis},
    journal = {Nature},
    doi = {https://doi.org/10.1038/s41586-023-06031-6},
    year = {2023},
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
