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

Please see the dedicated :doc:`Installation Guide </installation>` for information on installation options using ``conda``, ``pip`` and ``docker``. Have fun! 😁

Usage
-----

Please head over to the :doc:`Usage </usage>` tab to find step-by-step instructions to use CEBRA on your data. For example use cases, see the :doc:`Demos </demos>` tab.


Licensing
---------
The ideas presented in our package are currently patent pending (Patent No. WO2023143843).
Since version 0.4.0, CEBRA's source is licenced under an Apache 2.0 license.
Prior versions 0.1.0 to 0.3.1 were released for academic use only.

Please see the full license file on Github_ for further information.


Contributing
------------

Please refer to the :doc:`Contributing </contributing>` tab to find our guidelines on contributions.

Code Contributors
-----------------

The CEBRA code was originally developed by Steffen Schneider, Jin H. Lee, and Mackenzie Mathis (up to internal version 0.0.2). Please see our AUTHORS file for more information.

Integrations
------------

CEBRA can be directly integrated with existing libraries commonly used in data analysis. Namely, we provide a ``scikit-learn`` style interface to use CEBRA. Additionally, we offer integrations with our ``scikit-learn``-style of using CEBRA, a package making use of ``matplotlib`` and ``plotly`` to plot the CEBRA model results, as well as the possibility to compute CEBRA embeddings on DeepLabCut_ outputs directly. If you have another suggestion, please head over to Discussions_ on GitHub_!


Key References
--------------
.. code::

  @article{schneider2023cebra,
    author = {Schneider, Steffen and Lee, Jin H and Mathis, Mackenzie W},
    title = {Learnable latent embeddings for joint behavioural and neural analysis},
    journal = {Nature},
    doi = {https://doi.org/10.1038/s41586-023-06031-6},
    year = {2023},
  }

  @article{xCEBRA2025,
    author={Steffen Schneider and Rodrigo Gonz{\'a}lez Laiz and Anastasiia Filippova and Markus Frey and Mackenzie W Mathis},
    title = {Time-series attribution maps with regularized contrastive learning},
    journal = {AISTATS},
    url = {https://openreview.net/forum?id=aGrCXoTB4P},
    year = {2025},
  }

This documentation is based on the `PyData Theme`_.


.. _`Twitter`: https://twitter.com/cebraAI
.. _`PyData Theme`: https://github.com/pydata/pydata-sphinx-theme
.. _`DeepLabCut`: https://deeplabcut.org
.. _`Discussions`: https://github.com/AdaptiveMotorControlLab/CEBRA/discussions
.. _`Github`: https://github.com/AdaptiveMotorControlLab/cebra
.. _`email`: mailto:mackenzie.mathis@epfl.ch
.. _`Steffen Schneider`: https://github.com/stes
.. _`Mackenzie Mathis`: https://github.com/MMathisLab
