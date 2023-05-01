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
   :width: 480px

.. rst-class:: heading-center

Welcome to CEBRA's documentation!
=================================

a variety of different datasets common in biology and neuroscience.

Please support the development of CEBRA by starring and/or watching the project on Github_!

.. note::

   CEBRA is under active development and the API might include breaking changes
   your current version. For writing reproducible analysis and experiment code, we recommend
   to use Docker.

Installation and Setup
----------------------

Please see the dedicated :doc:`Installation Guide </installation>` for information on installation options using ``conda``, ``pip`` and ``docker``.


Integrations
------------

CEBRA can be directly integrated with existing libraries commonly used in data analysis. The ``cebra.integrations`` module
possibility to compute CEBRA embeddings on DeepLabCut_ outputs directly.


Licensing
---------
Contributing
------------

References
----------
.. code::

  @article{schneider2022cebra,
    author = {Schneider, Steffen and Lee, Jin H and Mathis, Mackenzie W},
    title = {Learnable latent embeddings for joint behavioral and neural analysis},
    journal = {CoRR},
    volume = {abs/2204.00673},
    year = {2022},
  }

This documentation is based on the `PyData Theme`_.

.. _`DeepLabCut`: https://deeplabcut.org
