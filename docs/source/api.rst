========
API Docs
========

CEBRA has two main APIs:

- The high-level API is based on `Scikit-learn estimators`_. To apply CEBRA to custom datasets, this ``scikit-learn``-compatible interface should be used. We also present a simple way to use a decoder on the CEBRA output embeddings. 👉 Different use cases for this interface are outlined in the :doc:`Demo notebooks </demos>`.

- The low-level ``torch`` API exposes models, layers, loss functions and other components. The ``torch`` API exposes all low-level functions and classes used for training CEBRA models.

For **day-to-day use of CEBRA**, it is sufficient to know the high-level ``scikit-learn`` API, which
is currently limited to a single estimator class, :py:class:`cebra.CEBRA`. CEBRA's main
functionalities are covered by this class.

For machine learning researchers, and everybody with **custom data analysis needs**, we expose
all core functions of CEBRA via our ``torch`` API. This allows more fine-grained control over
the different components of the algorithm (models used for encoders, addition of custom
sampling mechanisms, variations of the base loss function, etc.). It also allows to use
these components in other contexts and research code bases.

.. toctree::
   :hidden:
   :caption: scikit-learn API

   api/sklearn/cebra
   api/sklearn/metrics
   api/sklearn/decoder
   api/sklearn/helpers


.. toctree::
   :hidden:
   :caption: PyTorch API

   api/pytorch/solvers
   api/pytorch/data
   api/pytorch/datasets
   api/pytorch/distributions
   api/pytorch/models
   api/pytorch/helpers

.. toctree::
   :hidden:
   :caption: Integrations

   api/integrations/data
   api/integrations/matplotlib
   api/integrations/plotly
   api/integrations/deeplabcut


.. _Scikit-learn estimators: https://scikit-learn.org/stable/developers/develop.html
