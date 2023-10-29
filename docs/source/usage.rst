Using CEBRA
===========

This page covers a standard CEBRA usage. We recommend checking out the :py:doc:`demos` for in-depth CEBRA usage examples as well. Here we present a quick overview on how to use CEBRA on various datasets. Note that we provide two ways to interact with the code:

* For regular usage, we recommend leveraging the **high-level interface**, adhering to ``scikit-learn`` formatting.
* Upon specific needs, advanced users might consider diving into the **low-level interface** that adheres to ``PyTorch`` formatting.


Firstly, why use CEBRA?
-----------------------

CEBRA is primarily designed for producing robust, consistent extractions of latent factors from time-series data. It supports three modes, and is a self-supervised representation learning algorithm that uses our modified contrastive learning approach designed for multi-modal time-series data. In short, it is a type of non-linear dimensionality reduction, like `tSNE <https://www.jmlr.org/papers/v9/vandermaaten08a.html>`_ and `UMAP <https://arxiv.org/abs/1802.03426>`_. We show in our original paper that it outperforms tSNE and UMAP at producing closer-to-ground-truth latents and is more consistent.

That being said, CEBRA can be used on non-time-series data and it does not strictly require multi-modal data. In general, we recommend considering using CEBRA for measuring changes in consistency across conditions (brain areas, cells, animals), for hypothesis-guided decoding, and for toplogical exploration of the resulting embedding spaces. It can also be used for visualization and considering dynamics within the embedding space. For examples of how CEBRA can be used to map space, decode natural movies, and make hypotheses for neural coding of sensorimotor systems, see our paper (Schneider, Lee, Mathis, 2023).

The CEBRA workflow
------------------

CEBRA supports three modes: fully unsupervised (CEBRA-Time), supervised (via joint modeling of auxiliary variables; CEBRA-Behavior), and a hybrid variant (CEBRA-Hybrid).
We recommend to start with running CEBRA-Time (unsupervised) and look both at the loss value (goodness-of-fit) and visualize the embedding. Then use labels via CEBRA-Behavior to test which labels give you this goodness-of-fit (tip: see our :py:doc:`cebra-figures/figures/Figure2` hippocampus data example). Notably, if you use CEBRA-Behavior with labels that are not encoded in your data, the embedding will collapse (not-converged). This is a feature, not a bug. This allows you to rule out which obervable behaviors (labels) are truly in your data. To get a sense of this workflow, you can also look at :py:doc:`cebra-figures/figures/Figure2` and :py:doc:`cebra-figures/figures/ExtendedDataFigure5`.  👉 Here is a quick-start workflow, with many more details below:

(1) Use CEBRA-Time for unsupervised data exploration.
(2) Consider running a hyperparameter sweep on the inputs to the model, such as :py:attr:`cebra.CEBRA.model_architecture`, :py:attr:`cebra.CEBRA.time_offsets`, :py:attr:`cebra.CEBRA.output_dimension`, and set :py:attr:`cebra.CEBRA.batch_size` to be as high as your GPU allows. You want to see clear structure in the 3D plot (the first 3 latents are shown by default).
(3) Use CEBRA-Behavior with many different labels and combinations, then look at the InfoNCE loss - the lower the loss value, the better the fit (see :py:doc:`cebra-figures/figures/ExtendedDataFigure5`), and visualize the embeddings. The goal is to understand which labels are contributing to the structure you see in CEBRA-Time, and improve this structure. Again, you should consider a hyperparameter sweep.
(4) Interpretability: now you can use these latents in downstream tasks, such as measuring consistency, decoding, and determining the dimensionality of your data with topological data analysis.

All the steps to do this are described below. Enjoy using CEBRA! 🔥🦓


Step-by-step CEBRA
------------------

For a quick start into applying CEBRA to your own datasets, we provide a `scikit-learn compatible <https://scikit-learn.org/stable/glossary.html>`_ API, similar to methods such as tSNE, UMAP, etc.
We assume you have CEBRA installed in the environment you are working in, if not go to the :doc:`installation`.
Next, launch your conda env (e.g., ``conda activate cebra``).


Create a CEBRA workspace
^^^^^^^^^^^^^^^^^^^^^^^^

Assuming you have your data recorded, you want to start using CEBRA on it.
For instance you can create a new jupyter notebook.

For the sake of this usage guide, we create some example data:

.. testcode::

    # Create a .npz file
    import numpy as np

    X = np.random.normal(0,1,(100,3))
    X_new = np.random.normal(0,1,(100,4))
    np.savez("neural_data", neural = X, new_neural = X_new)

    # Create a .h5 file, containing a pd.DataFrame
    import pandas as pd

    X_continuous = np.random.normal(0,1,(100,3))
    X_discrete = np.random.randint(0,10,(100, ))
    df = pd.DataFrame(np.array(X_continuous), columns=["continuous1", "continuous2", "continuous3"])
    df["discrete"] = X_discrete
    df.to_hdf("auxiliary_behavior_data.h5", key="auxiliary_variables")


You can start by importing the CEBRA package, as well as the CEBRA model as a classical ``scikit-learn`` estimator.

.. testcode::

    import cebra
    from cebra import CEBRA

Data loading
^^^^^^^^^^^^

Get the data ready
""""""""""""""""""


We acknowledge that your data can come in all formats.
That is why we developed a loading helper function to help you get your data ready to be used by CEBRA.


The function :py:func:`cebra.load_data` supports various file formats to convert the data of interest to a :py:func:`numpy.array`.
It handles three categories of data. Note that it will only read the data of interest and output the corresponding :py:func:`numpy.array`.
It does not perform pre-processing so your data should be ready to be used for CEBRA.

* Your data is a **2D array**. In that case, we handle Numpy, HDF5, PyTorch, csv, Excel, Joblib, Pickle and MAT-files. If your file only containsyour data then you can use the default :py:func:`cebra.load_data`. If your file contains more than one dataset, you will have to provide a ``key``, which corresponds to the data of interest in the file.


* Your data is a :py:class:`pandas.DataFrame`. In that case, we handle HDF5 files only. Similarly, you can use the default :py:func:`cebra.load_data` if your file only contains a single dataset and you want to get the whole :py:class:`pandas.DataFrame` as your dataset. Else, if your file contains more than one dataset, you will have to provide the corresponding ``key``. Moreover, *if your* :py:class:`pandas.DataFrame` *is a single index*, you can precise the ``columns`` to fetch from the :py:class:`pandas.DataFrame` for your data of interest.

In the following example, ``neural_data.npz`` contains multiple :py:func:`numpy.array` and ``auxiliary_behavior_data.h5``, multiple :py:class:`pandas.DataFrame`.

.. testcode::

    import cebra

    # Load the .npz
    neural_data = cebra.load_data(file="neural_data.npz", key="neural")

    # ... and similarly load the .h5 file, providing the columns to keep
    continuous_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
    discrete_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["discrete"]).flatten()

You can then use ``neural_data``, ``continuous_label`` or ``discrete_label`` directly as the input or index data of your CEBRA model. Note that we flattened ``discrete_label``
in order to get a 1D :py:func:`numpy.array` as required for discrete index inputs.


.. note::
    :py:func:`cebra.load_data` only handles **one set of data at a time**, either the data or the labels, for one session only. To use multiple sessions and/or multiple labels, the function can be called for each of dataset. For files containing multiple matrices, the corresponding ``key``, referenciating the dataset in the file, must be provided.


.. admonition:: See API docs: :py:func:`cebra.load_data`
    :class: dropdown

    .. autofunction:: cebra.load_data
        :noindex:

.. admonition:: See API docs: :py:func:`cebra.load_deeplabcut`
    :class: dropdown

    .. autofunction:: cebra.load_deeplabcut
        :noindex:


.. _auxiliary variables:

Choose the CEBRA mode and related auxiliary variables
""""""""""""""""""""""""""""""""""""""""""""""""""""""

CEBRA allows you to jointly use time-series data and (optionally) auxiliary variables to extract latent spaces. If you want to use time-only (namely, unsupervised) select:

* **CEBRA-Time:** Discovery-driven: time contrastive learning. Set ``conditional='time'``. No assumption on the  behaviors that are influencing neural activity. It can be used as a first step into the data analysis for instance, or as a comparison point to multiple hypothesis-driven analyses.

To use auxiliary (behavioral) variables you can choose both continuous and discrete variables. The label information (none, discrete, continuous) determine the algorithm to use for data sampling. Using labels allows you to project future behavior onto past time-series activity, and explicitly use label-prior to shape the embedding. The conditional distribution can be chosen upon model initialization with the :py:attr:`cebra.CEBRA.conditional` parameter.

* **CEBRA-Behavior:** Hypothesis-driven: behavioral contrastive learning. Set ``conditional='time_delta'``. The user makes an hypothesis on the variables influencing neural activity (behavioral features such as position or head orientation, trial number, brain region, etc.). If the chosen auxiliary variables are in fact influencing the data to reduce, the resulting embedding should reflect that. Hence, it can easily be used to *compare hypotheses*. Auxiliary variables can be multiple, and both continuous and discrete. 👉 Examples on how to select them are presented in :py:doc:`demo_notebooks/Demo_primate_reaching`.

    * *Discrete auxiliary variables.* A 1D matrix, containing :py:class:`int`. *Example: trial ID, rewards, brain region ID.*

        .. note::
            There can be only one discrete set of index per model.

    * *Continuous auxiliary variables.* A 2D matrix, containing :py:class:`float`. Multiple continuous index can be chosen for the same model. *Example: kinematics, actions.*

* **CEBRA-Hybrid:** hybrid contrastive learning, using both time and behavioral variables. Set ``conditional='time_delta'`` and ``hybrid=True``.


.. figure:: docs-imgs/samplingScheme.png
    :width: 500
    :alt: CEBRA can be used in three modes: discovery-driven, hypothesis-driven, or in a hybrid mode, which allows for weaker priors on the latent embedding.
    :align: center

    *CEBRA sampling schemes: discovery-driven, hypothesis-driven, or in a hybrid mode. In the hypothesis-driven mode, the positive and negative samples are found based on the reference samples.*

👉 Examples on how to use each of the conditional distribution and how to compare them when analyzing data are presented in :doc:`demo_notebooks/Demo_hippocampus`.


Model definition
^^^^^^^^^^^^^^^^

CEBRA training is *modular*, and model fitting can serve different downstream applications and research questions. Here, we describe how you can adjust the parameters depending on your data type and the hypotheses you might have.

.. _Model architecture:

.. rubric:: Model architecture :py:attr:`~.CEBRA.model_architecture`

We provide a set of pre-defined models. You can access (and search) a list of available pre-defined models by running:

.. testcode::

    import cebra.models
    print(cebra.models.get_options('offset*', limit = 4))

.. testoutput::

    ['offset10-model', 'offset10-model-mse', 'offset5-model', 'offset1-model-mse']

Then, you can choose the one that fits best with your needs and provide it to the CEBRA model as the :py:attr:`~.CEBRA.model_architecture` parameter.

As an indication the table below presents the model architecture we used to train CEBRA on the datasets presented in our paper (Schneider, Lee, Mathis, 2022).

.. list-table::
    :widths: 25 25 20 30
    :header-rows: 1

    * - Dataset
      - Data type
      - Brain area
      - Model architecture
    * - Artificial spiking
      - Synthetic
      -
      - 'offset1-model-mse'
    * - Rat hippocampus
      - Electrophysiology
      - CA1 hippocampus
      - 'offset10-model'
    * - Macaque
      - Electrophysiology
      - Somatosensory cortex (S1)
      - 'offset10-model'
    * - Allen Mouse
      - Calcium imaging (2P)
      - Visual cortex
      - 'offset10-model'
    * - Allen Mouse
      - Neuropixels
      - Visual cortex
      - 'offset40-model-4x-subsample'


.. dropdown:: 🚀 Optional: design your own model architectures
    :color: light

     It is possible to construct a personalized model and use the ``@cebra.models.register`` decorator on it. For example:

     .. testcode::

        from torch import nn
        import cebra.models
        import cebra.data
        from cebra.models.model import _OffsetModel, ConvolutionalModelMixin

        @cebra.models.register("my-model") # --> add that line to register the model!
        class MyModel(_OffsetModel, ConvolutionalModelMixin):

            def __init__(self, num_neurons, num_units, num_output, normalize=True):
                super().__init__(
                    nn.Conv1d(num_neurons, num_units, 2),
                    nn.GELU(),
                    nn.Conv1d(num_units, num_units, 40),
                    nn.GELU(),
                    nn.Conv1d(num_units, num_output, 5),
                    num_input=num_neurons,
                    num_output=num_output,
                    normalize=normalize,
                )

            # ... and you can also redefine the forward method,
            # as you would for a typical pytorch model

            def get_offset(self):
                return cebra.data.Offset(22, 23)

        # Access the model
        print(cebra.models.get_options('my-model'))

    .. testoutput::
        ['my-model']

    Once your personalized model is defined, you can use by setting ``model_architecture='my-model'``. 👉 See the :ref:`Models and Criteria` API for more details.

.. rubric:: Criterion and distance :py:attr:`~.CEBRA.criterion` and :py:attr:`~.CEBRA.distance`

For standard usage we recommend the default values (i.e., ``InfoNCE`` and ``cosine`` respectively) which are specifically designed for our contrastive learning algorithms.

.. rubric:: Conditional distribution :py:attr:`~.CEBRA.conditional`

👉 See the :ref:`previous section <auxiliary variables>` on how to choose the auxiliary variables and a conditional distribution.

.. note::
    If the auxiliary variables types do not match with :py:attr:`~.CEBRA.conditional`, the model training will fall back to time contrastive learning.

.. rubric:: Temperature :py:attr:`~.CEBRA.temperature`

:py:attr:`~.CEBRA.temperature` has the largest effect on visualization of the embedding (see :py:doc:`cebra-figures/figures/ExtendedDataFigure2`). Hence, it is important that it is fitted to your specific data.

The simplest way to handle it is to use a *learnable temperature*. For that, set :py:attr:`~.CEBRA.temperature_mode` to ``auto``. :py:attr:`~.CEBRA.temperature` will be trained alongside the model.

🚀 For advance usage, you might need to find the optimal :py:attr:`~.CEBRA.temperature`. For that we recommend to perform a grid-search.

👉 More examples on how to handle :py:attr:`~.CEBRA.temperature` can be found in :py:doc:`demo_notebooks/Demo_learnable_temperature`.

.. rubric:: Time offsets :math:`\Delta` :py:attr:`~.CEBRA.time_offsets`

This corresponds to the distance (in time) between positive pairs and informs the algorithm about the time-scale of interest.

The interpretation of this parameter depends on the chosen conditional distribution. A higher time offset typically will increase the difficulty of the learning task, and (within a range) improve the quality of the representation.
For time-contrastive learning, we generally recommend that the time offset should be larger than the specified receptive field of the model.

.. rubric:: Number of iterations :py:attr:`~.CEBRA.max_iterations`

We recommend to use at least 10,000 iterations to train the model. For prototyping, it can be useful to start with a smaller number (a few 1,000 iterations). However, when you notice that the loss function does not converge or the embedding looks uniformly distributed (cloud-like), we recommend increasing the number of iterations.

.. note::
    You should always assess the `convergence <https://machine-learning.paperspace.com/wiki/convergence>`_ of your model at the end of training by observing the training loss (see `Visualize the training loss`_).

.. rubric:: Number of adaptation iterations :py:attr:`~.CEBRA.max_adapt_iterations`

One feature of CEBRA is you can apply (adapt) your model to new data. If you are planning to adapt your trained model to a new set of data, we recommend to use around 500 steps to re-tuned the first layer of the model.

In the paper, we show that fine-tuning the input embedding (first layer) on the novel data while using a pretrained model can be done with 500 steps in 3.5s only, and has better performance overall.

.. rubric:: Batch size :py:attr:`~.CEBRA.batch_size`

CEBRA should be trained on the biggest batch size possible. Ideally, and depending on the size of your dataset, you should set :py:attr:`~.CEBRA.batch_size` to ``None`` (default value) which will train the model drawing samples from the full dataset at each iteration. As an indication, all the models used in the paper were trained with ``batch_size=512``. You should avoid having to set your batch size to a smaller value.

.. warning::
    Using the full dataset (``batch_size=None``) is only implemented for single-session training with continuous auxiliary variables.

Here is an example of a CEBRA model initialization:

.. testcode::

    cebra_model = CEBRA(
        model_architecture = "offset10-model",
        batch_size = 1024,
        temperature_mode="auto",
        learning_rate = 0.001,
        max_iterations = 10,
        time_offsets = 10,
        output_dimension = 8,
        device = "cuda_if_available",
        verbose = False
    )

    print(cebra_model)

.. testoutput::

    CEBRA(batch_size=1024, learning_rate=0.001, max_iterations=10,
          model_architecture='offset10-model', temperature_mode='auto',
          time_offsets=10)

.. admonition:: See API docs
    :class: dropdown

    .. autoclass:: cebra.CEBRA
       :show-inheritance:
       :noindex:

Model training
^^^^^^^^^^^^^^

Single-session versus multi-session training
""""""""""""""""""""""""""""""""""""""""""""

.. Choose the desired invariances in the embedding.
.. Add trial-by-trial once implemented

We provide both single-sesison and multi-session training. The latest makes the resulting embeddings **invariant to the auxiliary variables** across all sessions.

.. note::
    For flexibility reasons, the multi-session training fits one model for each session and thus sessions don't necessarily have the same number of features (e.g., number of neurons).

Check out the following list to verify if the multi-session implementation is the right tool for your needs.


.. |uncheck| raw:: html

    <input type="checkbox">


|uncheck| I have multiple sessions/animals that I want to consider as a pseudo-subject and use them jointly for training CEBRA. That is the case because of limited access to simultaneously recorded neurons or looking for animal-invariant features in the neural data.

|uncheck| I want to get more consistent embeddings from one session/animal to the other.

|uncheck| I want to be able to use CEBRA for a new session that is fully unseen during training.

.. warning::
    Using multi-session training limits the **influence of individual variations per session** on the embedding. Make sure that this session/animal-specific information won't be needed in your downstream analysis.


👉 Have a look at :py:doc:`demo_notebooks/Demo_hippocampus_multisession` for more in-depth usage examples of the multi-session training.

Training
""""""""

.. rubric:: Single-session training

CEBRA is trained using :py:meth:`cebra.CEBRA.fit`, similarly to the examples below for single-session training, using ``cebra_model`` as defined above. You can pass the input data as well as the behavioral labels you selected.

.. testcode::

    timesteps = 5000
    neurons = 50
    out_dim = 8

    neural_data = np.random.normal(0,1,(timesteps, neurons))
    continuous_label = np.random.normal(0,1,(timesteps, 3))
    discrete_label = np.random.randint(0,10,(timesteps,))

    single_cebra_model = cebra.CEBRA(batch_size=512,
                                     output_dimension=out_dim,
                                     max_iterations=10,
                                     max_adapt_iterations=10)


Note that the ``discrete_label`` array needs to be one dimensional, and needs to be of type :py:class:`int`.

We can now fit the model in different modes.

* For **CEBRA-Time (time-contrastive training)** with the chosen ``time_offsets``, run:

.. testcode::

    single_cebra_model.fit(neural_data)

* For **CEBRA-Behavior (supervised constrastive learning)** using **discrete labels**, run:

.. testcode::

    single_cebra_model.fit(neural_data, discrete_label)

* For **CEBRA-Behavior (supervised constrastive learning)** using **continuous labels**, run:

.. testcode::

    single_cebra_model.fit(neural_data, continuous_label)

* For **CEBRA-Behavior (supervised constrastive learning)** using a **mix of discrete and continuous labels**, run

.. testcode::

    single_cebra_model.fit(neural_data, continuous_label, discrete_label)


.. rubric:: Multi-session training

For multi-sesson training, lists of data are provided instead of a single dataset and eventual corresponding auxiliary variables.

.. warning::
    For now, multi-session training can only handle a **unique set of continuous labels**. All other combinations will raise an error.


.. testcode::

    timesteps1 = 5000
    timesteps2 = 3000
    neurons1 = 50
    neurons2 = 30
    out_dim = 8

    neural_session1 = np.random.normal(0,1,(timesteps1, neurons1))
    neural_session2 = np.random.normal(0,1,(timesteps2, neurons2))
    continuous_label1 = np.random.uniform(0,1,(timesteps1, 3))
    continuous_label2 = np.random.uniform(0,1,(timesteps2, 3))

    multi_cebra_model = cebra.CEBRA(batch_size=512,
                                    output_dimension=out_dim,
                                    max_iterations=10,
                                    max_adapt_iterations=10)

Once you defined your CEBRA model, you can run:

.. testcode::

    multi_cebra_model.fit([neural_session1, neural_session2], [continuous_label1, continuous_label2])


.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.CEBRA.fit
       :noindex:

.. rubric:: Partial training

Consistently with the ``scikit-learn`` API, :py:meth:`cebra.CEBRA.partial_fit` can be used to perform incremental learning of your model on multiple data batches.
That means by using :py:meth:`cebra.CEBRA.partial_fit`, you can fit your model on a set of data a first time and the model training will take on from the resulting
parameters to train at the next call of :py:meth:`cebra.CEBRA.partial_fit`, either on a new batch of data with the same number of features or on the same dataset.
It can be used for both single-session or multi-session training, similarly to :py:meth:`cebra.CEBRA.fit`.

.. testcode::

    cebra_model = cebra.CEBRA(max_iterations=10)

    # The model is fitted a first time ...
    cebra_model.partial_fit(neural_data)

    # ... later on the model can be fitted again
    cebra_model.partial_fit(neural_data)


.. tip::
    Partial learning is useful if your dataset is too big to fit in memory. You can separate it into multiple batches and call :py:meth:`cebra.CEBRA.partial_fit` for each data batch.


.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.CEBRA.partial_fit
       :noindex:


Saving/Loading a model
""""""""""""""""""""""

You can save a (trained/untrained) CEBRA model on your disk using :py:meth:`cebra.CEBRA.save`, and load using :py:meth:`cebra.CEBRA.load`. If the model is trained, you'll be able to load it again to transform (adapt) your dataset in a different session.

The model will be saved as a ``.pt`` file.

.. testcode::

    cebra_model = cebra.CEBRA(max_iterations=10)
    cebra_model.fit(neural_data)

    # Save the model
    cebra_model.save('/tmp/foo.pt')

    # New session: load and use the model
    loaded_cebra_model = cebra.CEBRA.load('/tmp/foo.pt')
    embedding = loaded_cebra_model.transform(neural_data)


.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.CEBRA.save
       :noindex:

    .. autofunction:: cebra.CEBRA.load
        :noindex:


.. _Grid search:

Grid search
"""""""""""

.. tip::

    A **grid-search** is the process of performing hyperparameter tuning in order to determine the optimal values of a given model. Practically, it consists in running a model on the data, by modifying the hyperparameters values at each iteration. Then, evaluating the performances of each model allows the user to select the best set of hyperparameters for its specific data.


In order to optimize a CEBRA model to the data, we recommend fine-tuning the parameters. For that, you can perform a grid-search over the hyperparameters you want to optimize.

We provide a simple hyperparameters sweep to compare CEBRA models with different parameters over different datasets or combinations of data and auxiliary variables.

.. testcode::

    import cebra

    # 1. Define the parameters, either variable or fixed
    params_grid = dict(
        output_dimension = [3, 16],
        learning_rate = [0.001],
        time_offsets = 5,
        max_iterations = 5,
        temperature_mode = "auto",
        verbose = False)

    # 2. Define the datasets to iterate over
    datasets = {"dataset1": neural_session1,                      # time contrastive learning
                "dataset2": (neural_session1, continuous_label1), # behavioral contrastive learning
                "dataset3": (neural_session2, continuous_label2)} # a different set of data

    # 3. Create and fit the grid search to your data
    grid_search = cebra.grid_search.GridSearch()
    grid_search.fit_models(datasets=datasets, params=params_grid, models_dir="saved_models")

To work on the fitted and saved models later in your work, for instance in a different file, you can call the
method of interest by providing the directory name in which the models and parameters are saved (in that case,
``saved_models``).

.. testcode::

    # 4. Get the results
    df_results = grid_search.get_df_results(models_dir="saved_models")

    # 5. Get the best model for a given dataset
    best_model, best_model_name = grid_search.get_best_model(dataset_name="dataset2", models_dir="saved_models")


.. admonition:: See API docs
    :class: dropdown

    .. autoclass:: cebra.grid_search.GridSearch
       :noindex:


Model evaluation
^^^^^^^^^^^^^^^^

Computing the embedding
"""""""""""""""""""""""

Once the model is trained, embeddings can be computed using :py:meth:`cebra.CEBRA.transform`.

.. rubric:: Single-session training

For a model trained on a single session, you just have to provide the input data on which to compte the embedding.

.. testcode::

    embedding = single_cebra_model.transform(neural_data)
    assert(embedding.shape == (timesteps, out_dim))

.. rubric:: Multi-session training

For a model trained on multiple sessions, you will need to provide the ``session_id`` (between ``0`` and ``num_sessions-1``), to select the model corresponding to the accurate number of features.

.. testcode::

    embedding = multi_cebra_model.transform(neural_session1, session_id=0)
    assert(embedding.shape == (timesteps1, out_dim))


In both case, the embedding will be of size ``time x`` :py:attr:`~.CEBRA.output_dimension`.

.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.CEBRA.transform
       :noindex:


Results visualization
^^^^^^^^^^^^^^^^^^^^^

Here, we want to emphasize that if CEBRA is providing a low-dimensional representation of your data, i.e., the embedding, there are also plenty of elements that should be checked to assess the results. We provide a post-hoc package to easily visualize the crucial information.

The visualization functions all have the same structure such that they are merely wrappers around :py:func:`matplotlib.pyplot.plot` and :py:func:`matplotlib.pyplot.scatter`. Consequently, you can provide the functions parameters to be used by those ``matplotlib.pyplot`` functions.

*Note that all examples were computed on the rat hippocampus dataset (Grosmark & Buzsáki, 2016) with default parameters,* ``max_iterations=15000`` *,* ``batch_size=512`` *,* ``model_architecture=offset10-model`` *,* ``output_dimension=3`` *except if stated otherwise.*


Displaying the embedding
""""""""""""""""""""""""

To get a 3D visualization of an embedding ``embedding``, obtained using :py:meth:`cebra.CEBRA.transform` (see above), you can use :py:func:`~.plot_embedding`.


It takes a 2D matrix representing an embedding and returns a 3D scatter plot by taking the 3 first latents by default.

.. note::
    If your embedding only has 2 dimensions, then the plot will automatically switch to a 2D mode. You can then use the function
    similarly.


.. testcode::

    cebra.plot_embedding(embedding)

.. figure:: docs-imgs/default-embedding.png
    :width: 300
    :alt: Default embedding
    :align: center


.. note::

    Be aware that the latents are not visualized by rank of importance. Consequently if your embedding is initially larger than 3, a 3D-visualization taking the first 3 latents might not be a good representation of the most relevant features. Note that you can set the parameter ``idx_order`` to select the latents to display (see API).



.. dropdown:: 🚀 Go further: personalize your embedding visualization
    :color: light

    The function is a wrapper around :py:func:`matplotlib.pyplot.scatter` and consequently accepts all the parameters of that function (e.g., ``vmin``, ``vmax``, ``alpha``, ``markersize``, ``title``, etc.) as parameters.

    Regarding the **color** of the embedding, the default value is set to ``grey`` but can be customized using the parameter ``embedding_labels``. There are 3 ways of doing it.

    * By setting ``embedding_labels`` as a valid RGB(A) color (i.e., recognized by ``matplotlib``, see `Specifying colors <https://matplotlib.org/3.1.0/tutorials/colors/colors.html>`_ for more details). You can use the following list of named colors as a good set of options already.

    .. figure:: docs-imgs/named_colors.png
        :width: 500
        :alt: Matplotlib list of named colors
        :align: center


    .. testcode::

        cebra.plot_embedding(embedding, embedding_labels="darkorchid")


    .. figure:: docs-imgs/dark_orchid-embedding.png
        :width: 300
        :alt: darkorchid embedding
        :align: center


    * By setting ``embedding_labels`` to ``time``. It will use the color map ``cmap`` to display the embedding based on temporality. By default, ``cmap=cool``. You can customize it by setting it to a valid :py:class:`matplotlib.colors.Colormap` (see `Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_ for more information). You can also use our CEBRA-custom colormap by setting ``cmap="cebra"``.

    .. figure:: docs-imgs/cebra-colormap.png
        :width: 1000
        :alt: darkorchid embedding
        :align: center

        *CEBRA-custom colormap. You can use it by calling* ``cmap="cebra"`` *.*


    In the following example, you can also see how to change the size (``markersize``) or the transparency (``alpha``) of the markers.

    .. testcode::

        cebra.plot_embedding(embedding, embedding_labels="time", cmap="magma", markersize=5, alpha=0.5)


    .. figure:: docs-imgs/time-embedding.png
        :width: 300
        :alt: Time embedding
        :align: center


    * By setting ``embedding_labels`` as a vector of same size as the embedding to be mapped to colors, using ``cmap`` (see previous point for customization). The vector can consist of a discrete label or one of the auxiliary variables for example.

    .. testcode::

        cebra.plot_embedding(embedding, embedding_labels=continuous_label[:, 0])


    .. figure:: docs-imgs/auxiliary-embedding.png
        :width: 300
        :alt: Position embedding
        :align: center

    .. note::

        ``embedding_labels`` must be uni-dimensional. Be sure to provide only one dimension of your auxiliary variables if you are using multi-dimensional continuous data for instance (e.g., only the x-coordinate of the position).


    You can specify the **latents to display** by setting ``idx_order=(latent_num_1, latent_num_2, latent_num_3)`` with ``latent_num_*`` the latent indices of your choice.
    In the following example we trained a model with ``output_dimension==10`` and we show embeddings when displaying latents (1, 2, 3) on the left and (4, 5, 6) on the right respectively. The code snippet also offers an example on how to combine multiple graphs and how to set a customized title (``title``). Note the parameter ``projection="3d"`` when adding a subplot to the figure.

    .. testcode::

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        ax1 = cebra.plot_embedding(embedding, embedding_labels=continuous_label[:,0], idx_order=(1,2,3), title="Latents: (1,2,3)", ax=ax1)
        ax2 = cebra.plot_embedding(embedding, embedding_labels=continuous_label[:,0], idx_order=(4,5,6), title="Latents: (4,5,6)", ax=ax2)


    .. figure:: docs-imgs/reordered-embedding.png
        :width: 600
        :alt: Reordered embedding
        :align: center

    If your embedding only has 2 dimensions or if you only want to display 2 dimensions from it, you can use the same function. The plot will automatically switch to 2D. Then you can use the function as usual.

    The plot will be 2D if:

    * If your embedding only has 2 dimensions and you don't specify the ``idx_order`` (then the default will be ``idx_order=(0,1)``)
    * If your embedding is more than 2 dimensions but you specify the ``idx_order`` with only 2 dimensions.

    .. testcode::

        cebra.plot_embedding(embedding, idx_order=(0,1), title="2D Embedding")


    .. figure:: docs-imgs/2D-embedding.png
        :width: 300
        :alt: 2D embedding
        :align: center

    🚀 Look at the :py:func:`~.plot_embedding` API for more details on customization.


.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.plot_embedding
       :noindex:

.. _Visualize the training loss:

Displaying the training loss
""""""""""""""""""""""""""""

Observing the training loss is of great importance. It allows you to assess that your model `converged <https://machine-learning.paperspace.com/wiki/convergence>`_ for instance or to compare models performances and fine-tune the parameters.

To visualize the loss evolution through training, you can use :py:func:`~.plot_loss`.

It takes a CEBRA model and returns a 2D plot of the loss against the number of iterations. It can be used with default values as simply as this:


.. testcode::

    cebra.plot_loss(cebra_model)


.. figure:: docs-imgs/default-loss.png
    :width: 400
    :alt: Default loss
    :align: center


🚀 The function is a wrapper around :py:func:`matplotlib.pyplot.plot` and consequently accepts all the parameters of that function (e.g., ``alpha``, ``linewidth``, ``title``, ``color``, etc.) as parameters.

.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.plot_loss
       :noindex:

Displaying the temperature
""""""""""""""""""""""""""

:py:attr:`~.CEBRA.temperature` has the largest effect on the visualization of the embedding. Hence it might be interesting to check its evolution when ``temperature_mode=auto``.

To that extend, you can use the function :py:func:`~.plot_temperature`.

It takes a CEBRA model and returns a 2D plot of the value of :py:attr:`~.CEBRA.temperature` against the number of iterations. It can be used with default values as simply as this:


.. testcode::

    cebra.plot_temperature(cebra_model)


.. figure:: docs-imgs/default-temperature.png
    :width: 400
    :alt: Default temperature
    :align: center

🚀 The function is a wrapper around :py:func:`matplotlib.pyplot.plot` and consequently accepts all the parameters of that function (e.g., ``alpha``, ``linewidth``, ``title``, ``color``, etc.) as parameters.

.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.plot_temperature
       :noindex:



Comparing models
""""""""""""""""

In order to select the most performant model, you might need to plot the training loss for a set of models on the same figure.

First, we create a list of fitted models to compare. Here we suppose we have a dataset with neural data available, as well as the position and the direction of the animal. We will show differences in performance when training with any combination of these variables.

.. testcode::

    cebra_posdir_model = CEBRA(model_architecture='offset10-model',
                    batch_size=512,
                    output_dimension=32,
                    max_iterations=5,
                    time_offsets=10)
    cebra_posdir_model.fit(neural_data, continuous_label, discrete_label)

    cebra_pos_model = CEBRA(model_architecture='offset10-model',
                batch_size=512,
                output_dimension=32,
                max_iterations=5,
                time_offsets=10)
    cebra_pos_model.fit(neural_data, continuous_label)

    cebra_dir_model = CEBRA(model_architecture='offset10-model',
                batch_size=512,
                output_dimension=32,
                max_iterations=5,
                time_offsets=10)
    cebra_dir_model.fit(neural_data, discrete_label)


Then, you can compare their losses. To do that you can use :py:func:`~.compare_models`.
It takes a list of CEBRA models and returns a 2D plot displaying their training losses.
It can be used with default values as simply as this:

.. testcode::

    import cebra

    # Labels to be used for the legend of the plot (optional)
    labels = ["position+direction", "position", "direction"]

    cebra.compare_models([cebra_posdir_model, cebra_pos_model, cebra_dir_model], labels)

.. figure:: docs-imgs/default-comparison.png
    :width: 400
    :alt: Default comparison
    :align: center

🚀 The function is a wrapper around :py:func:`matplotlib.pyplot.plot` and consequently accepts all the parameters of that function (e.g., ``alpha``, ``linewidth``, ``title``, ``color``, etc.) as parameters. Note that
however, if you want to differentiate the traces with a set of colors, you need to provide a `colormap <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_ to the ``cmap`` parameter. If you want a unique
color for all traces, you can provide a `valid color <https://matplotlib.org/3.1.0/tutorials/colors/colors.html>`_ to the ``color`` parameter that will override the ``cmap`` parameter. By default, ``color=None`` and
``cmap="cebra"`` our very special CEBRA-custom color map.


.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.compare_models
       :noindex:


What else do to with your CEBRA model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned at the start of the guide, CEBRA is much more than a visualization tool. Here we present a (non-exhaustive) list of post-hoc analysis and investigations that we support with CEBRA. Happy hacking! 👩‍💻

Consistency across features
"""""""""""""""""""""""""""

One of the major strengths of CEBRA is measuring consistency across embeddings. We demonstrate in Schneider, Lee, Mathis 2023, that consistent latents can be derived across animals (i.e., across CA1 recordings in rats), and even across recording modalities (i.e., from calcium imaging to electrophysiology recordings).

Thus, we provide the :py:func:`~.consistency_score` metrics to compute consistency across model runs or models computed on different datasets (i.e., subjects, sessions).

To use it, you have to set the ``between`` parameter to either ``datasets`` or ``runs``. The main difference between the two modes is that for between-datasets comparisons you will provide
labels to align the embeddings on. When using between-runs comparison, it supposes that the embeddings are already aligned. The simplest example being the model was run on the same dataset
but it can also be for datasets that were recorded at the same time for example, i.e., neural activity in different brain regions, recorded during the same session.

.. note::
    As consistency between CEBRA runs on the same dataset is demonstrated in Schneider, Lee, Mathis 2023 (consistent up to linear transformations), assessing consistency between different runs on the same dataset is a good way to reinsure you that you set your CEBRA model properly.

We first create the embeddings to compare: we use two different datasets of data and fit a CEBRA model three times on each.

.. testcode::

    n_runs = 3
    dataset_ids = ["session1", "session2"]

    cebra_model = CEBRA(model_architecture='offset10-model',
                    batch_size=512,
                    output_dimension=32,
                    max_iterations=5,
                    time_offsets=10)

    embeddings_runs = []
    embeddings_datasets, ids, labels = [], [], []
    for i in range(n_runs):
        embeddings_runs.append(cebra_model.fit_transform(neural_session1, continuous_label1))

    labels.append(continuous_label1[:, 0])
    embeddings_datasets.append(embeddings_runs[-1])

    embeddings_datasets.append(cebra_model.fit_transform(neural_session2, continuous_label2))
    labels.append(continuous_label2[:, 0])

    n_datasets = len(dataset_ids)

To get the :py:func:`~.consistency_score` on the set of embeddings that we just generated:

.. testcode::

    # Between-runs
    scores_runs, pairs_runs, ids_runs = cebra.sklearn.metrics.consistency_score(embeddings=embeddings_runs,
                                                                                between="runs")
    assert scores_runs.shape == (n_runs**2 - n_runs, )
    assert pairs_runs.shape == (n_runs**2 - n_runs, 2)
    assert ids_runs.shape == (n_runs, )

    # Between-datasets, by aligning on the labels
    (scores_datasets,
        pairs_datasets,
        ids_datasets) = cebra.sklearn.metrics.consistency_score(embeddings=embeddings_datasets,
                                                                    labels=labels,
                                                                    dataset_ids=dataset_ids,
                                                                    between="datasets")
    assert scores_datasets.shape == (n_datasets**2 - n_datasets, )
    assert pairs_datasets.shape == (n_datasets**2 - n_datasets, 2)
    assert ids_datasets.shape == (n_datasets, )

.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.sklearn.metrics.consistency_score
       :noindex:

You can then display the resulting scores using :py:func:`~.plot_consistency`.

.. testcode::

    fig = plt.figure(figsize=(10,4))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1 = cebra.plot_consistency(scores_runs, pairs_runs, ids_runs, vmin=0, vmax=100, ax=ax1, title="Between-runs consistencies")
    ax2 = cebra.plot_consistency(scores_datasets, pairs_datasets, ids_runs, vmin=0, vmax=100, ax=ax2, title="Between-subjects consistencies")


.. figure:: docs-imgs/consistency-score.png
    :width: 700
    :alt: Consistency scores
    :align: center

🚀 This function is a wrapper around :py:func:`matplotlib.pyplot.imshow` and, similarly to the other plot functions we provide,
it accepts all the parameters of that function (e.g., cmap, vmax, vmin, etc.) as parameters. Check the full API for more details.


.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.plot_consistency
       :noindex:

Embeddings comparison via the InfoNCE loss
""""""""""""""""""""""""""""""""""""""""""

.. rubric:: Usage case 👩‍🔬

You can also compare how a new dataset compares to prior models. This can be useful when you have several groups of data and you want to see how a new session maps to the prior models. Then you will compute :py:func:`~.infonce_loss` of the new samples compared to other models.

.. rubric:: How to use it

The performances of a given model on a dataset can be evaluated by using the :py:func:`~.infonce_loss` function. That metric
corresponds to the loss over the data, obtained using the criterion on which the model was trained (by default, ``infonce``). Hence, the
smaller that metric is, the higher the model performances on a sample are, and so the better the fit to the positive samples is.

.. note::
    As an indication, you can consider that a good trained CEBRA model should get a value for the InfoNCE loss smaller than **~6.1**. If that is not the case,
    you might want to refer to the dedicated section `Improve your model`_.

Here are examples on how you can use :py:func:`~.infonce_loss` on your data for both single-session and multi-session trained models.

.. testcode::

    # single-session
    single_score = cebra.sklearn.metrics.infonce_loss(single_cebra_model,
                                                      neural_data,
                                                      continuous_label,
                                                      discrete_label,
                                                      num_batches=5)

    # multi-session
    multi_score = cebra.sklearn.metrics.infonce_loss(multi_cebra_model,
                                                     neural_session1,
                                                     continuous_label1,
                                                     session_id=0,
                                                     num_batches=5)


.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.sklearn.metrics.infonce_loss
       :noindex:


Adapt the model to new data
"""""""""""""""""""""""""""

In some cases, it can be useful to adapt your CEBRA model to a novel dataset, with a different number of features.
For that, you can set ``adapt=True`` as a parameter of :py:meth:`cebra.CEBRA.fit`. It will reset the first layer of
the model so that the input dimension corresponds to the new features dimensions and retrain it for
:py:attr:`cebra.CEBRA.max_adapt_iterations`. You can set that parameter :py:attr:`cebra.CEBRA.max_adapt_iterations`
when initializing your :py:class:`cebra.CEBRA` model.

.. note::
    Adapting your CEBRA model to novel data is only implemented for single session training. Make sure that your model was trained on a single dataset.

.. testcode::

    # Fit your model once ...
    single_cebra_model.fit(neural_session1)

    # ... do something with it (embedding, visualization, saving) ...

    # ... and adapt the model
    cebra_model.fit(neural_session2, adapt=True)


.. note::
    We recommend that you save your model, using :py:meth:`cebra.CEBRA.save`, before adapting it to a different dataset.
    The adapted model will replace the previous model in ``cebra_model.state_dict_`` so saving it beforehand allows you
    to keep the trained parameters for later. You can then load the model again, using :py:meth:`cebra.CEBRA.load` whenever
    you need it.


.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.CEBRA.fit
       :noindex:


Decoding
""""""""

The CEBRA latent embedding can be used for decoding analysis, meaning to investigate if a specific variable in the task can be decoded from the latent embeddings.
Decoding using the embedding can be easily perform by mean of the decoders we implemented as part of CEBRA and following the ``scikit-learn`` API. We provide
two decoders: :py:class:`~.KNNDecoder` and :py:class:`~.L1LinearRegressor`. Here is a simple usage of the :py:class:`~.KNNDecoder` after using CEBRA-Time.

.. testcode::

    from sklearn.model_selection import train_test_split

    # 1. Train a CEBRA-Time model on the whole dataset
    cebra_model = cebra.CEBRA(max_iterations=10)
    cebra_model.fit(neural_data)
    embedding = cebra_model.transform(neural_data)

    # 2. Split the embedding and label to decode into train/validation sets
    (
         train_embedding,
         valid_embedding,
         train_discrete_label,
         valid_discrete_label,
    ) = train_test_split(embedding,
                         discrete_label,
                         test_size=0.3)

    # 3. Train the decoder on the training set
    decoder = cebra.KNNDecoder()
    decoder.fit(train_embedding, train_discrete_label)

    # 4. Get the score on the validation set
    score = decoder.score(valid_embedding, valid_discrete_label)

    # 5. Get the discrete labels predictions
    prediction = decoder.predict(valid_embedding)

``prediction`` contains the predictions of the decoder on the discrete labels.

.. warning::
    Be careful to avoid `double dipping <https://www.nature.com/articles/nn.2303>`_ when using the decoder. The previous example uses time contrastive learning.
    If you are using CEBRA-Behavior or CEBRA-Hybrid and you consequently use labels, you will have to split your
    original data from start as you don't want decode labels from an embedding that is itself trained on those labels.


.. dropdown:: 👉 Decoder example with CEBRA-Behavior
    :color: light

    .. testcode::

        from sklearn.model_selection import train_test_split

        # 1. Split your neural data and auxiliary variable
        (
            train_data,
            valid_data,
            train_discrete_label,
            valid_discrete_label,
        ) = train_test_split(neural_data,
                             discrete_label,
                             test_size=0.2)

        # 2. Train a CEBRA-Behavior model on training data only
        cebra_model = cebra.CEBRA(max_iterations=10, batch_size=512)
        cebra_model.fit(train_data, train_discrete_label)

        # 3. Get embedding for training and validation data
        train_embedding = cebra_model.transform(train_data)
        valid_embedding = cebra_model.transform(valid_data)

        # 4. Train the decoder on training embedding and labels
        decoder = cebra.KNNDecoder()
        decoder.fit(train_embedding, train_discrete_label)

        # 5. Compute the score on validation embedding and labels
        score = decoder.score(valid_embedding, valid_discrete_label)



.. admonition:: See API docs
    :class: dropdown

    .. autofunction:: cebra.KNNDecoder.fit
       :noindex:

    .. autofunction:: cebra.KNNDecoder.score
       :noindex:


.. _Improve your model:

Improve model performance
^^^^^^^^^^^^^^^^^^^^^^^^^

🧐 Below is a (non-exhaustive) list of actions you can try if your embedding looks different from what you were expecting.

#. Assess that your model `converged <https://machine-learning.paperspace.com/wiki/convergence>`_. For that, observe if the training loss stabilizes itself around the end of the training or still seems to be decreasing. Refer to `Visualize the training loss`_ for more details on how to display the training loss.
#. Increase the number of iterations. It should be at least 10,000.
#. Make sure the batch size is big enough. It should be at least 512.
#. Fine-tune the model's hyperparameters, namely ``learning_rate``, ``output_dimension``, ``num_hidden_units`` and eventually ``temperature`` (by setting ``temperature_mode`` back to ``constant``). Refer to `Grid search`_ for more details on performing hyperparameters tuning.



Quick Start: Scikit-learn API example
-------------------------------------

Putting all previous snippet examples together, we obtain the following pipeline.

.. testcode::

     import cebra
     from numpy.random import uniform, randint
     from sklearn.model_selection import train_test_split

     # 1. Define a CEBRA model
     cebra_model = cebra.CEBRA(
         model_architecture = "offset10-model",
         batch_size = 512,
         learning_rate = 1e-4,
         max_iterations = 10, # TODO(user): to change to at least 10'000
         max_adapt_iterations = 10, # TODO(user): to change to ~100-500
         time_offsets = 10,
         output_dimension = 8,
         verbose = False
     )

     # 2. Load example data
     neural_data = cebra.load_data(file="neural_data.npz", key="neural")
     new_neural_data = cebra.load_data(file="neural_data.npz", key="new_neural")
     continuous_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
     discrete_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["discrete"]).flatten()

     assert neural_data.shape == (100, 3)
     assert new_neural_data.shape == (100, 4)
     assert discrete_label.shape == (100, )
     assert continuous_label.shape == (100, 3)

     # 3. Split data and labels
     (
         train_data,
         valid_data,
         train_discrete_label,
         valid_discrete_label,
         train_continuous_label,
         valid_continuous_label,
     ) = train_test_split(neural_data,
                         discrete_label,
                         continuous_label,
                         test_size=0.3)

     # 4. Fit the model
     # time contrastive learning
     cebra_model.fit(train_data)
     # discrete behavior contrastive learning
     cebra_model.fit(train_data, train_discrete_label,)
     # continuous behavior contrastive learning
     cebra_model.fit(train_data, train_continuous_label)
     # mixed behavior contrastive learning
     cebra_model.fit(train_data, train_discrete_label, train_continuous_label)

     # 5. Save the model
     cebra_model.save('/tmp/foo.pt')

     # 6. Load the model and compute an embedding
     cebra_model = cebra.CEBRA.load('/tmp/foo.pt')
     train_embedding = cebra_model.transform(train_data)
     valid_embedding = cebra_model.transform(valid_data)
     assert train_embedding.shape == (70, 8)
     assert valid_embedding.shape == (30, 8)

     # 7. Evaluate the model performances
     goodness_of_fit = cebra.sklearn.metrics.infonce_loss(cebra_model,
                                                          valid_data,
                                                          valid_discrete_label,
                                                          valid_continuous_label,
                                                          num_batches=5)

     # 8. Adapt the model to a new session
     cebra_model.fit(new_neural_data, adapt = True)

     # 9. Decode discrete labels behavior from the embedding
     decoder = cebra.KNNDecoder()
     decoder.fit(train_embedding, train_discrete_label)
     prediction = decoder.predict(valid_embedding)
     assert prediction.shape == (30,)

👉 For further guidance on different/customized applications of CEBRA on your own data, refer to the ``examples/`` folder or to the full documentation folder ``docs/``.


Quick Start: Torch API example
------------------------------

🚀 You have special custom data analysis needs? We invite you to use the ``torch``-API interface.

Refer to the ``examples/`` folder for a set of demo scripts.
Single- and multi-session training can be launched using the following ``bash`` command.

.. code:: bash

    $ PYTHONPATH=. python examples/train.py [customized arguments]

Below is the documentation on the available arguments.

.. code:: bash

    $ PYTHONPATH=. python examples/train.py --help
    usage: train.py [-h] [--data <dataclasses._MISSING_TYPE object at 0x7f2eeb13f070>] [--variant single-session]
                    [--logdir /logs/single-rat-hippocampus-behavior/] [--loss-distance cosine] [--temperature 1]
                    [--time-offset 10] [--conditional time_delta] [--num-steps 1000] [--learning-rate 0.0003]
                    [--model offset10-model] [--batch-size 512] [--num-hidden-units 32] [--num-output 8] [--device cpu]
                    [--tqdm False] [--save-frequency SAVE_FREQUENCY] [--valid-frequency 100] [--train-ratio 0.8]
                    [--valid-ratio 0.1] [--share-model]

    CEBRA Demo

    options:
    -h, --help            show this help message and exit
    --data <dataclasses._MISSING_TYPE object at 0x7f2eeb13f070>
                            The dataset to run CEBRA on. Standard datasets are available in cebra.datasets. Your own datasets can
                            be created by subclassing cebra.data.Dataset and registering the dataset using the
                            ``@cebra.datasets.register`` decorator.
    --variant single-session
                            The CEBRA variant to run.
    --logdir /logs/single-rat-hippocampus-behavior/
                            Model log directory. This should be either a new empty directory, or a pre-existing directory
                            containing a trained CEBRA model.
    --loss-distance cosine
                            Distance type to use in calculating loss
    --temperature 1       Temperature for InfoNCE loss
    --time-offset 10      Distance (in time) between positive pairs. The interpretation of this parameter depends on the chosen
                            conditional distribution, but generally a higher time offset increases the difficulty of the learning
                            task, and (in a certain range) improves the quality of the representation. The time offset would
                            typically be larger than the specified receptive field of the model.
    --conditional time_delta
                            Type of conditional distribution. Valid standard methods are "time_delta" and "time", and more
                            methods can be added to the ``cebra.data`` registry.
    --num-steps 1000      Number of total training steps. Number of total training steps. Note that training duration of CEBRA
                            is independent of the dataset size. The total training examples seen will amount to ``num-steps x
                            batch-size``, irrespective of dataset size.
    --learning-rate 0.0003
                            Learning rate for Adam optimizer.
    --model offset10-model
                            Model architecture. Available options are 'offset10-model', 'offset5-model' and 'offset1-model'.
    --batch-size 512      Total batch size for each training step.
    --num-hidden-units 32
                            Number of hidden units.
    --num-output 8        Dimension of output embedding
    --device cpu          Device for training. Options: cpu/cuda
    --tqdm False          Activate tqdm for logging during the training
    --save-frequency SAVE_FREQUENCY
                            Interval of saving intermediate model
    --valid-frequency 100
                            Interval of validation in training
    --train-ratio 0.8     Ratio of train dataset. The remaining will be used for valid and test split.
    --valid-ratio 0.1     Ratio of validation set after the train data split. The remaining will be test split
    --share-model

Model initialization using the Torch API 
----------------------------------------

The scikit-learn API provides parametrization to many common use cases.
The Torch API however allows for more flexibility and customization, for e.g. 
sampling, criterions, and data loaders.

In this minimal example we show how to initialize a CEBRA model using the Torch API.
Here the :py:class:`cebra.data.single_session.DiscreteDataLoader` 
gets initialized which also allows the `prior` to be directly parametrized.

👉 For an example notebook using the Torch API check out the :doc:`demo_notebooks/Demo_Allen`.


.. testcode::

    import numpy as np
    import cebra.datasets
    import torch
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    neural_data = cebra.load_data(file="neural_data.npz", key="neural")
    
    discrete_label = cebra.load_data(
        file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["discrete"],
    )
    
    # 1. Define Cebra Dataset
    input_data = cebra.data.TensorDataset(
        torch.from_numpy(neural_data).type(torch.FloatTensor),
        discrete=torch.from_numpy(np.array(discrete_label[:, 0])).type(torch.LongTensor),
    ).to(device)
    
    # 2. Define Cebra Model
    neural_model = cebra.models.init(
        name="offset10-model",
        num_neurons=input_data.input_dimension,
        num_units=32,
        num_output=2,
    ).to(device)
    
    input_data.configure_for(neural_model)
    
    # 3. Define Loss Function Criterion and Optimizer
    crit = cebra.models.criterions.LearnableCosineInfoNCE(
        temperature=0.001,
        min_temperature=0.0001
    ).to(device)
    
    opt = torch.optim.Adam(
        list(neural_model.parameters()) + list(crit.parameters()),
        lr=0.001,
        weight_decay=0,
    )
    
    # 4. Initialize Cebra Model
    solver = cebra.solver.init(
        name="single-session",
        model=neural_model,
        criterion=crit,
        optimizer=opt,
        tqdm_on=True,
    ).to(device)
    
    # 5. Define Data Loader
    loader = cebra.data.single_session.DiscreteDataLoader(
        dataset=input_data, num_steps=10, batch_size=200, prior="uniform"
    )
    
    # 6. Fit Model
    solver.fit(loader=loader)
    
    # 7. Transform Embedding
    train_batches = np.lib.stride_tricks.sliding_window_view(
        neural_data, neural_model.get_offset().__len__(), axis=0
    )
    
    x_train_emb = solver.transform(
        torch.from_numpy(train_batches[:]).type(torch.FloatTensor).to(device)
    ).to(device)
    
    # 8. Plot Embedding
    cebra.plot_embedding(
        x_train_emb,
        discrete_label[neural_model.get_offset().__len__() - 1 :, 0],
        markersize=10,
    )
