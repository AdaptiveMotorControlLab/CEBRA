Using CEBRA
===========

This page covers a standard CEBRA usage. We recommend checking out the :py:doc:`demos` for in-depth CEBRA usage examples as well. Here we present a quick overview on how to use CEBRA on various datasets. Note that we provide two ways to interact with the code: 

* Upon specific needs, advanced users might consider diving into the **low-level interface** that adheres to ``PyTorch`` formatting. 

Firstly, why use CEBRA?
-----------------------


That being said, CEBRA can be used on non-time-series data and it does not strictly require multi-modal data. In general, we recommend considering using CEBRA for measuring changes in consistency across conditions (brain areas, cells, animals), for hypothesis-guided decoding, and for toplogical exploration of the resulting embedding spaces. It can also be used for visualization and considering dynamics within the embedding space. For examples of how CEBRA can be used to map space, decode natural movies, and make hypotheses for neural coding of sensorimotor systems, see our paper (Schneider, Lee, Mathis, 2023).

The CEBRA workflow
------------------

CEBRA supports three modes: fully unsupervised (CEBRA-Time), supervised (via joint modeling of auxiliary variables; CEBRA-Behavior), and a hybrid variant (CEBRA-Hybrid).

(1) Use CEBRA-Time for unsupervised data exploration.
(2) Consider running a hyperparameter sweep on the inputs to the model, such as :py:attr:`cebra.CEBRA.model_architecture`, :py:attr:`cebra.CEBRA.time_offsets`, :py:attr:`cebra.CEBRA.output_dimension`, and set :py:attr:`cebra.CEBRA.batch_size` to be as high as your GPU allows. You want to see clear structure in the 3D plot (the first 3 latents are shown by default).
(3) Use CEBRA-Behavior with many different labels and combinations, then look at the InfoNCE loss - the lower the loss value, the better the fit (see :py:doc:`cebra-figures/figures/ExtendedDataFigure5`), and visualize the embeddings. The goal is to understand which labels are contributing to the structure you see in CEBRA-Time, and improve this structure. Again, you should consider a hyperparameter sweep.
(4) Interpretability: now you can use these latents in downstream tasks, such as measuring consistency, decoding, and determining the dimensionality of your data with topological data analysis.

All the steps to do this are described below. Enjoy using CEBRA! 🔥🦓


Create a CEBRA workspace
^^^^^^^^^^^^^^^^^^^^^^^^



Choose the CEBRA mode and related auxiliary variables
""""""""""""""""""""""""""""""""""""""""""""""""""""""
CEBRA allows you to jointly use time-series data and (optionally) auxiliary variables to extract latent spaces. If you want to use time-only (namely, unsupervised) select:
* **CEBRA-Time:** Discovery-driven: time contrastive learning. Set ``conditional='time'``. No assumption on the  behaviors that are influencing neural activity. It can be used as a first step into the data analysis for instance, or as a comparison point to multiple hypothesis-driven analyses.
To use auxiliary (behavioral) variables you can choose both continuous and discrete variables. The label information (none, discrete, continuous) determine the algorithm to use for data sampling. Using labels allows you to project future behavior onto past time-series activity, and explicitly use label-prior to shape the embedding. The conditional distribution can be chosen upon model initialization with the :py:attr:`cebra.CEBRA.conditional` parameter. 
.. figure:: docs-imgs/samplingScheme.png
    :width: 500
Model definition
^^^^^^^^^^^^^^^^
CEBRA training is *modular*, and model fitting can serve different downstream applications and research questions. Here, we describe how you can adjust the parameters depending on your data type and the hypotheses you might have. 
We provide a set of pre-defined models. You can access (and search) a list of available pre-defined models by running:
.. dropdown:: 🚀 Optional: design your own model architectures
     It is possible to construct a personalized model and use the ``@cebra.models.register`` decorator on it. For example:
For time-contrastive learning, we generally recommend that the time offset should be larger than the specified receptive field of the model.
We recommend to use at least 10,000 iterations to train the model. For prototyping, it can be useful to start with a smaller number (a few 1,000 iterations). However, when you notice that the loss function does not converge or the embedding looks uniformly distributed (cloud-like), we recommend increasing the number of iterations.
One feature of CEBRA is you can apply (adapt) your model to new data. If you are planning to adapt your trained model to a new set of data, we recommend to use around 500 steps to re-tuned the first layer of the model. 
In the paper, we show that fine-tuning the input embedding (first layer) on the novel data while using a pretrained model can be done with 500 steps in 3.5s only, and has better performance overall.
    Using the full dataset (``batch_size=None``) is only implemented for single-session training with continuous auxiliary variables. 
Here is an example of a CEBRA model initialization:
Model training
^^^^^^^^^^^^^^
    For flexibility reasons, the multi-session training fits one model for each session and thus sessions don't necessarily have the same number of features (e.g., number of neurons). 
    Using multi-session training limits the **influence of individual variations per session** on the embedding. Make sure that this session/animal-specific information won't be needed in your downstream analysis.
👉 Have a look at :py:doc:`demo_notebooks/Demo_hippocampus_multisession` for more in-depth usage examples of the multi-session training. 
* For **CEBRA-Time (time-contrastive training)** with the chosen ``time_offsets``, run:
* For **CEBRA-Behavior (supervised constrastive learning)** using **discrete labels**, run: 
* For **CEBRA-Behavior (supervised constrastive learning)** using **continuous labels**, run: 
* For **CEBRA-Behavior (supervised constrastive learning)** using a **mix of discrete and continuous labels**, run
.. rubric:: Multi-session training
For multi-sesson training, lists of data are provided instead of a single dataset and eventual corresponding auxiliary variables.
    For now, multi-session training can only handle a **unique set of continuous labels**. All other combinations will raise an error.
Once you defined your CEBRA model, you can run:
You can save a (trained/untrained) CEBRA model on your disk using :py:meth:`cebra.CEBRA.save`, and load using :py:meth:`cebra.CEBRA.load`. If the model is trained, you'll be able to load it again to transform (adapt) your dataset in a different session.
The model will be saved as a ``.pt`` file.
Model evaluation
^^^^^^^^^^^^^^^^
Computing the embedding
"""""""""""""""""""""""

    Be aware that the latents are not visualized by rank of importance. Consequently if your embedding is initially larger than 3, a 3D-visualization taking the first 3 latents might not be a good representation of the most relevant features. Note that you can set the parameter ``idx_order`` to select the latents to display (see API).


What else do to with your CEBRA model 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned at the start of the guide, CEBRA is much more than a visualization tool. Here we present a (non-exhaustive) list of post-hoc analysis and investigations that we support with CEBRA. Happy hacking! 👩‍💻

Consistency across features
"""""""""""""""""""""""""""


Embeddings comparison via the InfoNCE loss 
""""""""""""""""""""""""""""""""""""""""""
.. rubric:: How to use it
smaller that metric is, the higher the model performances on a sample are, and so the better the fit to the positive samples is. 
    you might want to refer to the dedicated section `Improve your model`_.

.. _Improve your model:

Improve model performance
^^^^^^^^^^^^^^^^^^^^^^^^^
🧐 Below is a (non-exhaustive) list of actions you can try if your embedding looks different from what you were expecting. 
#. Increase the number of iterations. It should be at least 10,000.

Quick Start: Scikit-learn API example
-------------------------------------



     from numpy.random import uniform, randint
         model_architecture = "offset10-model",
         learning_rate = 1e-4,
         time_offsets = 10,
         output_dimension = 8,
         verbose = False
     )



Quick Start: Torch API example
------------------------------




.. code:: bash

    usage: train.py [-h] [--data <dataclasses._MISSING_TYPE object at 0x7f2eeb13f070>] [--variant single-session]
                    [--logdir /logs/single-rat-hippocampus-behavior/] [--loss-distance cosine] [--temperature 1]
                    [--time-offset 10] [--conditional time_delta] [--num-steps 1000] [--learning-rate 0.0003]
                    [--model offset10-model] [--batch-size 512] [--num-hidden-units 32] [--num-output 8] [--device cpu]
                    [--tqdm False] [--save-frequency SAVE_FREQUENCY] [--valid-frequency 100] [--train-ratio 0.8]
                    [--valid-ratio 0.1] [--share-model]

    CEBRA Demo

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
    --share-model

