

    you might want to refer to the dedicated section `Improve your model`_.

.. _Improve your model:



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

