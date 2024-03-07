#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Define the CEBRA model."""

import copy
import itertools
import warnings
from typing import (Callable, Dict, Iterable, List, Literal, Optional, Tuple,
                    Union)

import numpy as np
import numpy.typing as npt
import pkg_resources
import sklearn.utils.validation as sklearn_utils_validation
import torch
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from torch import nn

import cebra.data
import cebra.integrations.sklearn
import cebra.integrations.sklearn.dataset as cebra_sklearn_dataset
import cebra.integrations.sklearn.utils as sklearn_utils
import cebra.models
import cebra.solver


def _init_loader(
    is_cont: bool,
    is_disc: bool,
    is_full: bool,
    is_multi: bool,
    is_hybrid: bool,
    shared_kwargs: dict,
    extra_kwargs: dict,
) -> Tuple[cebra.data.Loader, str]:
    """Select the right dataloader for the dataset and given arguments.

    Args:
        is_cont: Use continuous data loaders for training
        is_disc: Use a discrete data loader for training
        is_full: Train using the full dataset (batch gradient descent) instead of
            using stochastic gradient descent on mini batches
        is_multi: Use multi-session training
        is_hybrid: Jointly train on time and behavior
        shared_kwargs: Keyword arguments that will always be passed to the
            data loader. See ``extra_kwargs`` for arguments that are only relevant
            for some dataloaders.
        extra_kwargs: Additional keyword arguments used for other parts of the
            algorithm, which might (depending on the arguments for this function)
            be passed to the data loader.

    Raises:
        ValueError: If an argument is missing in ``extra_kwargs`` or ``shared_kwargs``
            needed to run the requested configuration.
        NotImplementedError: If the requested combinations of arguments is not yet
            implemented. If this error occurs, check if the desired functionality
            is implemented in :py:mod:`cebra.data`, and consider using the CEBRA
            PyTorch API directly.

    Returns:
        the data loader and name of a suitable solver

    Note:
        Not all dataloading options are implemented yet and this function can
        raise a ``NotImplementedError`` for invalid argument combinations.
        The pytorch API should be directly used in these cases.
    """

    raise_not_implemented_error = False
    incompatible_combinations = [
        # It is required to pass either a continuous or discrete index for hybrid
        # training
        (not is_cont, not is_disc, is_hybrid),
        # It is required to either pass a continuous or discrete index for multi-
        # session training
        (not is_cont, not is_disc, is_multi),
    ]
    if any(all(combination) for combination in incompatible_combinations):
        raise ValueError(f"Invalid index combination.\n"
                         f"Continuous: {is_cont},\n"
                         f"Discrete: {is_disc},\n"
                         f"Hybrid training: {is_hybrid},\n"
                         f"Full dataset: {is_full}.")

    if "conditional" in extra_kwargs:
        if extra_kwargs["conditional"] is None:
            del extra_kwargs["conditional"]

    if "time_offsets" in extra_kwargs:
        time_offsets = extra_kwargs["time_offsets"]
        if isinstance(time_offsets, Iterable):
            if len(time_offsets) > 1:
                raise NotImplementedError(
                    "Support for multiple time offsets is not yet implemented, "
                    f"but got {time_offsets}.")
            else:
                (time_offsets,) = time_offsets
            extra_kwargs["time_offsets"] = time_offsets
            if not isinstance(extra_kwargs["time_offsets"], int):
                raise TypeError(
                    f"Invalid type for time_offsets: {type(time_offsets)}")

    def _require_arg(key):
        if key not in extra_kwargs:
            raise ValueError(
                f"You need to specify '{key}' to run CEBRA in the selected configuration."
            )

    # TODO(celia): need to adapt _prepare_dataset() once more modes are added
    if is_multi:
        if not is_cont and not is_disc:
            raise_not_implemented_error = True

        # Continual behavior contrastive training is selected with "time_delta" distribution
        # as its default when a continual variable is specified.
        if is_cont and not is_disc:
            kwargs = dict(
                conditional=extra_kwargs.get("conditional", "time_delta"),
                time_offset=extra_kwargs["time_offsets"],
                **shared_kwargs,
            )
            if is_full:
                raise_not_implemented_error = True
            else:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    return (
                        cebra.data.ContinuousMultiSessionDataLoader(**kwargs),
                        "multi-session",
                    )

        # Discrete behavior contrastive training is selected with the default dataloader
        if not is_cont and is_disc:
            if is_full:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    raise_not_implemented_error = True
            else:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    raise_not_implemented_error = True

        # Mixed behavior contrastive training is selected with the default dataloader
        if is_cont and is_disc:
            if is_full:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    raise_not_implemented_error = True
            else:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    raise_not_implemented_error = True
    else:
        # Select time contrastive learning as the fallback option when no behavior variables
        # are provided.
        if not is_cont and not is_disc:
            kwargs = dict(
                conditional="time",
                time_offset=extra_kwargs["time_offsets"],
                **shared_kwargs,
            )
            if is_full:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    return cebra.data.FullDataLoader(
                        **kwargs), "single-session-full"
            else:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    return cebra.data.ContinuousDataLoader(
                        **kwargs), "single-session"

        # Continual behavior contrastive training is selected with "time_delta" distribution
        # as its default when a continual variable is specified.
        if is_cont and not is_disc:
            kwargs = dict(
                conditional=extra_kwargs.get("conditional", "time_delta"),
                time_offset=extra_kwargs["time_offsets"],
                delta=extra_kwargs["delta"],
                **shared_kwargs,
            )
            if is_full:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    return cebra.data.FullDataLoader(
                        **kwargs), "single-session-full"
            else:
                if is_hybrid:
                    return (
                        cebra.data.HybridDataLoader(**kwargs),
                        "single-session-hybrid",
                    )
                else:
                    return cebra.data.ContinuousDataLoader(
                        **kwargs), "single-session"

        # Discrete behavior contrastive training is selected with the default dataloader
        if not is_cont and is_disc:
            if is_full:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    raise_not_implemented_error = True
            else:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    return (
                        cebra.data.DiscreteDataLoader(**shared_kwargs),
                        "single-session",
                    )

        # Mixed behavior contrastive training is selected with the default dataloader
        if is_cont and is_disc:
            if is_full:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    raise_not_implemented_error = True
            else:
                if is_hybrid:
                    raise_not_implemented_error = True
                else:
                    return (
                        cebra.data.MixedDataLoader(
                            time_offset=extra_kwargs["time_offsets"],
                            **shared_kwargs),
                        "single-session",
                    )

    error_message = (f"Invalid index combination.\n"
                     f"Continuous: {is_cont},\n"
                     f"Discrete: {is_disc},\n"
                     f"Hybrid training: {is_hybrid},\n"
                     f"Full dataset: {is_full}.")

    if raise_not_implemented_error:
        raise NotImplementedError(error_message + (
            " "
            "This index combination might still be implemented in the future. "
            "Until then, please train using the PyTorch API."))
    else:
        raise RuntimeError(
            f"Index combination not covered. Please report this issue and add the following "
            f"information to your bug report: \n" + error_message)


def _check_type_checkpoint(checkpoint):
    if not isinstance(checkpoint, cebra.CEBRA):
        raise RuntimeError("Model loaded from file is not compatible with "
                           "the current CEBRA version.")
    if not sklearn_utils.check_fitted(checkpoint):
        raise ValueError(
            "CEBRA model is not fitted. Loading it is not supported.")

    return checkpoint


def _load_cebra_with_sklearn_backend(cebra_info: Dict) -> "CEBRA":
    """Loads a CEBRA model with a Sklearn backend.

    Args:
        cebra_info: A dictionary containing information about the CEBRA object,
            including the arguments, the state of the object and the state
            dictionary of the model.

    Returns:
       The loaded CEBRA object.

    Raises:
        ValueError: If the loaded CEBRA model was not already fit, indicating that loading it is not supported.
    """
    required_keys = ['args', 'state', 'state_dict']
    missing_keys = [key for key in required_keys if key not in cebra_info]
    if missing_keys:
        raise ValueError(
            f"Missing keys in data dictionary: {', '.join(missing_keys)}. "
            f"You can try loading the CEBRA model with the torch backend.")

    args, state, state_dict = cebra_info['args'], cebra_info[
        'state'], cebra_info['state_dict']
    cebra_ = cebra.CEBRA(**args)

    for key, value in state.items():
        setattr(cebra_, key, value)

    state_and_args = {**args, **state}

    if not sklearn_utils.check_fitted(cebra_):
        raise ValueError(
            "CEBRA model was not already fit. Loading it is not supported.")

    if cebra_.num_sessions_ is None:
        model = cebra.models.init(
            args["model_architecture"],
            num_neurons=state["n_features_in_"],
            num_units=args["num_hidden_units"],
            num_output=args["output_dimension"],
        ).to(state['device_'])

    elif isinstance(cebra_.num_sessions_, int):
        model = nn.ModuleList([
            cebra.models.init(
                args["model_architecture"],
                num_neurons=n_features,
                num_units=args["num_hidden_units"],
                num_output=args["output_dimension"],
            ) for n_features in state["n_features_in_"]
        ]).to(state['device_'])

    criterion = cebra_._prepare_criterion()
    criterion.to(state['device_'])

    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), criterion.parameters()),
        lr=args['learning_rate'],
        **dict(args['optimizer_kwargs']),
    )

    solver = cebra.solver.init(
        state['solver_name_'],
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        tqdm_on=args['verbose'],
    )
    solver.load_state_dict(state_dict)
    solver.to(state['device_'])

    cebra_.model_ = model
    cebra_.solver_ = solver

    return cebra_


class CEBRA(BaseEstimator, TransformerMixin):
    """CEBRA model defined as part of a ``scikit-learn``-like API.

    Attributes:
        model_architecture (str):
            The architecture of the neural network model trained with contrastive
            learning to encode the data. We provide a list of pre-defined models which can be displayed by running
            :py:func:`cebra.models.get_options`. The user can also register their own custom models (see in
            :ref:`Docs <Model architecture>`). |Default:| ``offset1-model``
        device (str):
            The device used for computing. Choose from ``cpu``, ``cuda``, ``cuda_if_available'`` or a
            particular GPU via ``cuda:0``. |Default:| ``cuda_if_available``
        criterion (str):
            The training objective. Currently only the default ``InfoNCE`` is supported. The InfoNCE loss
            is specifically designed for contrastive learning. |Default:| ``InfoNCE``
        distance (str):
            The distance function used in the training objective to define the positive and negative
            samples with respect to the reference samples. Currently supports ``cosine`` and ``euclidean`` distances,
            ``cosine`` being specifically adapted for contrastive learning. |Default:| ``cosine``
        conditional (str):
            The conditional distribution to use to sample the positive samples. Reference and negative samples
            are drawn from a uniform prior. For positive samples, it currently supports 3 types of distributions:
            ``time_delta``, ``time`` and ``delta``.
            Positive sample are distributed around the reference samples using either time information (``time``)
            with  a fixed :py:attr:`~.time_offset` from the reference samples' time steps, or the auxililary
            variables, considering the empirical distribution of how behavior vary across :py:attr:`~.time_offset`
            timesteps (``time_delta``). Alternatively (``delta``), the distribution is set as a Gaussian
            distribution, parametrized by a fixed ``delta`` around the reference sample. |Default:| ``None``
        temperature (float):
            Factor by which to scale the similarity of the positive
            and negative pairs in the InfoNCE loss. Higher values yield "sharper", more concentrated embeddings.
            |Default:| ``1.0``
        temperature_mode (str):
            The ``constant`` mode uses a temperature set by the user, constant
            during training. The ``auto`` mode trains the temperature alongside the model. If set to ``auto``.
            In that case, make sure to also set :py:attr:`~.min_temperature` to a value in the expected range
            (for that a simple grid-search over :py:attr:`~.temperature` can be run). Note that the ``auto``
            mode is an experimental feature for now. |Default:| ``constant``
        min_temperature (float):
            The minimum temperature to maintain in case the temperature is optimized with the model
            (when setting :py:attr:`temperature_mode` to ``auto``). This parameter will be ignored if
            :py:attr:`~.temperature_mode` is set to ``constant``. Select ``None`` if no constraint
            should be applied. |Default:| ``0.1``
        time_offsets (int):
            The offsets for building the empirical distribution within the chosen sampler. It can be a
            single value, or a tuple of values to sample from uniformly. Will only have an effect if
            :py:attr:`~.conditional` is set to ``time`` or ``time_delta``.
        max_iterations (int):
            The number of iterations to train for. To pick the optimal number of iterations, start with
            a lower number (like 1,000) for faster training, and observe the value of the loss function (see
            :py:func:`~.plot_loss` to display the model loss over training). Make sure to pick a number
            of iterations high enough for the loss to converge. |Default:| ``10000``.
        max_adapt_iterations (int):
            The number of samples for retraining the first layer when adapting the model to a new
            dataset. This parameter is only relevant when ``adapt=True`` in :py:meth:`cebra.CEBRA.fit`.
            |Default:| ``500``.
        batch_size (int):
            The batch size to use for training. If RAM or GPU memory allows, this parameter can be set to
            ``None`` to select batch gradient descent on the whole dataset. If you use mini-batch training,
            you should aim for a value greater than 512. Higher values typically get better results and
            smoother loss curves. |Default:| ``None``.
        learning_rate (float):
            The learning rate for optimization. Higher learning rates *can* yield faster convergence, but
            also lead to instability. Tune this parameter along with :py:attr:~.temperature`. For stable
            training with lower temperatures, it can make sense to lower the learning rate, and train a
            bit longer.
            |Default:| ``0.0003``.
        optimizer (str):
            The optimizer to use. Refer to :py:mod:`torch.optim` for all possible optimizers. Right now,
            only ``adam`` is supported. |Default:| ``adam``.
        output_dimension (int):
            The output dimensionality of the embedding. For visualization purposes, this can be set to 3
            for an embedding based on the cosine distance and 2-3 for an embedding based on the Eulidean
            distance (see :py:attr:`~.distance`). Alternatively, fit an embedding with a higher output
            dimensionality and then perform a linear ICA on top to visualize individual components.
            |Default:| ``8``.
        verbose (bool):
            If ``True``, show a progress bar during training. |Default:| ``False``.
        num_hidden_units (int):
            The number of dimensions to use within the neural network model. Higher numbers slow down training,
            but make the model more expressive and can result in a better embedding. Especially if you find
            that the embeddings are not consistent across runs, increase :py:attr:`~.num_hidden_units` and
            :py:attr:`~.output_dimension` to increase the model size and output dimensionality.
            |Default:| ``32``.
        pad_before_transform (bool):
            If ``False``, the output sequence will be smaller than the input sequence due to the
            receptive field of the model. For example, if the input sequence is ``100`` steps long,
            and a model with receptive field ``10`` is used, the output sequence will only be ``100-10+1``
            steps long. For typical use cases, this parameters can be left at the default. |Default:| ``True``.
        hybrid (bool):
            If ``True``, the model will be trained using both the time-contrastive and the selected
            behavior-constrastive loss functions. |Default:| ``False``.
        optimizer_kwargs (dict):
            Additional optimization parameters. These have the form ``((key, value), (key, value))`` and
            are passed to the PyTorch optimizer specified through the ``optimizer`` argument. Refer to the
            optimizer documentation in :py:mod:`torch.optim` for further information on how to format the
            arguments.
            |Default:| ``(('betas', (0.9, 0.999)), ('eps', 1e-08), ('weight_decay', 0), ('amsgrad', False))``

    Example:

        >>> import cebra
        >>> cebra_model = cebra.CEBRA(model_architecture='offset10-model',
        ...                           batch_size=512,
        ...                           learning_rate=3e-4,
        ...                           temperature=1,
        ...                           output_dimension=3,
        ...                           max_iterations=10,
        ...                           distance='cosine',
        ...                           conditional='time_delta',
        ...                           device='cuda_if_available',
        ...                           verbose=True,
        ...                           time_offsets = 10)

    """

    @classmethod
    def supported_model_architectures(self, pattern: str = "*") -> List[str]:
        """Get a list of supported model architectures.

        These values can be directly passed to the ``model_architecture``
        argument.

        Args:
            pattern: Optional pattern for filtering the architecture list.
                Should use the :py:mod:`fnmatch` patterns.

        Returns:
            A list of all supported model architectures.

        Note:
            It is always possible to use the additional model architectures
            given by :py:func:`cebra.models.get_options` via the CEBRA pytorch
            API.
        """

        # TODO(stes): Check directly via the classes (but without initializing)
        return [
            option for option in cebra.models.get_options(pattern)
            if ("subsample" not in option and "resample" not in option and
                "supervised" not in option)
        ]

    def __init__(
        self,
        model_architecture: str = "offset1-model",
        device: str = "cuda_if_available",
        criterion: str = "infonce",
        distance: str = "cosine",
        conditional: str = None,
        temperature: float = 1.0,
        temperature_mode: Literal["constant", "auto"] = "constant",
        min_temperature: Optional[float] = 0.1,
        time_offsets: int = 1,
        delta: float = None,
        max_iterations: int = 10000,
        max_adapt_iterations: int = 500,
        batch_size: int = None,
        learning_rate: float = 3e-4,
        optimizer: str = "adam",
        output_dimension: int = 8,
        verbose: bool = False,
        num_hidden_units: int = 32,
        pad_before_transform: bool = True,
        hybrid: bool = False,
        optimizer_kwargs: Tuple[Tuple[str, object], ...] = (
            ("betas", (0.9, 0.999)),
            ("eps", 1e-08),
            ("weight_decay", 0),
            ("amsgrad", False),
        ),
    ):
        self.__dict__.update(locals())

        if self.optimizer != "adam":
            raise NotImplementedError(
                "Only adam optimizer supported currently.")

    @property
    def num_sessions(self) -> Optional[int]:
        """The number of sessions.

        Note:
            It will be None for single session.
        """
        return self.num_sessions_

    @property
    def state_dict_(self) -> dict:
        return self.solver_.state_dict()

    def _prepare_data(
        self, X: Union[List[Iterable], Iterable], y
    ) -> Union[cebra_sklearn_dataset.SklearnDataset,
               cebra.data.DatasetCollection]:
        """Create dataset from data and labels

        Note:
            The method handles both single- and multi-session datasets. The difference will
            be in the data and labels format.

            For now, multisession does not handle more than one set of indexes.

        Args:
            X: Either a 2D data matrix (single-session) or a list of 2D data matrix (multi-session).
            y: An arbitrary amount of continuous indices passed as either 2D matrices (single-session)
                or lists of 2D matrices (multi-session). For single-session only, up to one discrete
                index passed as a 1D array. Each index has to match the length of the corresponding
                data in ``X``.

        Returns:
            dataset (first return) is either single session dataset, or multisession dataset.
            is_multisession (second return) is a boolean indicating the type of model, either single- or multi-session.
        """

        def _are_sessions_equal(X, y):
            """Check if data and labels have the same number of sessions for all sets of labels."""
            return np.array([len(X) == len(y_i) for y_i in y]).all()

        def _get_dataset(X: Iterable, y: tuple):
            """Create single-session dataset from data and labels.

            Args:
                X: A 2D matrix data.
                y: A tuple containing the sets of indices.

            Returns:
                A single-session dataset consisting of X as input data and y as labels.
            """
            X = sklearn_utils.check_input_array(X,
                                                min_samples=len(self.offset_))

            return cebra_sklearn_dataset.SklearnDataset(X,
                                                        y,
                                                        device=self.device_)

        def _get_dataset_multi(X: List[Iterable], y: List[Iterable]):
            """Create a multi-session dataset iteratively.

            Note:
                Data and indices need to be the same number of samples and same number of sessions.

            Args:
                X: A list of 2D data matrices, each corresponding to a session.
                y: A list of tuple, each corresponding to the different indices of the dataset.

            Returns:
                A multisession dataset.
            """
            if y is None or len(y) == 0:
                raise RuntimeError(
                    "No label: labels are needed for alignment in the multisession implementation."
                )

            # TODO(celia): to make it work for multiple set of index. For now, y should be a tuple of one list only
            if isinstance(y, tuple) and len(y) > 1:
                raise NotImplementedError(
                    f"Support for multiple set of index is not implemented in multissesion training, "
                    f"got {len(y)} sets of indexes.")

            if not _are_sessions_equal(X, y):
                raise ValueError(
                    f"Invalid number of sessions: number of sessions in X and y need to match, "
                    f"got X:{len(X)} and y:{[len(y_i) for y_i in y]}.")

            for session in range(len(X)):
                for labels in range(len(y)):
                    if len(X[session]) != len(y[labels][session]):
                        raise ValueError(
                            f"Invalid number of samples in session {session} for set of label {labels}: "
                            f"X has {len(X[session])}, while y has {len(y[labels][session])}"
                        )

            return cebra.data.DatasetCollection(*[
                _get_dataset(X_session, (y_session,))
                for X_session, y_session in zip(X, *y)
            ])

        # The dataset is a multisession dataset if data consists of a list of iterables
        if isinstance(X, list) and isinstance(X[0], Iterable) and len(
                X[0].shape) == 2:
            is_multisession = True
            dataset = _get_dataset_multi(X, y)
        else:
            if not _are_sessions_equal(X, y):
                raise ValueError(
                    f"Invalid number of samples or labels sessions: provide one session for single-session training, "
                    f"and make sure the number of samples in X and y need match, "
                    f"got {len(X)} and {[len(y_i) for y_i in y]}.")
            is_multisession = False
            dataset = _get_dataset(X, y)
        return dataset, is_multisession

    def _compute_offset(self) -> cebra.data.Offset:
        """Compute the offset from a mock cebra model."""
        # TODO(stes): workaround to get the offset - should be removed again
        #            once a better solution is implemented in cebra.models
        return cebra.models.init(self.model_architecture,
                                 num_neurons=1,
                                 num_units=2,
                                 num_output=1).get_offset()

    def _prepare_loader(self, dataset: cebra.data.Dataset, max_iterations: int,
                        is_multisession: bool):
        """Prepare the data loader based on the dataset properties.

        Args:
            dataset: A dataset, either single or multisession.
            is_multisession: A boolean that indicates if the dataset is a single or multisession dataset.

        Returns:
            A data loader.
        """
        return _init_loader(
            is_cont=dataset.continuous_index is not None,
            is_disc=dataset.discrete_index is not None,
            is_hybrid=self.hybrid,
            is_full=self.batch_size is None,
            is_multi=is_multisession,
            shared_kwargs=dict(
                dataset=dataset,
                batch_size=self.batch_size,
                num_steps=max_iterations,
            ),
            extra_kwargs=dict(
                time_offsets=self.time_offsets,
                conditional=self.conditional,
                delta=self.delta,
            ),
        )

    def _prepare_criterion(self):
        """Prepare criterion based on :py:attr:`self.criterion`.

        Returns:
            The required criterion for the model.
        """
        if self.criterion == "infonce":
            if self.temperature_mode == "auto":
                if self.distance == "cosine":
                    return cebra.models.LearnableCosineInfoNCE(
                        temperature=self.temperature,
                        min_temperature=self.min_temperature,
                    )
                elif self.distance == "euclidean":
                    return cebra.models.LearnableEuclideanInfoNCE(
                        temperature=self.temperature,
                        min_temperature=self.min_temperature,
                    )
            elif self.temperature_mode == "constant":
                if self.distance == "cosine":
                    return cebra.models.FixedCosineInfoNCE(
                        temperature=self.temperature,)
                elif self.distance == "euclidean":
                    return cebra.models.FixedEuclideanInfoNCE(
                        temperature=self.temperature,)

        raise ValueError(f"Unknown similarity measure '{self.distance}' for "
                         f"criterion '{self.criterion}'.")

    def _prepare_model(self, dataset: cebra.data.Dataset,
                       is_multisession: bool):
        """Create the model based on the dataset properties.

        Args:
            dataset: Either a single or multi session dataset for which the model is created.
            is_multisession: A boolean that indicates if the dataset is a single or multisession dataset.

        Returns:
            A model or a list of models depending on the type of session (``is_multisession``).
        """
        if is_multisession:
            model = nn.ModuleList([
                cebra.models.init(
                    self.model_architecture,
                    num_neurons=dataset.input_dimension,
                    num_units=self.num_hidden_units,
                    num_output=self.output_dimension,
                ) for dataset in dataset.iter_sessions()
            ]).to(self.device_)
        else:
            model = cebra.models.init(
                self.model_architecture,
                num_neurons=dataset.input_dimension,
                num_units=self.num_hidden_units,
                num_output=self.output_dimension,
            ).to(self.device_)

        return model

    def _configure_for_all(
        self,
        dataset: cebra.data.Dataset,
        model: Union[cebra.models.Model, nn.ModuleList],
        is_multisession: bool,
    ):
        """Configure the dataset depending on its type.

        Args:
            dataset: Either a single or multi session dataset.
            model: model or list of models.
            is_multisession: indicates if the dataset is a multisession or single session dataset.
        """

        if is_multisession:
            for n, d in enumerate(dataset.iter_sessions()):
                if not isinstance(model[n],
                                  cebra.models.ConvolutionalModelMixin):
                    if len(model[n].get_offset()) > 1:
                        raise ValueError(
                            f"It is not yet supported to run non-convolutional models with "
                            f"receptive fields/offsets larger than 1 via the sklearn API. "
                            f"Please use a different model, or revert to the pytorch "
                            f"API for training.")

                d.configure_for(model[n])
        else:
            if not isinstance(model, cebra.models.ConvolutionalModelMixin):
                if len(model.get_offset()) > 1:
                    raise ValueError(
                        f"It is not yet supported to run non-convolutional models with "
                        f"receptive fields/offsets larger than 1 via the sklearn API. "
                        f"Please use a different model, or revert to the pytorch "
                        f"API for training.")

            dataset.configure_for(model)

    def _select_model(self, X: Union[npt.NDArray, torch.Tensor],
                      session_id: int):
        # Choose the model and get its corresponding offset
        if self.num_sessions is not None:  # multisession implementation
            if session_id is None:
                raise RuntimeError(
                    "No session_id provided: multisession model requires a session_id to choose the model corresponding to your data shape."
                )
            if session_id >= self.num_sessions or session_id < 0:
                raise RuntimeError(
                    f"Invalid session_id {session_id}: session_id for the current multisession model must be between 0 and {self.num_sessions-1}."
                )
            if self.n_features_[session_id] != X.shape[1]:
                raise ValueError(
                    f"Invalid input shape: model for session {session_id} requires an input of shape"
                    f"(n_samples, {self.n_features_[session_id]}), got (n_samples, {X.shape[1]})."
                )

            model = self.model_[session_id]
            model.to(self.device_)
        else:  # single session
            if session_id is not None and session_id > 0:
                raise RuntimeError(
                    f"Invalid session_id {session_id}: single session models only takes an optional null session_id."
                )
            model = self.model_

        offset = model.get_offset()
        return model, offset

    def _check_labels_types(self, y: tuple, session_id: Optional[int] = None):
        """Check that the input labels are compatible with the labels used to fit the model.

        Note:
            The input labels to compare correspond to a single session. For multisession model,
            the session ID must be provided to select the required session. Then, we check that
            the dtype and number of features of the labels is similar to the ones used to fit
            the model.

        Args:
            y: A tuple containing the indexes to compare. Input labels can only correspond to a
                single session.
            session_id: The session ID, an :py:class:`int` between 0 and :py:attr:`num_sessions` for
                multisession, set to ``None`` for single session.

        """
        n_idx = len(y)
        # Check that same number of index
        if len(self.label_types_) != n_idx:
            raise ValueError(
                f"Number of index invalid: labels must have the same number of index as for fitting,"
                f"expects {len(self.label_types_)}, got {n_idx} idx.")

        for i in range(len(self.label_types_)):  # for each index
            if self.num_sessions is None:
                label_types_idx = self.label_types_[i]
            else:
                label_types_idx = self.label_types_[i][session_id]

            if (len(label_types_idx[1]) > 1 and len(y[i].shape)
                    > 1):  # is there more than one feature in the index
                if label_types_idx[1][1] != y[i].shape[1]:
                    raise ValueError(
                        f"Labels invalid: must have the same number of features as the ones used for fitting,"
                        f"expects {label_types_idx[1]}, got {y[i].shape}.")

            if label_types_idx[0] != y[i].dtype:
                raise ValueError(
                    f"Labels invalid: must have the same type of features as the ones used for fitting,"
                    f"expects {label_types_idx[0]}, got {y[i].dtype}.")

    def _prepare_fit(
        self,
        X: Union[npt.NDArray, torch.Tensor],
        *y,
    ) -> Tuple[cebra.solver.Solver, cebra.models.Model, cebra.data.Loader,
               bool]:
        """Initialize the loader, model and solver to fit CEBRA to the provided data.

        This method will be called when the model is fitted for the first time (i.e., upon first
        call of :py:meth:`cebra.CEBRA.fit`).

        Args:
            X: A 2D data matrix.
            y: An arbitrary amount of continuous indices passed as 2D matrices, and up to one
                discrete index passed as a 1D array. Each index has to match the length of ``X``.

        Returns:
            The solver (first return), model (second return), loader (third return), and a bool indicating if the
                training is performed on multiple sessions (fourth return) initialized for ``X``.

        """
        self.device_ = sklearn_utils.check_device(self.device)
        self.offset_ = self._compute_offset()
        dataset, is_multisession = self._prepare_data(X, y)

        loader, solver_name = self._prepare_loader(
            dataset,
            max_iterations=self.max_iterations,
            is_multisession=is_multisession)
        model = self._prepare_model(dataset, is_multisession)

        self._configure_for_all(dataset, model, is_multisession)

        criterion = self._prepare_criterion()
        criterion.to(self.device_)
        optimizer = torch.optim.Adam(
            itertools.chain(model.parameters(), criterion.parameters()),
            lr=self.learning_rate,
            **dict(self.optimizer_kwargs),
        )

        solver = cebra.solver.init(
            solver_name,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            tqdm_on=self.verbose,
        )
        solver.to(self.device_)
        self.solver_name_ = solver_name

        self.label_types_ = ([[(y_session.dtype, y_session.shape)
                               for y_session in y_index]
                              for y_index in y] if is_multisession else
                             [(y_.dtype, y_.shape) for y_ in y])

        # NOTE(stes): The model is explicitly passed because the solver can freely modify
        # the original model, e.g. in cebra.models.MultiobjectiveModel
        return solver, model, loader, is_multisession

    def _adapt_model(
        self, X: Union[npt.NDArray, torch.Tensor], *y
    ) -> Tuple[cebra.solver.Solver, cebra.models.Model, cebra.data.Loader,
               bool]:
        """Adapt the loader, model and solver to the new data.

        This method will be called when the parameter ``adapt`` from :py:meth:`cebra.CEBRA.fit` is set to
        ``True`` and the model was already fitted to a different dataset. It re-initializes the
        loader, so that the model is refitted over :py:attr:`CEBRA.max_adapt_iterations` iterations,
        the first layer of the model, while keeping the weights for the other layers and the solver
        so that it considers the new model

        Args:
            X: A 2D data matrix.
            y: An arbitrary amount of continuous indices passed as 2D matrices, and up to one
                discrete index passed as a 1D array. Each index has to match the length of ``X``.

        Returns:
            The solver (first return), model (second return), loader (third return), and a bool indicating if the
                training is performed on multiple sessions (fourth return) initialized for ``X``.

        """

        dataset, is_multisession = self._prepare_data(X, y)

        if is_multisession or isinstance(self.model_, nn.ModuleList):
            raise NotImplementedError(
                "The adapt option with a multisession training is not handled. Please use adapt=True for single-trained estimators only."
            )

        # Re-initialize the loader to iterate over max_adapt_iterations
        loader, solver_name = self._prepare_loader(
            dataset,
            max_iterations=self.max_adapt_iterations,
            is_multisession=is_multisession,
        )

        adapt_model = self._prepare_model(dataset, is_multisession)

        # Resize the first layer of the model
        pretrained_dict = self.model_.state_dict()
        pretrained_params = list(pretrained_dict.keys())
        trained_params = pretrained_params[2:]
        untrained_params = pretrained_params[:
                                             2]  # the weight and bias of the 1st layer

        adapted_dict = {
            k: v for k, v in pretrained_dict.items() if k in trained_params
        }  # all layers but first (weight + bias) take the pre-trained parameters
        for k in untrained_params:  # first layer stays the same
            adapted_dict[k] = adapt_model.state_dict()[k]

        adapt_model.load_state_dict(adapted_dict)

        for name, param in adapt_model.named_parameters():
            if name in trained_params:
                param.requires_grad = False

        adapt_model.to(self.device_)

        self._configure_for_all(dataset, adapt_model, is_multisession)

        criterion = self._prepare_criterion()
        criterion.to(self.device_)

        optimizer = torch.optim.Adam(
            list(adapt_model.parameters()) + list(criterion.parameters()),
            lr=self.learning_rate,
            **dict(self.optimizer_kwargs),
        )

        solver = cebra.solver.init(
            solver_name,
            model=adapt_model,
            criterion=criterion,
            optimizer=optimizer,
            tqdm_on=self.verbose,
        )
        solver.to(self.device_)

        return solver, adapt_model, loader, is_multisession

    def _partial_fit(
        self,
        solver: cebra.solver.Solver,
        model: cebra.models.Model,
        loader: cebra.data.Loader,
        is_multisession: bool,
        callback: Callable[[int, cebra.solver.Solver], None] = None,
        callback_frequency: int = None,
    ) -> "CEBRA":
        """Fit the adapted estimator to the given dataset and given solver, model and loader.

        Args:
            solver: The solver to use to training the estimator.
            model: The model to use to train the estimator.
            loader: The loader to use to train the estimator
            callback: If a function is passed here with signature ``callback(num_steps, solver)``,
                the function will be regularly called at the specified ``callback_frequency``.
            callback_frequency: Specify the number of iterations that need to pass before triggering
                the specified ``callback``.

        Returns:
            ``self``, to allow chaining of operations.

        """
        if callback_frequency is not None:
            if callback is None:
                raise ValueError(
                    "callback_frequency requires to specify a callback.")

        model.train()

        solver.fit(
            loader,
            valid_loader=None,
            save_frequency=callback_frequency,
            valid_frequency=None,
            decode=False,
            logdir=None,
            save_hook=callback,
        )

        # Save variables of interest as semi-private attributes
        self.model_ = model
        self.n_features_ = ([
            loader.dataset.get_input_dimension(session_id)
            for session_id in range(loader.dataset.num_sessions)
        ] if is_multisession else loader.dataset.input_dimension)
        self.solver_ = solver
        self.n_features_in_ = ([model[n].num_input for n in range(len(model))]
                               if is_multisession else model.num_input)
        self.num_sessions_ = loader.dataset.num_sessions if is_multisession else None

        return self

    def partial_fit(
        self,
        X: Union[npt.NDArray, torch.Tensor],
        *y,
        callback: Callable[[int, cebra.solver.Solver], None] = None,
        callback_frequency: int = None,
    ) -> "CEBRA":
        """Partially fit the estimator to the given dataset.

        It is useful when the whole dataset is too big to fit in memory at once.

        Note:
            The method allows to perform incremental learning from batch instance. Using :py:meth:`partial_fit`
            on a partially fitted model will iteratively continue training, over the partially fitted parameters.
            To reset the parameters at each new fitting, :py:meth:`fit` must be used.

        Args:
            X: A 2D data matrix.
            y: An arbitrary amount of continuous indices passed as 2D matrices, and up to one
                discrete index passed as a 1D array. Each index has to match the length of ``X``.
            callback: If a function is passed here with signature ``callback(num_steps, solver)``,
                the function will be regularly called at the specified ``callback_frequency``.
            callback_frequency: Specify the number of iterations that need to pass before triggering
                the specified ``callback``.

        Returns:
            ``self``, to allow chaining of operations.

        Example:

            >>> import cebra
            >>> import numpy as np
            >>> dataset =  np.random.uniform(0, 1, (1000, 30))
            >>> cebra_model = cebra.CEBRA(max_iterations=10)
            >>> cebra_model.partial_fit(dataset)
            CEBRA(max_iterations=10)

        """
        if not hasattr(self, "state_") or self.state_ is None:
            self.state_ = self._prepare_fit(X, *y)
        self._partial_fit(*self.state_,
                          callback=callback,
                          callback_frequency=callback_frequency)
        return self

    def _adapt_fit(
        self,
        X: Union[npt.NDArray, torch.Tensor],
        *y,
        callback: Callable[[int, cebra.solver.Solver], None] = None,
        callback_frequency: int = None,
    ) -> "CEBRA":
        """Fit the adapted estimator to the given dataset.

        Args:
            X: A 2D data matrix.
            y: An arbitrary amount of continuous indices passed as 2D matrices, and up to one
                discrete index passed as a 1D array. Each index has to match the length of ``X``.
            callback: If a function is passed here with signature ``callback(num_steps, solver)``,
                the function will be regularly called at the specified ``callback_frequency``.
            callback_frequency: Specify the number of iterations that need to pass before triggering
                the specified ``callback``.

        Returns:
            ``self``, to allow chaining of operations.

        """
        self.state_ = self._adapt_model(X, *y)
        self._partial_fit(*self.state_,
                          callback=callback,
                          callback_frequency=callback_frequency)
        return self

    def fit(
        self,
        X: Union[List[Iterable], Iterable],
        *y,
        adapt: bool = False,
        callback: Callable[[int, cebra.solver.Solver], None] = None,
        callback_frequency: int = None,
    ) -> "CEBRA":
        """Fit the estimator to the given dataset, either by initializing a new model or
        by adapting the existing trained model.

        Note:
            Re-fitting a fitted model with :py:meth:`fit` will reset the parameters and number of iterations.
            To continue fitting from the previous fit, :py:meth:`partial_fit` must be used.

        Tip:
            We recommend saving the model, using :py:meth:`cebra.CEBRA.save`, before adapting it to a different
            dataset (setting ``adapt=True``) as the adapted model will replace the previous model in ``cebra_model.state_dict_``.

        Args:
            X: A 2D data matrix.
            y: An arbitrary amount of continuous indices passed as 2D matrices, and up to one
                discrete index passed as a 1D array. Each index has to match the length of ``X``.
            adapt: If True, the estimator will be adapted to the given data. This parameter is of
                use only once the estimator has been fitted at least once (i.e., :py:meth:`cebra.CEBRA.fit`
                has been called already). Note that it can be used on a fitted model that was saved
                and reloaded, using :py:meth:`cebra.CEBRA.save` and :py:meth:`cebra.CEBRA.load`. To adapt the
                model, the first layer of the model is reset so that it corresponds to the new features dimension.
                The parameters for all other layers are fixed and the first reinitialized layer is re-trained for
                :py:attr:`cebra.CEBRA.max_adapt_iterations`.
            callback: If a function is passed here with signature ``callback(num_steps, solver)``,
                the function will be regularly called at the specified ``callback_frequency``.
            callback_frequency: Specify the number of iterations that need to pass before triggering
                the specified ``callback``,

        Returns:
            ``self``, to allow chaining of operations.

        Example:

            >>> import cebra
            >>> import numpy as np
            >>> import tempfile
            >>> from pathlib import Path
            >>> tmp_file = Path(tempfile.gettempdir(), 'cebra.pt')
            >>> dataset =  np.random.uniform(0, 1, (1000, 20))
            >>> dataset2 =  np.random.uniform(0, 1, (1000, 40))
            >>> cebra_model = cebra.CEBRA(max_iterations=10)
            >>> cebra_model.fit(dataset)
            CEBRA(max_iterations=10)
            >>> cebra_model.save(tmp_file)
            >>> cebra_model.fit(dataset2, adapt=True)
            CEBRA(max_iterations=10)
            >>> tmp_file.unlink()
        """
        if adapt and sklearn_utils.check_fitted(self):
            self._adapt_fit(X,
                            *y,
                            callback=callback,
                            callback_frequency=callback_frequency)
        else:
            self.partial_fit(X,
                             *y,
                             callback=callback,
                             callback_frequency=callback_frequency)
            del self.state_

        return self

    def transform(self,
                  X: Union[npt.NDArray, torch.Tensor],
                  session_id: Optional[int] = None) -> npt.NDArray:
        """Transform an input sequence and return the embedding.

        Args:
            X: A numpy array or torch tensor of size ``time x dimension``.
            session_id: The session ID, an :py:class:`int` between 0 and :py:attr:`num_sessions` for
                multisession, set to ``None`` for single session.

        Returns:
            A :py:func:`numpy.array` of size ``time x output_dimension``.

        Example:

            >>> import cebra
            >>> import numpy as np
            >>> dataset =  np.random.uniform(0, 1, (1000, 30))
            >>> cebra_model = cebra.CEBRA(max_iterations=10)
            >>> cebra_model.fit(dataset)
            CEBRA(max_iterations=10)
            >>> embedding = cebra_model.transform(dataset)

        """

        sklearn_utils_validation.check_is_fitted(self, "n_features_")
        model, offset = self._select_model(X, session_id)

        # Input validation
        X = sklearn_utils.check_input_array(X, min_samples=len(self.offset_))
        input_dtype = X.dtype

        with torch.no_grad():
            model.eval()

            if self.pad_before_transform:
                X = np.pad(X, ((offset.left, offset.right - 1), (0, 0)),
                           mode="edge")
            X = torch.from_numpy(X).float().to(self.device_)

            if isinstance(model, cebra.models.ConvolutionalModelMixin):
                # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
                X = X.transpose(1, 0).unsqueeze(0)
                output = model(X).cpu().numpy().squeeze(0).transpose(1, 0)
            else:
                # Standard evaluation, (T, C, dt)
                output = model(X).cpu().numpy()

        if input_dtype == "float64":
            return output.astype(input_dtype)

        return output

    def fit_transform(
        self,
        X: Union[npt.NDArray, torch.Tensor],
        *y,
        adapt: bool = False,
        callback: Callable[[int, cebra.solver.Solver], None] = None,
        callback_frequency: int = None,
    ) -> npt.NDArray:
        """Composition of :py:meth:`fit` and :py:meth:`transform`.

        Args:
            X: A 2D data matrix.
            y: An arbitrary amount of continuous indices passed as 2D matrices, and up to one
                discrete index passed as a 1D array. Each index has to match the length of ``X``.
            adapt: If True, the estimator will be adapted to the given data. This parameter is of
                use only once the estimator has been fitted at least once (i.e., :py:meth:`cebra.CEBRA.fit`
                has been called already). Note that it can be used on a fitted model that was saved
                and reloaded, using :py:meth:`cebra.CEBRA.save` and :py:meth:`cebra.CEBRA.load`.
            callback: If a function is passed here with signature ``callback(num_steps, solver)``,
                the function will be regularly called at the specified ``callback_frequency``.
            callback_frequency: Specify the number of iterations that need to pass before triggering
                the specified ``callback``,

        Returns:
            A :py:func:`numpy.array` of size ``time x output_dimension``.

        Example:

            >>> import cebra
            >>> import numpy as np
            >>> dataset =  np.random.uniform(0, 1, (1000, 30))
            >>> cebra_model = cebra.CEBRA(max_iterations=10)
            >>> embedding = cebra_model.fit_transform(dataset)

        """
        self.fit(X,
                 *y,
                 adapt=adapt,
                 callback=callback,
                 callback_frequency=callback_frequency)
        return self.transform(X)

    def _more_tags(self):
        # NOTE(stes): This tag is needed as seeding is not fully implemented in the
        # current version of CEBRA.
        return {"non_deterministic": True}

    def _get_state(self):
        cebra_dict = self.__dict__
        state = {
            'label_types_': cebra_dict['label_types_'],
            'device_': cebra_dict['device_'],
            'n_features_': cebra_dict['n_features_'],
            'n_features_in_': cebra_dict['n_features_in_'],
            'num_sessions_': cebra_dict['num_sessions_'],
            'offset_': cebra_dict['offset_'],
            'solver_name_': cebra_dict['solver_name_'],
        }
        return state

    def save(self,
             filename: str,
             backend: Literal["torch", "sklearn"] = "sklearn"):
        """Save the model to disk.

        Args:
            filename: The path to the file in which to save the trained model.
            backend: A string identifying the used backend. Default is "sklearn".

        Returns:
            The saved model checkpoint.

        Note:
            The save/load functionalities may change in a future version.

            File Format:
                The saved model checkpoint file format depends on the specified backend.

                "sklearn" backend (default):
                    The model is saved in a PyTorch-compatible format using `torch.save`. The saved checkpoint
                    is a dictionary containing the following elements:
                    - 'args': A dictionary of parameters used to initialize the CEBRA model.
                    - 'state': The state of the CEBRA model, which includes various internal attributes.
                    - 'state_dict': The state dictionary of the underlying solver used by CEBRA.
                    - 'metadata': Additional metadata about the saved model, including the backend used and the version of CEBRA PyTorch, NumPy and scikit-learn.

                "torch" backend:
                    The model is directly saved using `torch.save` with no additional information. The saved
                    file contains the entire CEBRA model state.


        Example:

            >>> import cebra
            >>> import numpy as np
            >>> import tempfile
            >>> from pathlib import Path
            >>> tmp_file = Path(tempfile.gettempdir(), 'test.jl')
            >>> dataset =  np.random.uniform(0, 1, (1000, 30))
            >>> cebra_model = cebra.CEBRA(max_iterations=10)
            >>> cebra_model.fit(dataset)
            CEBRA(max_iterations=10)
            >>> cebra_model.save(tmp_file)
            >>> tmp_file.unlink()

        """
        if sklearn_utils.check_fitted(self):
            if backend == "torch":
                checkpoint = torch.save(self, filename)

            elif backend == "sklearn":
                checkpoint = torch.save(
                    {
                        'args': self.get_params(),
                        'state': self._get_state(),
                        'state_dict': self.solver_.state_dict(),
                        'metadata': {
                            'backend':
                                backend,
                            'cebra_version':
                                cebra.__version__,
                            'torch_version':
                                torch.__version__,
                            'numpy_version':
                                np.__version__,
                            'sklearn_version':
                                pkg_resources.get_distribution("scikit-learn"
                                                              ).version
                        }
                    }, filename)
            else:
                raise NotImplementedError(f"Unsupported backend: {backend}")
        else:
            raise ValueError("CEBRA object is not fitted. "
                             "Saving a non-fitted model is not supported.")
        return checkpoint

    @classmethod
    def load(cls,
             filename: str,
             backend: Literal["auto", "sklearn", "torch"] = "auto",
             **kwargs) -> "CEBRA":
        """Load a model from disk.

        Args:
            filename: The path to the file in which to save the trained model.
            backend: A string identifying the used backend.
            kwargs: Optional keyword arguments passed directly to the loader.

        Return:
            The model to load.

        Note:
            Experimental functionality. Do not expect the save/load functionalities to be
            backward compatible yet between CEBRA versions!

            For information about the file format please refer to :py:meth:`cebra.CEBRA.save`.

        Example:

            >>> import cebra
            >>> import numpy as np
            >>> import tempfile
            >>> from pathlib import Path
            >>> tmp_file = Path(tempfile.gettempdir(), 'cebra.pt')
            >>> dataset =  np.random.uniform(0, 1, (1000, 20))
            >>> cebra_model = cebra.CEBRA(max_iterations=10)
            >>> cebra_model.fit(dataset)
            CEBRA(max_iterations=10)
            >>> cebra_model.save(tmp_file)
            >>> loaded_model = cebra.CEBRA.load(tmp_file)
            >>> embedding = loaded_model.transform(dataset)
            >>> tmp_file.unlink()

        """

        supported_backends = ["auto", "sklearn", "torch"]
        if backend not in supported_backends:
            raise NotImplementedError(
                f"Unsupported backend: '{backend}'. Supported backends are: {', '.join(supported_backends)}"
            )

        checkpoint = torch.load(filename, **kwargs)

        if backend == "auto":
            backend = "sklearn" if isinstance(checkpoint, dict) else "torch"

        if isinstance(checkpoint, dict) and backend == "torch":
            raise RuntimeError(
                f"Cannot use 'torch' backend with a dictionary-based checkpoint. "
                f"Please try a different backend.")
        if not isinstance(checkpoint, dict) and backend == "sklearn":
            raise RuntimeError(
                f"Cannot use 'sklearn' backend a non dictionary-based checkpoint. "
                f"Please try a different backend.")

        if backend == "sklearn":
            cebra_ = _load_cebra_with_sklearn_backend(checkpoint)
        else:
            cebra_ = _check_type_checkpoint(checkpoint)

        return cebra_

    def to(self, device: Union[str, torch.device]):
        """Moves the cebra model to the specified device.

        Args:
            device: The device to move the cebra model to. This can be a string representing
                    the device ('cpu','cuda', cuda:device_id, or 'mps') or a torch.device object.

        Returns:
            The cebra model instance.

        Example:

            >>> import cebra
            >>> import numpy as np
            >>> dataset =  np.random.uniform(0, 1, (1000, 30))
            >>> cebra_model = cebra.CEBRA(max_iterations=10, device = "cuda_if_available")
            >>> cebra_model.fit(dataset)
            CEBRA(max_iterations=10)
            >>> cebra_model = cebra_model.to("cpu")
        """

        if not isinstance(device, (str, torch.device)):
            raise TypeError(
                "The 'device' parameter must be a string or torch.device object."
            )

        if isinstance(device, str):
            if (not device == 'cpu') and (not device.startswith('cuda')) and (
                    not device == 'mps'):
                raise ValueError(
                    "The 'device' parameter must be a valid device string or device object."
                )

        elif isinstance(device, torch.device):
            if (not device.type == 'cpu') and (
                    not device.type.startswith('cuda')) and (not device
                                                             == 'mps'):
                raise ValueError(
                    "The 'device' parameter must be a valid device string or device object."
                )
            device = device.type

        if hasattr(self, "device_"):
            self.device_ = device

        self.device = device
        self.solver_.model.to(device)

        return self
