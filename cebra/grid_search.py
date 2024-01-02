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
"""Utilities for performing a grid search across CEBRA models."""

import copy
import pathlib
import pickle
from typing import Iterable, List, Literal, Optional, Tuple

import matplotlib.axes
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.utils.validation as sklearn_utils_validation
import torch

import cebra.integrations.matplotlib as cebra_matplotlib
import cebra.integrations.sklearn.cebra as cebra_sklearn_cebra


class GridSearch:
    """Define and run a grid search on the CEBRA hyperparameters.

    Note:
        We recommend that you use that grid-search implementation for rather small and simple grid-search.

    Depending on the usage, one needs to optimize some parameters used in the CEBRA model, e.g., the
    :py:attr:`~.CEBRA.temperature`, the :py:attr:`~.CEBRA.batch_size`, the :py:attr:`~.CEBRA.learning_rate`.
    A grid-search on that set of parameters consists in finding the best combination of values
    for those parameters. For that, models with different combinations of parameters are trained and the
    parameters used to get the best performing model are considered to be the optimal parameters. One can also
    define the fixed parameters, which will stay constant
    from one model to the other, e.g., the :py:attr:`~.CEBRA.max_iterations` or the :py:attr:`~.CEBRA.verbose`.

    The class also allows to iterate over multiple datasets and combinations of auxiliary variables.
    """

    def generate_models(self, params: dict) -> Tuple[dict, List[dict]]:
        """Generate the models to compare, based on fixed and variable CEBRA parameters.

        Args:
            params: Dict of parameter values provided by the user, either as a single value, for
                fixed hyperparameter values, or with a list of values for hyperparameters to optimize.
                If the value is a list of a single element, the hyperparameter is considered as fixed.

        Returns:
            A dict of (unfitted) models (first returns) and a list of dict of the parameters for each
            model (second returns).

        Example:

            >>> import cebra.grid_search
            >>> # 1. Define the parameters for the models
            >>> params_grid = dict(
            ...     output_dimension = [3, 16],
            ...     learning_rate = [0.001],
            ...     time_offsets = 5,
            ...     max_iterations = 10,
            ...     verbose = False)
            >>> # 2. Define the grid search and generate the models
            >>> grid_search = cebra.grid_search.GridSearch()
            >>> models, parameter_grid = grid_search.generate_models(params=params_grid)

        """
        # determine if each param is fixed or to optimize
        fixed_params, variable_params = {}, {}
        for key in list(params.keys()):
            if not isinstance(params[key], list):  # fixed parameters
                if (isinstance(params[key], dict) or
                        isinstance(params[key], tuple) or
                        isinstance(params[key], torch.Tensor) or
                        isinstance(params[key], np.ndarray)):
                    raise ValueError(
                        f"Invalid list of variable parameters, provide a list of values, got {type(params[key])}."
                    )
                fixed_params[key] = params[key]
            elif len(params[key]) < 2:  # fixed parameters
                if not isinstance(params[key][0], Iterable) or isinstance(
                        params[key][0], str):
                    fixed_params[key] = params[key][0]
            elif not any(
                    isinstance(params[key][i], Iterable) or
                    isinstance(params[key][0], str)
                    for i in range(len(params[key]))):  # variable parameters
                variable_params[key] = params[key]
            else:
                raise ValueError(
                    f"Invalid parameter {params[key]} of type {type(params[key])}."
                )

        parameter_grid = list(
            sklearn.model_selection.ParameterGrid(variable_params))

        models = {}
        for params in parameter_grid:
            params_name = [f"{name}_{value}" for name, value in params.items()]
            model_name = "_".join(map(str, list(params_name)))
            models[model_name] = cebra_sklearn_cebra.CEBRA(
                **params, **fixed_params)

        return models, parameter_grid

    def fit_models(self,
                   datasets: dict,
                   params: dict,
                   models_dir: str = "saved_models") -> "GridSearch":
        """Fit the models to compare in the grid search on the provided datasets.

        Args:
            datasets: A dict of datasets to iterate over. The values in the dict can either be a tuple or an
                iterable structure (``Iterable``, i.e., list, :py:func:`numpy.array`, :py:class:`torch.Tensor`).
                If the value provided for a given dataset is a tuple, contain multiple elements and those elements are all iterable structures (as defined
                before), then the first element is considered to be the ``data`` to fit the CEBRA models (``X``
                to be used in :py:meth:`cebra.CEBRA.fit`) on and the other values to be the auxiliary variables
                be to use in the training process (``y`` s to be used in :py:meth:`cebra.CEBRA.fit`). The models
                are then trained using behavioral contrastive learning (either CEBRA-Behavior or CEBRA-Hybrid).
                If the value provided for a given dataset is a tuple containing a single value and it is an iterable structure (as defined before) or is
                such an iterable structure directly (not a tuple), then the value is considered to be the ``data``
                to fit the CEBRA models on. The models are then trained using temporal contrastive learning
                (CEBRA-Time).
                An example of a valid ``datasets`` value could be:
                ``datasets={"dataset1": neural_data, "dataset2": (neurald_data, continuous_data, discrete_data), "dataset3": (neural_data2, continuous_data2)}``.
            params: Dict of parameter values provided by the user, either as a single value, for
                fixed hyperparameter values, or with a list of values for hyperparameters to optimize.
                If the value is a list of a single element, the hyperparameter is considered as fixed.
            models_dir: The path to the folder in which save the (fitted) models.

        Returns:
            ``self`` for chaining operations.

        Example:

            >>> import cebra.grid_search
            >>> import numpy as np
            >>> neural_data =  np.random.uniform(0, 1, (300, 30))
            >>> # 1. Define the parameters for the models
            >>> params_grid = dict(
            ...     output_dimension = [3, 16],
            ...     learning_rate = [0.001],
            ...     time_offsets = 5,
            ...     max_iterations = 10,
            ...     verbose = False)
            >>> # 2. Fit the models generated from the list of parameters
            >>> grid_search = cebra.grid_search.GridSearch()
            >>> grid_search = grid_search.fit_models(datasets={"neural_data": neural_data},
            ...                                      params=params_grid,
            ...                                      models_dir="grid_search_models")

        """
        if models_dir is not None:
            models_dir = pathlib.Path(models_dir)
            if not pathlib.Path.exists(models_dir):
                pathlib.Path.mkdir(models_dir)
        self.models_dir = models_dir

        # get the list of models to iterate over based on the parameters
        unfitted_models, parameter_grid = self.generate_models(params)

        self.models = {}
        self.parameter_grid = []
        self.models_names = []
        for dataset_name in datasets.keys():
            if isinstance(datasets[dataset_name], tuple):
                for data in datasets[
                        dataset_name]:  # check that data is iterable
                    if not isinstance(data, Iterable):
                        raise ValueError(
                            f"Invalid dataset {dataset_name}, provide either the data or a tuple containing the data and auxiliary variables."
                        )
                X = datasets[dataset_name][0]
                if len(datasets[dataset_name]) > 1:
                    y = datasets[dataset_name][1:]
            elif (isinstance(datasets[dataset_name], np.ndarray) or
                  isinstance(datasets[dataset_name], torch.Tensor) or
                  isinstance(datasets[dataset_name], list)):
                X = datasets[dataset_name]
                y = None
            else:
                ValueError(
                    f"Invalid dataset {dataset_name}, provide either the data or a tuple containing the data and auxiliary variables."
                )

            # fit and save each model
            for model_idx, model_name in enumerate(unfitted_models.keys()):
                model = copy.deepcopy(unfitted_models[model_name])

                if y is None:
                    model.fit(X)
                else:
                    model.fit(X, *y)

                if self.models_dir is not None:
                    model.save(self.models_dir /
                               f"{model_name}_{dataset_name}.pt")

                self.models[f"{model_name}_{dataset_name}"] = model
                self.parameter_grid.append(parameter_grid[model_idx])
                self.models_names.append(f"{model_name}_{dataset_name}")

        with open(models_dir / "parameter_grid.pkl", "wb") as fp:
            pickle.dump(self.parameter_grid, fp)
        with open(models_dir / "models_names.pkl", "wb") as fp:
            pickle.dump(self.models_names, fp)

        return self

    @classmethod
    def load(cls, dir: str) -> Tuple[cebra_sklearn_cebra.CEBRA, List[dict]]:
        """Load the *fitted* models and parameter grid present in ``dir``.

        Note:
            It is recommended to generate the models to iterate over by using :py:meth:`fit_models`, but you can also run
            the following function on a folder containing valid *fitted* models **and** the corresponding parameter grid.

        Args:
            dir: The directory in which the fitted models are saved.

        Returns:
            A dict containing the fitted models (first returns) and a list of dict containing the parameters
            used for each model present in the ``dir`` (second returns).

        Example:

            >>> import cebra.grid_search
            >>> models, parameter_grid = cebra.grid_search.GridSearch().load(dir="grid_search_models")

        """
        dir = pathlib.Path(dir)
        if not pathlib.Path.exists(dir):
            raise ValueError(
                f"Invalid model directory, provide a directory that exists and contains the models to load, got {dir}."
            )

        cosine_files = []
        euclidean_files = []
        parameter_grid = None
        models_names = None
        for filename in pathlib.Path.iterdir(dir):
            if (str(filename).endswith(".pth") or str(filename).endswith(".pt")
               ) and not str(filename).startswith(".DS_Store"):
                if "cosine" in str(filename):
                    cosine_files.append(pathlib.Path(filename))
                else:
                    euclidean_files.append(pathlib.Path(filename))
            if str(filename).endswith("parameter_grid.pkl") and not str(
                    filename).startswith(".DS_Store"):
                with open(filename, "rb") as fp:
                    parameter_grid = pickle.load(fp)
            if str(filename).endswith("models_names.pkl") and not str(
                    filename).startswith(".DS_Store"):
                with open(filename, "rb") as fp:
                    models_names = pickle.load(fp)

        all_files = sorted(cosine_files) + sorted(euclidean_files)

        if len(all_files) < 0:
            raise ValueError(
                f"Missing models, the {dir} directory doesn't contain any model."
            )
        if not parameter_grid:
            raise ValueError(
                f"Missing parameter grid, the {dir} directory doesn't contain a parameter grid file."
            )
        if not models_names:
            raise ValueError(
                f"Missing parameter grid, the {dir} directory doesn't contain a parameter grid file."
            )

        # check that all models are fitted
        models = {}
        datasets = []
        for filename in all_files:
            model = cebra_sklearn_cebra.CEBRA.load(filename)
            sklearn_utils_validation.check_is_fitted(model, "n_features_")
            if filename.stem.split("_")[-1] not in datasets:
                datasets.append(filename.stem.split("_")[-1])
            models[filename.stem] = model

        if len(models) != len(models_names) or len(models) != len(
                parameter_grid):
            raise ValueError(
                f"Number of models doesn't correspond to the number of sets of parameters in the parameter grid, "
                f"got {len(models)} against {len(models_names)}.")

        models = {k: models[k] for k in models_names}
        return models, parameter_grid

    def get_best_model(
        self,
        scoring: Literal["infonce_loss"] = "infonce_loss",
        dataset_name: Optional[str] = None,
        models_dir: str = None,
    ) -> Tuple[cebra_sklearn_cebra.CEBRA, str]:
        """Get the model with the best performance across all sets of parameters and datasets.

        Args:
            scoring: Metric to use to evaluate the models performances.
            dataset_name: Name of the dataset to find the best model for. By default, ``dataset_name``
                is set to ``None`` and the best model is searched from the list of all models for all
                sets of parameters and across all datasets. A ``dataset_name`` is valid if models were
                fitted on that set of data and those models are present in ``models_dir``. Then, the
                returned model will correspond to the model with the highest performances on *that*
                dataset only.
            models_dir: The path to the folder in which save the (fitted) models.

        Returns:
            The :py:class:`cebra.CEBRA` model with the highest performance for a given ``dataset_name``
            (first returns) and its name (second returns).

        Example:

            >>> import cebra.grid_search
            >>> import numpy as np
            >>> neural_data =  np.random.uniform(0, 1, (300, 30))
            >>> # 1. Define the parameters for the models
            >>> params_grid = dict(
            ...     output_dimension = [3, 16],
            ...     learning_rate = [0.001],
            ...     time_offsets = 5,
            ...     max_iterations = 10,
            ...     verbose = False)
            >>> # 2. Fit the models generated from the list of parameters
            >>> grid_search = cebra.grid_search.GridSearch()
            >>> grid_search = grid_search.fit_models(datasets={"neural_data": neural_data},
            ...                        params=params_grid,
            ...                        models_dir="grid_search_models")
            >>> # 3. Get model with the best performances and use it as usual
            >>> best_model, best_model_name = grid_search.get_best_model()
            >>> embedding = best_model.transform(neural_data)

        """

        if not hasattr(self, "models_dir"):
            if models_dir is None:
                raise ValueError(
                    "Missing models directory, provide a value for the models_dir parameter."
                )
            self.models_dir = models_dir

        if not hasattr(self, "models"):
            self.models, _ = GridSearch().load(self.models_dir)

        best_score = [cebra_sklearn_cebra.CEBRA(), float("inf"), "test"]
        if scoring == "infonce_loss":
            for model_name in self.models.keys():
                if dataset_name is None or dataset_name in model_name:
                    model = self.models[model_name]
                    if best_score[1] > model.state_dict_["loss"][-1].item():
                        best_score[0] = model
                        best_score[1] = model.state_dict_["loss"][-1].item()
                        best_score[2] = model_name
        else:
            raise NotImplementedError(
                f"Invalid scoring, expect infonce_loss, got {scoring}.")

        if best_score[1] == float("inf"):
            raise ValueError(
                f"Invalid dataset name, that dataset wasn't find in the list of fitted models, got {dataset_name}."
            )

        return best_score[0], best_score[2]

    def get_df_results(self, models_dir: str = None) -> pd.DataFrame:
        """Create a :py:class:`pandas.DataFrame` containing the parameters and results for each model.

        Args:
            models_dir: The path to the folder in which save the (fitted) models.

        Returns:
            A :py:class:`pandas.DataFrame` in which each row corresponds to a model from the grid search
            and contains the parameters used for the model, the dataset name that was used for fitting and
            the performances obtained with that model.

        Example:

            >>> import cebra.grid_search
            >>> import numpy as np
            >>> neural_data =  np.random.uniform(0, 1, (300, 30))
            >>> # 1. Define the parameters for the models
            >>> params_grid = dict(
            ...     output_dimension = [3, 16],
            ...     learning_rate = [0.001],
            ...     time_offsets = 5,
            ...     max_iterations = 10,
            ...     verbose = False)
            >>> # 2. Fit the models generated from the list of parameters
            >>> grid_search = cebra.grid_search.GridSearch()
            >>> grid_search = grid_search.fit_models(datasets={"neural_data": neural_data},
            ...                        params=params_grid,
            ...                        models_dir="grid_search_models")
            >>> # 3. Get results for all models
            >>> df_results = grid_search.get_df_results()

        """
        if not hasattr(self, "models_dir"):
            if models_dir is None:
                raise ValueError(
                    "Missing models directory, provide a value for the models_dir parameter."
                )
            self.models_dir = models_dir

        if not hasattr(self, "models") or not hasattr(self, "parameter_grid"):
            self.models, self.parameter_grid = GridSearch().load(
                self.models_dir)

        results = pd.DataFrame(self.parameter_grid)
        results["loss"] = [
            model.state_dict_["loss"][-1].item()
            for model in list(self.models.values())
        ]
        results["dataset_name"] = [
            model_name.split("_")[-1] for model_name in self.models.keys()
        ]

        return results

    def plot_loss_comparison(self,
                             models_dir: str = None,
                             **kwargs) -> matplotlib.axes.Axes:
        """Display the losses for all fitted models present in ``models_dir``.

        Note:
            The method is a wrapper around :py:func:`~.compare_models`, meaning you can provide all
            parameters that are provided to :py:func:`~.compare_models` to that function.

        Args:
            models_dir: The path to the folder in which save the (fitted) models.

        Returns:
            A ``matplotlib.axes.Axes`` on which to generate the plot.

        Example:

            >>> import cebra.grid_search
            >>> import numpy as np
            >>> neural_data =  np.random.uniform(0, 1, (300, 30))
            >>> # 1. Define the parameters for the models
            >>> params_grid = dict(
            ...     output_dimension = [3, 16],
            ...     learning_rate = [0.001],
            ...     time_offsets = 5,
            ...     max_iterations = 10,
            ...     verbose = False)
            >>> # 2. Fit the models generated from the list of parameters
            >>> grid_search = cebra.grid_search.GridSearch()
            >>> grid_search = grid_search.fit_models(datasets={"neural_data": neural_data},
            ...                        params=params_grid,
            ...                        models_dir="grid_search_models")
            >>> # 3. Plot losses for all models
            >>> ax = grid_search.plot_loss_comparison()

        """
        if not hasattr(self, "models_dir"):
            if models_dir is None:
                raise ValueError(
                    "Missing models directory, provide a value for the models_dir parameter."
                )
            self.models_dir = models_dir

        if not hasattr(self, "models"):
            self.models, _ = GridSearch().load(self.models_dir)

        ax = cebra_matplotlib.compare_models(models=list(self.models.values()),
                                             labels=list(self.models.keys()),
                                             **kwargs)
        return ax

    def plot_transform(self):
        """TODO."""
        raise NotImplementedError()
