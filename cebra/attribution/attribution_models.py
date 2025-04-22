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
import dataclasses
import time

import cvxpy as cp
import numpy as np
import scipy.linalg
import sklearn.metrics
import torch
import torch.nn as nn
import tqdm
from captum.attr import NeuronFeatureAblation
from captum.attr import NeuronGradient
from captum.attr import NeuronGradientShap
from captum.attr import NeuronIntegratedGradients

import cebra
import cebra.attribution._jacobian
from cebra.attribution import register


@dataclasses.dataclass
class AttributionMap:
    """Base class for computing attribution maps for CEBRA models.

    This class provides the foundation for various attribution methods that analyze
    how input features contribute to a model's output. It handles data preprocessing,
    metric computation, and provides utility functions for matrix operations.

    Args:
        model: The trained CEBRA model to analyze
        input_data: Input data tensor to compute attributions for
        output_dimension: Output dimension to analyze. If ``None``, uses model's output dimension
        num_samples: Number of samples to use for attribution. If ``None``, uses full dataset
        seed: Random seed which is used to subsample the data. Only relevant if ``num_samples`` is not ``None``.

    Attributes:
        model: The CEBRA model being analyzed
        input_data: Preprocessed input data tensor
        output_dimension: Selected output dimension for analysis
        num_samples: Number of samples used for attribution
        seed: Random seed for reproducibility
    """

    model: nn.Module
    input_data: torch.Tensor
    output_dimension: int = None
    num_samples: int = None
    seed: int = 9712341

    def __post_init__(self):
        """Initialize the attribution map with proper data preprocessing.

        This method handles:
        1. Data configuration for convolutional models
        2. Subsampling of data if num_samples is specified
        3. Device management (CPU/GPU)
        """
        if isinstance(self.model, cebra.models.ConvolutionalModelMixin):
            data = cebra.data.TensorDataset(self.input_data,
                                            continuous=torch.zeros(
                                                len(self.input_data)))
            data.configure_for(self.model)
            offset = self.model.get_offset()

            input_data = data[torch.arange(offset.left,
                                           len(data) - offset.right + 1)].to(
                                               self.input_data.device)

            # subsample the data
            if self.num_samples is not None:
                if self.num_samples > input_data.shape[0]:
                    raise ValueError(
                        f"You are using a bigger number of samples to "
                        f"subsample ({self.num_samples}) than the number "
                        f"of samples in the dataset ({input_data.shape[0]}).")

                random_generator = torch.Generator()
                random_generator.manual_seed(self.seed)
                num_elements = input_data.size(0)
                random_indices = torch.randperm(
                    num_elements, generator=random_generator)[:self.num_samples]
                input_data = input_data[random_indices]

            self.input_data = input_data

    def compute_attribution_map(self):
        """Compute the attribution map for the model.

        This method must be implemented by subclasses to define specific attribution
        computation methods (e.g., Jacobian-based, gradient-based, etc.).

        Returns:
            dict: Attribution maps and their variants, typically containing:
                - 'attribution': The main attribution map
                - 'attribution_abs': Absolute values of attribution
                - 'attribution_squared': Squared attribution values

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def compute_metrics(self, attribution_map, ground_truth_map):
        """Compute metrics comparing attribution map to ground truth.

        This function computes various statistical metrics to compare the attribution values
        between connected and non-connected neurons based on a ground truth connectivity map.
        It separates the attribution values into two groups based on the binary ground truth,
        and calculates summary statistics and differences between these groups.

        Args:
            attribution_map: Computed attribution values representing the strength of connections
                between neurons. Shape should match ground_truth_map.
            ground_truth_map: Binary ground truth connectivity map where True indicates a
                connected neuron and False indicates a non-connected neuron.

        Returns:
            dict: Dictionary containing the following metrics:
                - max/mean/min_nonconnected: Statistics for non-connected neurons
                - max/mean/min_connected: Statistics for connected neurons
                - gap_max: Difference between max connected and max non-connected values
                - gap_mean: Difference between mean connected and mean non-connected values
                - gap_min: Difference between min connected and min non-connected values
                - gap_minmax: Difference between min connected and max non-connected values
                - max/min_jacobian: Global max/min values across all neurons

        Raises:
            AssertionError: If ground_truth_map is not boolean or shapes don't match
        """
        assert np.issubdtype(ground_truth_map.dtype, bool)
        connected_neurons = attribution_map[np.where(ground_truth_map)]
        non_connected_neurons = attribution_map[np.where(~ground_truth_map)]
        assert connected_neurons.size == ground_truth_map.sum()
        assert non_connected_neurons.size == ground_truth_map.size - ground_truth_map.sum()
        assert connected_neurons.size + non_connected_neurons.size == attribution_map.size == ground_truth_map.size

        max_connected = np.max(connected_neurons)
        mean_connected = np.mean(connected_neurons)
        min_connected = np.min(connected_neurons)

        max_nonconnected = np.max(non_connected_neurons)
        mean_nonconnected = np.mean(non_connected_neurons)
        min_nonconnected = np.min(non_connected_neurons)

        metrics = {
            'max_nonconnected': max_nonconnected,
            'mean_nonconnected': mean_nonconnected,
            'min_nonconnected': min_nonconnected,
            'max_connected': max_connected,
            'mean_connected': mean_connected,
            'min_connected': min_connected,
            'gap_max': max_connected - max_nonconnected,
            'gap_mean': mean_connected - mean_nonconnected,
            'gap_min': min_connected - min_nonconnected,
            'gap_minmax': min_connected - max_nonconnected,
            'max_jacobian': np.max(attribution_map),
            'min_jacobian': np.min(attribution_map),
        }
        return metrics

    def compute_attribution_score(self, attribution_map, ground_truth_map):
        """Compute ROC AUC score between attribution map and ground truth.

        This method evaluates how well the attribution map can distinguish between
        connected and non-connected neurons using the area under the ROC curve.

        Args:
            attribution_map: Computed attribution values. Shape should match ground_truth_map.
            ground_truth_map: Binary ground truth connectivity map.

        Returns:
            float: ROC AUC score between 0 and 1, where 1 indicates perfect discrimination
                between connected and non-connected neurons.

        Raises:
            AssertionError: If shapes don't match or ground_truth_map is not boolean
        """
        assert attribution_map.shape == ground_truth_map.shape
        assert np.issubdtype(ground_truth_map.dtype, bool)
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            ground_truth_map.flatten(), attribution_map.flatten())
        auc = sklearn.metrics.auc(fpr, tpr)
        return auc

    @staticmethod
    def _check_moores_penrose_conditions(
            matrix: np.ndarray, matrix_inverse: np.ndarray) -> np.ndarray:
        """Check Moore-Penrose conditions for a single matrix pair.

        This method verifies if a given matrix and its putative inverse satisfy
        the four Moore-Penrose conditions for pseudoinverses.

        Args:
            matrix: Input matrix of shape (m, n)
            matrix_inverse: Putative pseudoinverse matrix of shape (n, m)

        Returns:
            np.ndarray: Boolean array of length 4 indicating which conditions are satisfied:
                [condition_1, condition_2, condition_3, condition_4]
        """
        matrix_inverse = matrix_inverse.T
        condition_1 = np.allclose(matrix @ matrix_inverse @ matrix, matrix)
        condition_2 = np.allclose(matrix_inverse @ matrix @ matrix_inverse,
                                  matrix_inverse)
        condition_3 = np.allclose((matrix @ matrix_inverse).T,
                                  matrix @ matrix_inverse)
        condition_4 = np.allclose((matrix_inverse @ matrix).T,
                                  matrix_inverse @ matrix)

        return np.array([condition_1, condition_2, condition_3, condition_4])

    def check_moores_penrose_conditions(
            self, jacobian: np.ndarray,
            jacobian_pseudoinverse: np.ndarray) -> np.ndarray:
        """Check Moore-Penrose conditions for Jacobian matrices.

        This method verifies the Moore-Penrose conditions for a batch of Jacobian
        matrices and their pseudoinverses.

        Args:
            jacobian: Jacobian matrices of shape (num_samples, output_dim, num_neurons)
            jacobian_pseudoinverse: Pseudoinverse matrices of shape (num_samples, num_neurons, output_dim)

        Returns:
            np.ndarray: Boolean array of shape (num_samples, 4) indicating which
                conditions are satisfied for each sample
        """
        conditions = np.zeros((jacobian.shape[0], 4))
        for i, (matrix, inverse_matrix) in enumerate(
                zip(jacobian, jacobian_pseudoinverse)):
            conditions[i] = self._check_moores_penrose_conditions(
                matrix, inverse_matrix)
        return conditions

    def _inverse(self, jacobian, method="lsq"):
        """Compute inverse/pseudoinverse of Jacobian matrices.

        This method computes the inverse or pseudoinverse of Jacobian matrices
        using different numerical methods for improved stability.

        Args:
            jacobian: Input Jacobian matrices of shape (num_samples, output_dim, num_neurons)
            method: Inversion method to use:
                - 'lsq_cvxpy': Uses CVXPY for least squares solution
                - 'lsq': Uses scipy's least squares solver
                - 'svd': Uses singular value decomposition

        Returns:
            tuple: (Inverse matrices of shape (num_samples, num_neurons, output_dim),
                   computation time in seconds)

        Raises:
            NotImplementedError: If the specified method is not implemented
        """
        start_time = time.time()
        Jfinv = np.zeros_like(jacobian)
        if method == "lsq_cvxpy":
            for i in tqdm(range(len(jacobian))):
                Jfinv[i] = self._inverse_lsq_cvxpy(jacobian[i]).T
        elif method == "lsq":
            for i in range(len(jacobian)):
                Jfinv[i] = self._inverse_lsq_scipy(jacobian[i]).T
        elif method == "svd":
            for i in range(len(jacobian)):
                Jfinv[i] = self._inverse_svd(jacobian[i]).T
        else:
            raise NotImplementedError(f"Method {method} not implemented.")
        end_time = time.time()
        return Jfinv, end_time - start_time

    @staticmethod
    def _inverse_lsq_cvxpy(matrix: np.ndarray,
                           solver: str = 'SCS') -> np.ndarray:
        """Compute least squares inverse using CVXPY.

        This method computes the pseudoinverse using CVXPY's optimization framework,
        which can provide more stable solutions for ill-conditioned matrices.

        Args:
            matrix: Input matrix of shape (m, n)
            solver: CVXPY solver to use (default: 'SCS')

        Returns:
            np.ndarray: Pseudoinverse matrix of shape (n, m)
        """

        matrix_param = cp.Parameter((matrix.shape[0], matrix.shape[1]))
        matrix_param.value = matrix

        identity = np.eye(matrix.shape[0])
        matrix_inverse = cp.Variable((matrix.shape[1], matrix.shape[0]))
        # noqa: codespell
        objective = cp.Minimize(
            cp.norm(matrix @ matrix_inverse - identity,
                    "fro"))  # noqa: codespell:ignore fro
        prob = cp.Problem(objective)
        prob.solve(verbose=False, solver=solver)

        return matrix_inverse.value

    @staticmethod
    def _inverse_lsq_scipy(jacobian):
        """Compute least squares inverse using scipy.linalg.lstsq.

        Args:
            jacobian: Input Jacobian matrix

        Returns:
            np.ndarray: Least squares inverse matrix
        """
        return scipy.linalg.lstsq(jacobian, np.eye(jacobian.shape[0]))[0]

    @staticmethod
    def _inverse_svd(jacobian):
        """Compute pseudoinverse using SVD.

        Args:
            jacobian: Input Jacobian matrix of shape (output_dim, num_neurons)

        Returns:
            np.ndarray: Pseudoinverse matrix of shape (num_neurons, output_dim)
        """
        return scipy.linalg.pinv(jacobian)

    def _reduce_attribution_map(self, attribution_maps):
        """Reduce attribution maps to a single value per neuron pair.

        This method combines multiple attribution maps (e.g., from different samples
        or different variants) into a single map by taking the mean across samples.

        Args:
            attribution_maps: Dictionary of attribution maps, where each value is
                a numpy array of shape (num_samples, num_neurons, num_neurons)

        Returns:
            dict: Reduced attribution maps with the same keys as input, but with
                shape (num_neurons, num_neurons)
        """
        def _reduce(full_jacobian):
            if full_jacobian.ndim == 4:
                jf_convabs = abs(full_jacobian).mean(-1)
                jf = full_jacobian.mean(-1)
            else:
                jf_convabs = full_jacobian
                jf = full_jacobian
            return jf, jf_convabs

        result = {}
        for key, value in attribution_maps.items():
            result[key], result[f'{key}-convabs'] = _reduce(value)
        return result


@dataclasses.dataclass
@register("jacobian-based")
class JFMethodBased(AttributionMap):
    """Jacobian-based attribution method for CEBRA models.

    This class implements attribution using the Jacobian matrix of the model,
    which represents the first-order partial derivatives of the output with
    respect to the input. It provides insights into how small changes in input
    features affect the model's output.

    Args:
        model: The trained CEBRA model to analyze
        input_data: Input data tensor to compute attributions for
        output_dimension: Output dimension to analyze. If ``None``, uses model's output dimension
        num_samples: Number of samples to use for attribution. If ``None``, uses full dataset
        seed: Random seed which is used to subsample the data
    """

    def _compute_jacobian(self, input_data):
        """Compute the Jacobian matrix for the given input data.

        Args:
            input_data: Input tensor of shape (batch_size, input_dim)

        Returns:
            np.ndarray: Jacobian matrix of shape (batch_size, output_dim, input_dim)
        """
        return cebra.attribution._jacobian.compute_jacobian(
            self.model,
            input_vars=[input_data],
            mode="autograd",
            cuda_device=self.input_data.device,
            double_precision=False,
            convert_to_numpy=True,
            hybrid_solver=False,
        )

    def compute_attribution_map(self):
        """Compute the attribution map using Jacobian-based method.

        full_jacobian = self._compute_jacobian(self.input_data)

        result = {}
        for key, value in self._reduce_attribution_map({
                'jf': full_jacobian
        }).items():
            result[key] = value
            for method in ['lsq', 'svd']:
                print(f"Computing inverse for {key} with method {method}")
                result[f"{key}-inv-{method}"], result[
                    f'time_inversion_{method}'] = self._inverse(value,
                                                                method=method)
                # result[f"{key}-inv-{method}-conditions"] = self.check_moores_penrose_conditions(value, result[f"{key}-inv-{method}"])

        return result


@dataclasses.dataclass
@register("jacobian-based-batched")
class JFMethodBasedBatched(JFMethodBased):
    """Batched version of the Jacobian-based attribution method.

    This class extends JFMethodBased to compute attribution maps using mini-batches,
    which is useful for handling large datasets that don't fit in memory. It processes
    the input data in smaller chunks and combines the results.

    Args:
        model: The trained CEBRA model to analyze
        input_data: Input data tensor to compute attributions for
        output_dimension: Output dimension to analyze. If ``None``, uses model's output dimension
        num_samples: Number of samples to use for attribution. If ``None``, uses full dataset
        seed: Random seed which is used to subsample the data

    See also:
        :py:class:`JFMethodBased`: The base class for Jacobian-based attribution
    """

    def compute_attribution_map(self, batch_size=1024):
        """Compute the attribution map using batched Jacobian method.

        This method processes the input data in mini-batches to handle large datasets
        that don't fit in memory. It computes the Jacobian for each batch and combines
        the results.

        Args:
            batch_size: Size of each mini-batch. Default is 1024.

        Returns:
            dict: Dictionary containing attribution maps and their variants

        Raises:
            ValueError: If batch_size is larger than the number of samples
        """
        if batch_size > self.input_data.shape[0]:
            raise ValueError(
                f"Batch size ({batch_size}) is bigger than data ({self.input_data.shape[0]})"
            )

        input_data_batches = torch.split(self.input_data, batch_size)
        full_jacobian = []
        for input_data_batch in input_data_batches:
            jacobian_batch = self._compute_jacobian(input_data_batch)
            full_jacobian.append(jacobian_batch)
        full_jacobian = np.vstack(full_jacobian)

        result = {}
        for key, value in self._reduce_attribution_map({
                'jf': full_jacobian
        }).items():
            result[key] = value
            for method in ['lsq', 'svd']:
                result[f"{key}-inv-{method}"], result[
                    f'time_inversion_{method}'] = self._inverse(value,
                                                                method=method)

        return result


@dataclasses.dataclass
@register("neuron-gradient")
class NeuronGradientMethod(AttributionMap):
    """Compute the attribution map using the neuron gradient from Captum.

    This class implements attribution using the gradients of specific output
    neurons with respect to the input. It provides insights into how input
    features influence specific aspects of the model's output.

    Args:
        model: The trained CEBRA model to analyze
        input_data: Input data tensor to compute attributions for
        output_dimension: Output dimension to analyze. If ``None``, uses model's output dimension
        num_samples: Number of samples to use for attribution. If ``None``, uses full dataset
        seed: Random seed which is used to subsample the data
    """

    def __post_init__(self):
        super().__post_init__()
        self.captum_model = NeuronGradient(forward_func=self.model,
                                           layer=self.model)

    def compute_attribution_map(self, attribute_to_neuron_input=False):
        attribution_map = []
        for s in range(self.output_dimension):
            att = self.captum_model.attribute(
                inputs=self.input_data,
                attribute_to_neuron_input=attribute_to_neuron_input,
                neuron_selector=s)

            attribution_map.append(att.detach().cpu().numpy())

        attribution_map = np.array(attribution_map)
        attribution_map = np.swapaxes(attribution_map, 1, 0)

        result = {}
        for key, value in self._reduce_attribution_map({
                'neuron-gradient': attribution_map
        }).items():
            result[key] = value

            for method in ['lsq', 'svd']:
                result[f"{key}-inv-{method}"], result[
                    f'time_inversion_{method}'] = self._inverse(value,
                                                                method=method)
                # result[f"{key}-inv-{method}-conditions"] = self.check_moores_penrose_conditions(value, result[f"{key}-inv-{method}"])

        return result


@dataclasses.dataclass
@register("neuron-gradient-batched")
class NeuronGradientMethodBatched(NeuronGradientMethod):
    """As :py:class:`NeuronGradientMethod`, but using mini-batches.

    See also:
        :py:class:`NeuronGradientMethod`
    """

    def compute_attribution_map(
        self,
        attribute_to_neuron_input: bool = False,
        batch_size: int = 1024
    ) -> dict:
        """Compute attribution map using mini-batches.

        Args:
            attribute_to_neuron_input: If True, attribute to neuron input
            batch_size: Size of mini-batches for processing

        Returns:
            Dictionary containing attribution maps
        """
        input_data_batches = torch.split(self.input_data, batch_size)

        attribution_map = []
        for input_data_batch in input_data_batches:
            attribution_map_batch = []
            for s in range(self.output_dimension):
                att = self.captum_model.attribute(
                    inputs=input_data_batch,
                    attribute_to_neuron_input=attribute_to_neuron_input,
                    neuron_selector=s)

                attribution_map_batch.append(att.detach().cpu().numpy())

            attribution_map_batch = np.array(attribution_map_batch)
            attribution_map_batch = np.swapaxes(attribution_map_batch, 1, 0)
            attribution_map.append(attribution_map_batch)

        attribution_map = np.vstack(attribution_map)
        return self._reduce_attribution_map(
            {'neuron-gradient': attribution_map})


@dataclasses.dataclass
@register("feature-ablation")
class FeatureAblationMethod(AttributionMap):
    """Feature ablation-based attribution method for CEBRA models.

    This class implements attribution by systematically ablating (zeroing out)
    input features and measuring the impact on the model's output. It provides
    insights into the importance of individual features for the model's predictions.

    Args:
        model: The trained CEBRA model to analyze
        input_data: Input data tensor to compute attributions for
        output_dimension: Output dimension to analyze. If ``None``, uses model's output dimension
        num_samples: Number of samples to use for attribution. If ``None``, uses full dataset
        seed: Random seed which is used to subsample the data
    """

    def __post_init__(self):
        """Initialize the feature ablation method.

        This method sets up the feature ablation attribution object and ensures
        the model and input data are properly configured.
        """
        super().__post_init__()
        self.captum_model = NeuronFeatureAblation(forward_func=self.model,
                                                 layer=self.model)

    def compute_attribution_map(self,
                              baselines=None,
                              feature_mask=None,
                              perturbations_per_eval=1,
                              attribute_to_neuron_input=False):
        """Compute the attribution map using feature ablation method.

        Args:
            baselines: Baseline values to use for feature ablation. If None,
                uses zero baseline
            feature_mask: Binary mask indicating which features to ablate
            perturbations_per_eval: Number of perturbations to evaluate at once
            attribute_to_neuron_input: If True, attribute to the input of the
                neuron layer instead of the raw input

        Returns:
            dict: Dictionary containing attribution maps and their variants
        """
        attribution = self.feature_ablation.attribute(
            self.input_data,
            self.output_dimension,
            baselines=baselines,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            attribute_to_neuron_input=attribute_to_neuron_input)
        attribution_maps = {
            'attribution': attribution.detach().cpu().numpy(),
            'attribution_abs': np.abs(attribution.detach().cpu().numpy()),
            'attribution_squared': (attribution.detach().cpu().numpy())**2,
        }
        return self._reduce_attribution_map(attribution_maps)


@dataclasses.dataclass
@register("feature-ablation-batched")
class FeatureAblationMethodBatched(FeatureAblationMethod):
    """As :py:class:`FeatureAblationMethod`, but using mini-batches.

    See also:
        :py:class:`FeatureAblationMethod`
    """

    def compute_attribution_map(self,
                              baselines=None,
                              feature_mask=None,
                              perturbations_per_eval=1,
                              attribute_to_neuron_input=False,
                              batch_size=1024):
        input_data_batches = torch.split(self.input_data, batch_size)
        attribution_map = []
        for input_data_batch in input_data_batches:
            attribution_map_batch = []
            for s in range(self.output_dimension):
                att = self.captum_model.attribute(
                    inputs=input_data_batch,
                    neuron_selector=s,
                    baselines=baselines,
                    perturbations_per_eval=perturbations_per_eval,
                    feature_mask=feature_mask,
                    attribute_to_neuron_input=attribute_to_neuron_input)

                attribution_map_batch.append(att.detach().cpu().numpy())

            attribution_map_batch = np.array(attribution_map_batch)
            attribution_map_batch = np.swapaxes(attribution_map_batch, 1, 0)
            attribution_map.append(attribution_map_batch)

        attribution_map = np.vstack(attribution_map)
        return self._reduce_attribution_map(
            {'feature-ablation': attribution_map})


@dataclasses.dataclass
@register("integrated-gradients")
class IntegratedGradientsMethod(AttributionMap):
    """Integrated gradients-based attribution method for CEBRA models.

    This class implements attribution using the integrated gradients method,
    which computes the integral of gradients along the path from a baseline
    to the input. It provides a more robust measure of feature importance
    by considering multiple points along this path.

    Args:
        model: The trained CEBRA model to analyze
        input_data: Input data tensor to compute attributions for
        output_dimension: Output dimension to analyze. If ``None``, uses model's output dimension
        num_samples: Number of samples to use for attribution. If ``None``, uses full dataset
        seed: Random seed which is used to subsample the data
    """

    def __post_init__(self):
        """Initialize the integrated gradients method.

        This method sets up the integrated gradients attribution object and ensures
        the model and input data are properly configured.
        """
        super().__post_init__()
        self.integrated_gradients = NeuronIntegratedGradients(self.model)

    def compute_attribution_map(self,
                              n_steps=50,
                              method='gausslegendre',
                              internal_batch_size=None,
                              attribute_to_neuron_input=False,
                              baselines=None):
        """Compute the attribution map using integrated gradients method.

        Args:
            n_steps: Number of steps to use for numerical integration
            method: Integration method to use ('gausslegendre' or 'riemann')
            internal_batch_size: Batch size for internal computations
            attribute_to_neuron_input: If True, attribute to the input of the
                neuron layer instead of the raw input
            baselines: Baseline values to use for integration. If None,
                uses zero baseline

        Returns:
            dict: Dictionary containing attribution maps and their variants
        """
        attribution = self.integrated_gradients.attribute(
            self.input_data,
            self.output_dimension,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            attribute_to_neuron_input=attribute_to_neuron_input,
            baselines=baselines)
        attribution_maps = {
            'attribution': attribution.detach().cpu().numpy(),
            'attribution_abs': np.abs(attribution.detach().cpu().numpy()),
            'attribution_squared': (attribution.detach().cpu().numpy())**2,
        }
        return self._reduce_attribution_map(attribution_maps)


@dataclasses.dataclass
@register("integrated-gradients-batched")
class IntegratedGradientsMethodBatched(IntegratedGradientsMethod):
    """As :py:class:`IntegratedGradientsMethod`, but using mini-batches.

    See also:
        :py:class:`IntegratedGradientsMethod`
    """

    def compute_attribution_map(self,
                                n_steps=50,
                                method='gausslegendre',
                                internal_batch_size=None,
                                attribute_to_neuron_input=False,
                                baselines=None,
                                batch_size=1024):

        input_data_batches = torch.split(self.input_data, batch_size)
        attribution_map = []
        for input_data_batch in input_data_batches:
            attribution_map_batch = []
            if internal_batch_size == "dataset":
                internal_batch_size = len(input_data_batch)
            for s in range(self.output_dimension):
                att = self.captum_model.attribute(
                    inputs=input_data_batch,
                    neuron_selector=s,
                    n_steps=n_steps,
                    method=method,
                    internal_batch_size=internal_batch_size,
                    attribute_to_neuron_input=attribute_to_neuron_input,
                    baselines=baselines,
                )
                attribution_map_batch.append(att.detach().cpu().numpy())

            attribution_map_batch = np.array(attribution_map_batch)
            attribution_map_batch = np.swapaxes(attribution_map_batch, 1, 0)
            attribution_map.append(attribution_map_batch)

        attribution_map = np.vstack(attribution_map)
        return self._reduce_attribution_map(
            {'integrated-gradients': attribution_map})


@dataclasses.dataclass
@register("neuron-gradient-shap")
class NeuronGradientShapMethod(AttributionMap):
    """Neuron gradient SHAP-based attribution method for CEBRA models.

    This class implements attribution using the gradient SHAP method, which
    combines ideas from SHAP (SHapley Additive exPlanations) and gradient-based
    attribution. It provides a theoretically grounded measure of feature importance
    that satisfies certain desirable properties.

    Args:
        model: The trained CEBRA model to analyze
        input_data: Input data tensor to compute attributions for
        output_dimension: Output dimension to analyze. If ``None``, uses model's output dimension
        num_samples: Number of samples to use for attribution. If ``None``, uses full dataset
        seed: Random seed which is used to subsample the data
    """

    def __post_init__(self):
        """Initialize the neuron gradient SHAP method.

        This method sets up the gradient SHAP attribution object and ensures
        the model and input data are properly configured.
        """
        super().__post_init__()
        self.captum_model = NeuronGradientShap(forward_func=self.model,
                                               layer=self.model)

    def compute_attribution_map(self,
                              baselines: str,
                              n_samples=5,
                              stdevs=0.0,
                              attribute_to_neuron_input=False):
        """Compute the attribution map using neuron gradient SHAP method.

        Args:
            baselines: Type of baseline to use ('random', 'zero', or 'uniform')
            n_samples: Number of samples to use for SHAP value estimation
            stdevs: Standard deviation for random sampling
            attribute_to_neuron_input: If True, attribute to the input of the
                neuron layer instead of the raw input

        Returns:
            dict: Dictionary containing attribution maps and their variants
        """
        attribution = self.gradient_shap.attribute(
            self.input_data,
            self.output_dimension,
            baselines=baselines,
            n_samples=n_samples,
            stdevs=stdevs,
            attribute_to_neuron_input=attribute_to_neuron_input)
        attribution_maps = {
            'attribution': attribution.detach().cpu().numpy(),
            'attribution_abs': np.abs(attribution.detach().cpu().numpy()),
            'attribution_squared': (attribution.detach().cpu().numpy())**2,
        }
        return self._reduce_attribution_map(attribution_maps)


@dataclasses.dataclass
@register("neuron-gradient-shap-batched")
class NeuronGradientShapMethodBatched(NeuronGradientShapMethod):
    """As :py:class:`NeuronGradientShapMethod`, but using mini-batches.

    See also:
        :py:class:`NeuronGradientShapMethod`
    """

    def compute_attribution_map(self,
                                baselines: str,
                                n_samples=5,
                                stdevs=0.0,
                                attribute_to_neuron_input=False,
                                batch_size=1024):

        if baselines == "zeros":
            baselines = torch.zeros(size=(self.input_data.shape),
                                    device=self.input_data.device)
        elif baselines == "shuffle":
            data = self.input_data.flatten()
            data = data[torch.randperm(len(data))]
            baselines = data.reshape(self.input_data.shape)
        else:
            raise NotImplementedError(f"Baseline {baselines} not implemented.")

        input_data_batches = torch.split(self.input_data, batch_size)
        attribution_map = []
        for input_data_batch in input_data_batches:
            attribution_map_batch = []
            for s in range(self.output_dimension):
                att = self.captum_model.attribute(
                    inputs=input_data_batch,
                    neuron_selector=s,
                    baselines=baselines,
                    n_samples=n_samples,
                    stdevs=stdevs,
                    attribute_to_neuron_input=attribute_to_neuron_input,
                )

                attribution_map_batch.append(att.detach().cpu().numpy())

            attribution_map_batch = np.array(attribution_map_batch)
            attribution_map_batch = np.swapaxes(attribution_map_batch, 1, 0)
            attribution_map.append(attribution_map_batch)

        attribution_map = np.vstack(attribution_map)
        return self._reduce_attribution_map(
            {'neuron-gradient-shap': attribution_map})
