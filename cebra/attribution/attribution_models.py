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

import numpy as np
import scipy.linalg
import sklearn.metrics
import torch
import torch.nn as nn
import tqdm

import cebra
import cebra.attribution._jacobian
from cebra.attribution import register


_VALID_ATTRIBUTION_RESULT_KEYS = {
    "jf",
    "jf-convabs",
    "jf-inv-lsq",
    "jf-inv-svd",
    "jf-convabs-inv-lsq",
    "jf-convabs-inv-svd",
}


def _normalize_selected_result_keys(selected_keys):
    """Map names as a set (``_`` read as ``-``), or ``None``; raise on unknown names."""
    if selected_keys is None:
        return None
    normalized = []
    for key in selected_keys:
        key_str = str(key).strip().replace("_", "-")
        if not key_str:
            continue
        if key_str not in _VALID_ATTRIBUTION_RESULT_KEYS:
            raise ValueError(
                f"Unknown attribution result key {key!r}. "
                f"Valid keys: {sorted(_VALID_ATTRIBUTION_RESULT_KEYS)}"
            )
        normalized.append(key_str)
    return set(normalized)


@dataclasses.dataclass
class AttributionMap:
    """Base class for computing attribution maps for CEBRA models.

    Args:
        model: The trained CEBRA model to analyze
        input_data: Input data tensor to compute attributions for
        output_dimension: Output dimension to analyze. If ``None``, uses model's output dimension
        num_samples: Number of samples to use for attribution. If ``None``, uses full dataset
        seed: Random seed which is used to subsample the data. Only relevant if ``num_samples`` is not ``None``.
    """

    model: nn.Module
    input_data: torch.Tensor
    output_dimension: int = None
    num_samples: int = None
    seed: int = 9712341

    def __post_init__(self):
        if isinstance(self.model, cebra.models.ConvolutionalModelMixin):
            data = cebra.data.TensorDataset(self.input_data,
                                            continuous=torch.zeros(
                                                len(self.input_data)))
            data.configure_for(self.model)
            offset = self.model.get_offset()

            #NOTE: explain, why do we do this again?
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

        Returns:
            dict: Attribution maps and their variants

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
                between neurons
            ground_truth_map: Binary ground truth connectivity map where True indicates a
                connected neuron and False indicates a non-connected neuron

        Returns:
            dict: Dictionary containing the following metrics:
                - max/mean/min_nonconnected: Statistics for non-connected neurons
                - max/mean/min_connected: Statistics for connected neurons
                - gap_max: Difference between max connected and max non-connected values
                - gap_mean: Difference between mean connected and mean non-connected values
                - gap_min: Difference between min connected and min non-connected values
                - gap_minmax: Difference between min connected and max non-connected values
                - max/min_jacobian: Global max/min values across all neurons
        """
        assert np.issubdtype(ground_truth_map.dtype, bool)
        connected_neurons = attribution_map[np.where(ground_truth_map)]
        non_connected_neurons = attribution_map[np.where(~ground_truth_map)]
        assert connected_neurons.size == ground_truth_map.sum()
        assert non_connected_neurons.size == ground_truth_map.size - ground_truth_map.sum(
        )
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

        Args:
            attribution_map: Computed attribution values
            ground_truth_map: Binary ground truth connectivity map

        Returns:
            float: ROC AUC score
        """
        assert attribution_map.shape == ground_truth_map.shape
        assert np.issubdtype(ground_truth_map.dtype, bool)
        fpr, tpr, _ = sklearn.metrics.roc_curve(  # noqa: codespell:ignore fpr, tpr
            ground_truth_map.flatten(), attribution_map.flatten())
        auc = sklearn.metrics.auc(fpr, tpr)  # noqa: codespell:ignore fpr, tpr
        return auc

    @staticmethod
    def _check_moores_penrose_conditions(
            matrix: np.ndarray, matrix_inverse: np.ndarray) -> np.ndarray:
        """Check Moore-Penrose conditions for a single matrix pair.

        Args:
            matrix: Input matrix
            matrix_inverse: Putative pseudoinverse matrix

        Returns:
            np.ndarray: Boolean array indicating which conditions are satisfied
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

        Args:
            jacobian: Jacobian matrices of shape (num samples, output_dim, num_neurons)
            jacobian_pseudoinverse: Pseudoinverse matrices of shape (num samples, num_neurons, output_dim)

        Returns:
            Boolean array of shape (num samples, 4) indicating satisfied conditions
        """
        # check the four conditions
        conditions = np.zeros((jacobian.shape[0], 4))
        for i, (matrix, inverse_matrix) in enumerate(
                zip(jacobian, jacobian_pseudoinverse)):
            conditions[i] = self._check_moores_penrose_conditions(
                matrix, inverse_matrix)
        return conditions

    def _inverse(self, jacobian, method="lsq"):
        """Compute inverse/pseudoinverse of Jacobian matrices.

        Args:
            jacobian: Input Jacobian matrices
            method: Inversion method ('lsq_cvxpy', 'lsq', or 'svd')

        Returns:
            (Inverse matrices, computation time)
        """
        # NOTE(stes): Before we used "np.linalg.pinv" here, which
        # is numerically not stable for the Jacobian matrices we
        # need to compute.
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

        Args:
            matrix: Input matrix
            solver: CVXPY solver to use

        Returns:
            np.ndarray: Least squares inverse matrix
        """
        import cvxpy as cp

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
            jacobian: Input Jacobian matrix

        Returns:
            np.ndarray: Pseudoinverse matrix
        """
        return scipy.linalg.pinv(jacobian)

    def _reduce_attribution_map(self, attribution_maps):
        """Reduce attribution maps by averaging across dimensions.

        Args:
            attribution_maps: Dictionary of attribution maps to reduce

        Returns:
            dict: Reduced attribution maps
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
    """Compute the attribution map using the Jacobian of the model encoder."""

    def _compute_jacobian(self, input_data, output_batch_size=None):
        return cebra.attribution._jacobian.compute_jacobian(
            self.model,
            input_vars=[input_data],
            mode="autograd",
            cuda_device=self.input_data.device,
            double_precision=False,
            convert_to_numpy=True,
            hybrid_solver=False,
            output_batch_size=output_batch_size,
        )

    def compute_attribution_map(self, selected_keys=None):
        selected_keys = _normalize_selected_result_keys(selected_keys)

        full_jacobian = self._compute_jacobian(self.input_data)

        result = {}
        for key, value in self._reduce_attribution_map({
                'jf': full_jacobian
        }).items():
            if selected_keys is None or key in selected_keys:
                result[key] = value
            for method in ['lsq', 'svd']:
                inv_key = f"{key}-inv-{method}"
                if selected_keys is not None and inv_key not in selected_keys:
                    continue
                print(f"Computing inverse for {key} with method {method}")
                result[inv_key], result[f'time_inversion_{method}'] = self._inverse(
                    value, method=method
                )
                # result[f"{key}-inv-{method}-conditions"] = self.check_moores_penrose_conditions(value, result[f"{key}-inv-{method}"])

        return result


@dataclasses.dataclass
@register("jacobian-based-batched")
class JFMethodBasedBatched(JFMethodBased):
    """Compute an attribution map based on the Jacobian using mini-batches.

    See also:
        :py:class:`JFMethodBased`
    """

    def compute_attribution_map(self, batch_size=1024, selected_keys=None):
        selected_keys = _normalize_selected_result_keys(selected_keys)
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
            if selected_keys is None or key in selected_keys:
                result[key] = value
            for method in ['lsq', 'svd']:
                inv_key = f"{key}-inv-{method}"
                if selected_keys is not None and inv_key not in selected_keys:
                    continue
                result[inv_key], result[f'time_inversion_{method}'] = self._inverse(
                    value, method=method
                )

        return result


@dataclasses.dataclass
@register("jacobian-based-batched-optimized")
class JFMethodBasedBatchedOptimized(JFMethodBased):
    """Jacobian-based attribution computed in mini-batches, with an optional running mean.

    Same results as ``jacobian-based-batched``, with less memory.
    """

    def compute_attribution_map(
        self,
        batch_size=1024,
        return_full=True,
        selected_keys=None,
        latent_batch_size=None,
    ):
        """Compute the attribution maps.

        Args:
            batch_size: Samples per Jacobian batch.
            return_full: Return one map per sample. If ``False``, return the mean over samples.
            selected_keys: Maps to compute, e.g. ``["jf"]``. ``None``: all of them.
            latent_batch_size: Embedding dimensions per autograd pass. ``None``: all at once.

        Returns:
            Dict from map name to array, ``(T, D, N)`` per sample or ``(D, N)`` averaged.
        """
        selected_keys = _normalize_selected_result_keys(selected_keys)
        if batch_size > self.input_data.shape[0]:
            raise ValueError(
                f"Batch size ({batch_size}) is bigger than data ({self.input_data.shape[0]})"
            )

        input_data_batches = torch.split(self.input_data, batch_size)
        keys = sorted(_VALID_ATTRIBUTION_RESULT_KEYS if selected_keys is None else selected_keys)

        if return_full:
            per_key_chunks = {key: [] for key in keys}
            running_sums = None
            total_samples = None
        else:
            per_key_chunks = None
            running_sums = {key: None for key in keys}
            total_samples = 0

        time_inversion = {'lsq': 0.0, 'svd': 0.0}

        for input_data_batch in input_data_batches:
            jacobian_batch = self._compute_jacobian(
                input_data_batch, output_batch_size=latent_batch_size
            )
            reduced_batch = self._reduce_attribution_map({'jf': jacobian_batch})
            batch_samples = next(iter(reduced_batch.values())).shape[0]

            for key, value in reduced_batch.items():
                if selected_keys is None or key in selected_keys:
                    if return_full:
                        per_key_chunks[key].append(value)
                    else:
                        if running_sums[key] is None:
                            running_sums[key] = value.sum(axis=0)
                        else:
                            running_sums[key] += value.sum(axis=0)

                for method in ['lsq', 'svd']:
                    inv_key = f"{key}-inv-{method}"
                    if selected_keys is not None and inv_key not in selected_keys:
                        continue
                    inv_value, dt = self._inverse(value, method=method)
                    if return_full:
                        per_key_chunks[inv_key].append(inv_value)
                    else:
                        if running_sums[inv_key] is None:
                            running_sums[inv_key] = inv_value.sum(axis=0)
                        else:
                            running_sums[inv_key] += inv_value.sum(axis=0)

                    time_inversion[method] += dt

            if not return_full:
                total_samples += batch_samples

            del jacobian_batch
            del reduced_batch

        if return_full:
            result = {
                key: np.concatenate(chunks, axis=0)
                for key, chunks in per_key_chunks.items()
            }
        else:
            result = {
                key: value / float(total_samples)
                for key, value in running_sums.items()
            }
            result['num_samples'] = int(total_samples)

        result['time_inversion_lsq'] = time_inversion['lsq']
        result['time_inversion_svd'] = time_inversion['svd']

        return result


@dataclasses.dataclass
@register("neuron-gradient")
class NeuronGradientMethod(AttributionMap):
    """Compute the attribution map using the neuron gradient from Captum.

    Note:
        This method is equivalent to Jacobian-based attributions, but
        uses a different backend implementation.
    """

    def __post_init__(self):
        super().__post_init__()
        from captum.attr import NeuronGradient

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

    def compute_attribution_map(self,
                                attribute_to_neuron_input=False,
                                batch_size=1024):
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
        return self._reduce_attribution_map({
            'neuron-gradient': attribution_map,
            #'neuron-gradient-invsvd': self._inverse_svd(attribution_map)
        })


@dataclasses.dataclass
@register("feature-ablation")
class FeatureAblationMethod(AttributionMap):
    """Compute the attribution map using the feature ablation method from Captum."""

    def __post_init__(self):
        super().__post_init__()
        from captum.attr import NeuronFeatureAblation

        self.captum_model = NeuronFeatureAblation(forward_func=self.model,
                                                  layer=self.model)

    def compute_attribution_map(self,
                                baselines=None,
                                feature_mask=None,
                                perturbations_per_eval=1,
                                attribute_to_neuron_input=False):
        attribution_map = []
        for s in range(self.output_dimension):
            att = self.captum_model.attribute(
                inputs=self.input_data,
                neuron_selector=s,
                baselines=baselines,
                perturbations_per_eval=perturbations_per_eval,
                feature_mask=feature_mask,
                attribute_to_neuron_input=attribute_to_neuron_input)

            attribution_map.append(att.detach().cpu().numpy())

        attribution_map = np.array(attribution_map)
        attribution_map = np.swapaxes(attribution_map, 1, 0)
        return self._reduce_attribution_map(
            {'feature-ablation': attribution_map})


@dataclasses.dataclass
@register("feature-ablation-batched")
class FeatureAblationMethodBAtched(FeatureAblationMethod):
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
    """Compute the attribution map using the integrated gradients method from Captum."""

    def __post_init__(self):
        super().__post_init__()
        from captum.attr import NeuronIntegratedGradients

        self.captum_model = NeuronIntegratedGradients(forward_func=self.model,
                                                      layer=self.model)

    def compute_attribution_map(self,
                                n_steps=50,
                                method='gausslegendre',
                                internal_batch_size=None,
                                attribute_to_neuron_input=False,
                                baselines=None):
        if internal_batch_size == "dataset":
            internal_batch_size = len(self.input_data)

        attribution_map = []
        for s in range(self.output_dimension):
            att = self.captum_model.attribute(
                inputs=self.input_data,
                neuron_selector=s,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
                attribute_to_neuron_input=attribute_to_neuron_input,
                baselines=baselines,
            )
            attribution_map.append(att.detach().cpu().numpy())

        attribution_map = np.array(attribution_map)
        attribution_map = np.swapaxes(attribution_map, 1, 0)
        return self._reduce_attribution_map(
            {'integrated-gradients': attribution_map})


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
    """Compute the attribution map using the neuron gradient SHAP method from Captum."""

    def __post_init__(self):
        super().__post_init__()
        from captum.attr import NeuronGradientShap

        self.captum_model = NeuronGradientShap(forward_func=self.model,
                                               layer=self.model)

    def compute_attribution_map(self,
                                baselines: str,
                                n_samples=5,
                                stdevs=0.0,
                                attribute_to_neuron_input=False):

        if baselines == "zeros":
            baselines = torch.zeros(size=(self.input_data.shape),
                                    device=self.input_data.device)
        elif baselines == "shuffle":
            data = self.input_data.flatten()
            data = data[torch.randperm(len(data))]
            baselines = data.reshape(self.input_data.shape)
        else:
            raise NotImplementedError(f"Baseline {baselines} not implemented.")

        attribution_map = []
        for s in range(self.output_dimension):
            att = self.captum_model.attribute(
                inputs=self.input_data,
                neuron_selector=s,
                baselines=baselines,
                n_samples=n_samples,
                stdevs=stdevs,
                attribute_to_neuron_input=attribute_to_neuron_input,
            )

            attribution_map.append(att.detach().cpu().numpy())

        attribution_map = np.array(attribution_map)
        attribution_map = np.swapaxes(attribution_map, 1, 0)
        return self._reduce_attribution_map(
            {'neuron-gradient-shap': attribution_map})


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
