#
# Regularized contrastive learning implementation.
#
# Not licensed yet. Distribution for review.
# Code will be open-sourced upon publication.
#
import dataclasses
import sys
import time

import numpy as np
import scipy.linalg
import sklearn.metrics
import torch
import torch.nn as nn
from captum.attr import NeuronFeatureAblation
from captum.attr import NeuronGradient
from captum.attr import NeuronGradientShap
from captum.attr import NeuronIntegratedGradients

import cebra
import cebra.attribution.jacobian
from cebra.attribution import register


@dataclasses.dataclass
class AttributionMap:
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
        raise NotImplementedError

    def compute_metrics(self, attribution_map, ground_truth_map):
        # Note: 0: nonconnected, 1: connected
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
        assert attribution_map.shape == ground_truth_map.shape
        assert np.issubdtype(ground_truth_map.dtype, bool)
        fpr, tpr, _ = sklearn.metrics.roc_curve(ground_truth_map.flatten(),
                                                attribution_map.flatten())
        auc = sklearn.metrics.auc(fpr, tpr)
        return auc

    @staticmethod
    def _check_moores_penrose_conditions(
            matrix: np.ndarray, matrix_inverse: np.ndarray) -> np.ndarray:
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
        """
        Checks the four conditions for the Moore-Penrose conditions for the
        pseudo-inverse of a matrix.
        Args:
            jacobian: The Jacobian matrix of dhape (num samples, output_dim, num_neurons).
            jacobian_pseudoinverse: The pseudo-inverse of the Jacobian matrix of shape (num samples, num_neurons, output_dim).
        Returns:
            moores_penrose_conditions: A boolean array of shape (num samples, 4) where each row corresponds to a sample and each column to a condition.
        """
        # check the four conditions
        conditions = np.zeros((jacobian.shape[0], 4))
        for i, (matrix, inverse_matrix) in enumerate(
                zip(jacobian, jacobian_pseudoinverse)):
            conditions[i] = self._check_moores_penrose_conditions(
                matrix, inverse_matrix)
        return conditions

    def _inverse(self, jacobian, method="lsq"):
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
        """
        Solves the least squares problem
        min ||A @ X - I||_2 = (A @ X - I, A @ X - I) = (A @ X)**2 - 2 * (A @ X, I) + (I, I) =
        = (A @ X)**2 - 2 * (A @ X, I) + const -> min quadratic function of X
        """

        matrix_param = cp.Parameter((matrix.shape[0], matrix.shape[1]))
        matrix_param.value = matrix

        I = np.eye(matrix.shape[0])
        matrix_inverse = cp.Variable((matrix.shape[1], matrix.shape[0]))

        objective = cp.Minimize(cp.norm(matrix @ matrix_inverse - I, "fro"))
        prob = cp.Problem(objective)
        prob.solve(verbose=False, solver=solver)

        return matrix_inverse.value

    @staticmethod
    def _inverse_lsq_scipy(jacobian):
        return scipy.linalg.lstsq(jacobian, np.eye(jacobian.shape[0]))[0]

    @staticmethod
    def _inverse_svd(jacobian):
        return scipy.linalg.pinv(jacobian)

    def _reduce_attribution_map(self, attribution_maps):

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

    def _compute_jacobian(self, input_data):
        return cebra.attribution.jacobian.compute_jacobian(
            self.model,
            input_vars=[input_data],
            mode="autograd",
            cuda_device=self.input_data.device,
            double_precision=False,
            convert_to_numpy=True,
            hybrid_solver=False,
        )

    def compute_attribution_map(self):

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

    def compute_attribution_map(self, batch_size=1024):
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
                # result[f"{key}-inv-{method}-conditions"] = self.check_moores_penrose_conditions(value, result[f"{key}-inv-{method}"])

        return result


@dataclasses.dataclass
@register("neuron-gradient")
class NeuronGradientMethod(AttributionMap):

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

    def __post_init__(self):
        super().__post_init__()
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

    def __post_init__(self):
        super().__post_init__()
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

    def __post_init__(self):
        super().__post_init__()
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
