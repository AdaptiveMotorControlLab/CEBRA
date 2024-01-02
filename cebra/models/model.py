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
"""Neural network models and criterions for training CEBRA models."""
import abc

import literate_dataclasses as dataclasses
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

import cebra.data
import cebra.data.datatypes
import cebra.models.layers as cebra_layers
from cebra.models import register


def _check_torch_version(raise_error=False):
    current_version = tuple(
        [int(i) for i in torch.__version__.split(".")[:2] if len(i) > 0])
    required_version = (1, 12)
    if current_version < required_version:
        if raise_error:
            raise ImportError(
                f"PyTorch < 1.12 is not supported for models using "
                f"Dropout1D, but got PyTorch={torch.__version__}.")
        else:
            return False
    return True


def _register_conditionally(*args, **kwargs):
    if _check_torch_version(raise_error=False):
        return register(*args, **kwargs)
    else:

        def do_nothing(cls):
            return cls

        return do_nothing


class Model(nn.Module):
    """Base model for CEBRA experiments.

    The model is a pytorch ``nn.Module``. Features can be computed by
    calling the ``forward()`` or ``__call__`` method. This class should not be
    directly instantiated, and instead used as the base class for CEBRA
    models.

    Args:
        num_input: The number of input dimensions. The tensor passed to
            the ``forward`` method will have shape ``(batch, num_input, in_time)``.
        num_output: The number of output dimensions. The tensor returned
            by the ``forward`` method will have shape ``(batch, num_output, out_time)``.
        offset: A specification of the offset to the left and right of
            the signal due to the network's receptive field. The offset specifies the
            relation between the input and output times, ``in_time - out_time = len(offset)``.

    Attributes:
        num_input: The input dimensionality (of the input signal). When calling
            ``forward``, this is the dimensionality expected for the input
            argument. In typical applications of CEBRA, the input dimension
            corresponds to the number of neurons for neural data analysis, number
            of keypoints for kinematik analysis, or can also be the dimension
            of a feature space in case preprocessing happened before feeding the
            data into the model.
        num_output: The output dimensionality (of the embedding space).
            This is the feature dimension of value returned by
            ``forward``. Note that for models using normalization,
            the output dimension should be at least 3D, and 2D without
            normalization to learn meaningful embeddings. The output
            dimensionality is typically smaller than :py:attr:`num_input`,
            but this is not enforced.
    """

    def __init__(
        self,
        *,
        num_input: int,
        num_output: int,
        offset: cebra.data.datatypes.Offset = None,
    ):
        super().__init__()
        if num_input < 1:
            raise ValueError(
                f"Input dimension needs to be at least 1, but got {num_input}.")
        if num_output < 1:
            raise ValueError(
                f"Output dimension needs to be at least 1, but got {num_output}."
            )
        self.num_input: int = num_input
        self.num_output: int = num_output

    @abc.abstractmethod
    def get_offset(self) -> cebra.data.datatypes.Offset:
        """Offset between input and output sequence caused by the receptive field.

        The offset specifies the relation between the length of the input and output
        time sequences. The output sequence is ``len(offset)`` steps shorter than the
        input sequence. For input sequences of shape ``(*, *, len(offset))``, the model
        should return an output sequence that drops the last dimension (which would be 1).

        Returns
            The offset of the network. See :py:class:`cebra.data.datatypes.Offset` for full
            documentation.
        """
        raise NotImplementedError()


class ConvolutionalModelMixin:
    """Mixin for models that support operating on a time-series.

    The input for convolutional models should be ``batch, dim, time``
    and the convolution will be applied across the last dimension.
    """

    pass


class ResampleModelMixin:
    """Mixin for models that re-sample the signal over time."""

    @property
    def resample_factor(self) -> float:
        """The factor by which the signal is downsampled."""
        return NotImplementedError()


class HasFeatureEncoder:
    """Networks with an explicitly defined feature encoder."""

    @property
    def feature_encoder(self) -> nn.Module:
        return self.net


class ClassifierModel(Model, HasFeatureEncoder):
    """Base model for classifiers.

    Adds an additional :py:attr:`classifier` layer to the model which is lazily
    initialized after calling :py:meth:`set_output_num`.

    Args:
        num_input: The number of input units
        num_output: The number of output units
        offset: The offset introduced by the model's receptive field

    Attributes:
        features_encoder: The feature encoder to map the input tensor (2d or 3d depending
            on the exact model implementation) into a feature space of same dimension
        classifier: Map from the feature space to class scores
    """

    def __init__(
        self,
        *,
        num_input: int,
        num_output: int,
        offset: cebra.data.datatypes.Offset = None,
    ):
        super().__init__(num_input=num_input, num_output=num_output)
        self.classifier: nn.Module = None

    @abc.abstractmethod
    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        raise NotImplementedError

    def set_output_num(self, label_num: int, override: bool = False):
        """Set the number of output classes.

        Args:
            label_num: The number of labels to be added to the classifier layer.
            override: If `True`, override an existing classifier layer. If you
                passed the parameters of this model to an optimizer, make sure
                to correctly handle the replacement of the classifier there.
        """
        if self.classifier is None or override:
            self.classifier = nn.Linear(self.num_output, label_num)
        else:
            raise RuntimeError("classifier is already initialized.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """See :py:class:`ClassifierModel`."""
        features = self.feature_encoder.forward(inputs)
        features = F.relu(features)
        prediction = self.classifier(features)
        return features, prediction


class _OffsetModel(Model, HasFeatureEncoder):

    def __init__(self,
                 *layers,
                 num_input=None,
                 num_output=None,
                 normalize=True):
        super().__init__(num_input=num_input, num_output=num_output)

        if normalize:
            layers += (cebra_layers._Norm(),)
        layers += (cebra_layers.Squeeze(),)
        self.net = nn.Sequential(*layers)
        # TODO(stes) can this layer be removed? it is already added to
        # the self.net
        self.normalize = normalize

    def forward(self, inp):
        """Compute the embedding given the input signal.

        Args:
            inp: The input tensor of shape `num_samples x self.num_input x time`

        Returns:
            The output tensor of shape `num_samples x self.num_output x (time - receptive field)`.

        Based on the parameters used for initializing, the output embedding
        is normalized to the hypersphere (`normalize = True`).
        """
        return self.net(inp)


class ParameterCountMixin:
    """Add a parameter counter to a torch.nn.Module."""

    @property
    def num_parameters(self) -> int:
        """Total parameter count of the model."""
        return sum(param.numel() for param in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(
            param.numel() for param in self.parameters() if param.requires_grad)


@register("offset10-model")
class Offset10Model(_OffsetModel, ConvolutionalModelMixin):
    """CEBRA model with a 10 sample receptive field."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(5, 5)


@register("offset10-model-mse")
class Offset10ModelMSE(Offset10Model):
    """Symmetric model with 10 sample receptive field, without normalization.

    Suitable for use with InfoNCE metrics for Euclidean space.
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=False):
        super().__init__(num_neurons, num_units, num_output, normalize)


@register("offset5-model")
class Offset5Model(_OffsetModel, ConvolutionalModelMixin):
    """CEBRA model with a 5 sample receptive field and output normalization."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 2),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(2, 3)


@register("offset1-model-mse")
class Offset0ModelMSE(_OffsetModel):
    """CEBRA model with a single sample receptive field, without output normalization."""

    def __init__(self, num_neurons, num_units, num_output, normalize=False):
        super().__init__(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons,
                num_output * 30,
            ),
            nn.GELU(),
            nn.Linear(num_output * 30, num_output * 30),
            nn.GELU(),
            nn.Linear(num_output * 30, num_output * 10),
            nn.GELU(),
            nn.Linear(int(num_output * 10), num_output),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(0, 1)


@register("offset1-model")
class Offset0Model(_OffsetModel):
    """CEBRA model with a single sample receptive field, with output normalization."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 2:
            raise ValueError(
                f"Number of hidden units needs to be at least 2, but got {num_units}."
            )
        super().__init__(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons,
                num_units,
            ),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
            nn.Linear(num_units, int(num_units // 2)),
            nn.GELU(),
            nn.Linear(int(num_units // 2), num_output),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(0, 1)


@register("offset1-model-v2")
class Offset0Modelv2(_OffsetModel):
    """CEBRA model with a single sample receptive field, with output normalization.

    This is a variant of :py:class:`Offset0Model`.
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 2:
            raise ValueError(
                f"Number of hidden units needs to be at least 2, but got {num_units}."
            )
        super().__init__(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons,
                num_units,
            ),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_output),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(0, 1)


@register("offset1-model-v3")
class Offset0Modelv3(_OffsetModel):
    """CEBRA model with a single sample receptive field, with output normalization.

    This is a variant of :py:class:`Offset0Model`.
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 2:
            raise ValueError(
                f"Number of hidden units needs to be at least 2, but got {num_units}."
            )
        super().__init__(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons,
                num_units,
            ),
            nn.GELU(),
            cebra_layers._Skip(nn.Linear(num_units, num_units), crop=(0, None)),
            nn.GELU(),
            cebra_layers._Skip(nn.Linear(num_units, num_units), crop=(0, None)),
            nn.GELU(),
            nn.Linear(num_units, num_output),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(0, 1)


@register("offset1-model-v4")
class Offset0Modelv4(_OffsetModel):
    """CEBRA model with a single sample receptive field, with output normalization.

    This is a variant of :py:class:`Offset0Model`.
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 2:
            raise ValueError(
                f"Number of hidden units needs to be at least 2, but got {num_units}."
            )
        super().__init__(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons,
                num_units,
            ),
            nn.GELU(),
            cebra_layers._Skip(nn.Linear(num_units, num_units),
                               nn.GELU(),
                               crop=(0, None)),
            cebra_layers._Skip(nn.Linear(num_units, num_units),
                               nn.GELU(),
                               crop=(0, None)),
            cebra_layers._Skip(nn.Linear(num_units, num_units),
                               nn.GELU(),
                               crop=(0, None)),
            cebra_layers._Skip(nn.Linear(num_units, num_units),
                               nn.GELU(),
                               crop=(0, None)),
            nn.Linear(num_units, num_output),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(0, 1)


@register("offset1-model-v5")
class Offset0Modelv5(_OffsetModel):
    """CEBRA model with a single sample receptive field, with output normalization.

    This is a variant of :py:class:`Offset0Model`.
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 2:
            raise ValueError(
                f"Number of hidden units needs to be at least 2, but got {num_units}."
            )
        super().__init__(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons,
                num_units,
            ),
            nn.GELU(),
            cebra_layers._Skip(nn.Linear(num_units, num_units), crop=(0, None)),
            nn.GELU(),
            cebra_layers._Skip(nn.Linear(num_units, num_units), crop=(0, None)),
            nn.GELU(),
            cebra_layers._Skip(nn.Linear(num_units, num_units), crop=(0, None)),
            nn.GELU(),
            cebra_layers._Skip(nn.Linear(num_units, num_units), crop=(0, None)),
            nn.GELU(),
            nn.Linear(num_units, num_output),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(0, 1)


@register("resample-model",
          deprecated=True)  # NOTE(stes) deprecated name for compatibility
@register("offset40-model-4x-subsample")
class ResampleModel(_OffsetModel, ConvolutionalModelMixin, ResampleModelMixin):
    """CEBRA model with 40 sample receptive field, output normalization and 4x subsampling."""

    ##120Hz
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            cebra_layers._MeanAndConv(num_neurons, num_units, 4, stride=2),
            nn.Conv1d(num_neurons + num_units, num_units, 3, stride=2),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    @property
    def resample_factor(self):
        return 4

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(20, 20)


@register("resample5-model", deprecated=True)
@register("offset20-model-4x-subsample")
class Resample5Model(_OffsetModel, ConvolutionalModelMixin, ResampleModelMixin):
    """CEBRA model with 20 sample receptive field, output normalization and 4x subsampling."""

    ##120Hz
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            cebra_layers._MeanAndConv(num_neurons, num_units, 4, stride=2),
            nn.Conv1d(num_neurons + num_units, num_units, 3, stride=2),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 2),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    @property
    def resample_factor(self):
        return 4

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(10, 10)


@register("resample1-model", deprecated=True)
@register("offset4-model-2x-subsample")
class Resample1Model(_OffsetModel, ResampleModelMixin):
    """CEBRA model with 4 sample receptive field, output normalization and 2x subsampling.

    This model is not convolutional, and needs to be applied to fixed ``(N, d, 4)`` inputs.
    """

    ##120Hz
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            cebra_layers._MeanAndConv(num_neurons, num_units, 4, stride=2),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons + num_units,
                num_units,
            ),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
            nn.Linear(num_units, int(num_units // 2)),
            nn.GELU(),
            nn.Linear(int(num_units // 2), num_output),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    @property
    def resample_factor(self):
        return 2

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(2, 2)


@register("supervised10-model")
class SupervisedNN10(ClassifierModel):
    """A supervised model with 10 sample receptive field."""

    def __init__(self, num_neurons, num_units, num_output):
        super(SupervisedNN10, self).__init__(num_input=num_neurons,
                                             num_output=num_output)

        self.net = nn.Sequential(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            cebra_layers.Squeeze(),
        )
        self.num_output = num_output

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(5, 5)


@register("supervised1-model")
class SupervisedNN1(ClassifierModel):
    """A supervised model with single sample receptive field."""

    def __init__(self, num_neurons, num_units, num_output):
        super(SupervisedNN1, self).__init__(num_input=num_neurons,
                                            num_output=num_output)

        self.net = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_neurons,
                num_units,
            ),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
            nn.Linear(num_units, int(num_units // 2)),
            nn.GELU(),
            nn.Linear(int(num_units // 2), num_output),
            cebra_layers.Squeeze(),
        )
        self.num_output = num_output

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(0, 1)


@register("offset36-model")
class Offset36(_OffsetModel, ConvolutionalModelMixin):
    """CEBRA model with a 10 sample receptive field."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See `:py:meth:Model.get_offset`"""
        return cebra.data.Offset(18, 18)


@_register_conditionally("offset36-model-dropout")
class Offset36Dropout(_OffsetModel, ConvolutionalModelMixin):
    """CEBRA model with a 10 sample receptive field.

    Note:
        Requires ``torch>=1.12``.
    """

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            torch.nn.Dropout1d(p=0.1),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See `:py:meth:Model.get_offset`"""
        return cebra.data.Offset(18, 18)


@_register_conditionally("offset36-model-more-dropout")
class Offset36Dropoutv2(_OffsetModel, ConvolutionalModelMixin):
    """CEBRA model with a 10 sample receptive field.

    Note:
        Requires ``torch>=1.12``.
    """

    def _make_layers(self, num_units, p, n):
        return [
            cebra_layers._Skip(torch.nn.Dropout1d(p=p),
                               nn.Conv1d(num_units, num_units, 3), nn.GELU())
            for _ in range(n)
        ]

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            torch.nn.Dropout1d(p=0.1),
            nn.GELU(),
            *self._make_layers(num_units, 0.1, 16),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See `:py:meth:Model.get_offset`"""
        return cebra.data.Offset(18, 18)
