import torch.nn as nn

from cebra.models import register


@register("one-layer-mlp-decoder")
class SingleLayerDecoder(nn.Module):
    """Supervised module to predict behaviors.

    Note:
        By default, the output dimension is 2, to predict x/y velocity
        (Perich et al., 2018).
    """

    def __init__(self, input_dim, output_dim=2):
        super(SingleLayerDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


@register("two-layers-mlp-decoder")
class TwoLayersDecoder(nn.Module):
    """Supervised module to predict behaviors.

    Note:
        By default, the output dimension is 2, to predict x/y velocity
        (Perich et al., 2018).
    """

    def __init__(self, input_dim, output_dim=2):
        super(TwoLayersDecoder, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(),
                                nn.Linear(32, output_dim))

    def forward(self, x):
        return self.fc(x)
