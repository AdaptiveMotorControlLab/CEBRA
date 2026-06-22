from typing import Literal

import torch

from cebra.dynamics import register


@register("identity")
class Identity(torch.nn.Identity):
    pass


@register("linear")
class Linear(torch.nn.Linear):

    def __init__(self, latent_dim: int, bias: bool = True):
        super().__init__(latent_dim, latent_dim, bias)


@register("orthogonal-linear")
class OrthogonalLinear(Linear):
    """
    A LinearDynamicsModel that is parametrized to only allow orthogonal dynamics matrices.
    """

    def __init__(
        self,
        latent_dim: int,
        bias: bool = True,
        orthogonal_map: Literal[
            "matrix_exp",
            "cayley",
            "householder",
        ] = "matrix_exp",
        use_trivialization: bool = True,
    ):
        super().__init__(latent_dim, bias)
        self.orthogonal_map = orthogonal_map
        self.use_trivialization = use_trivialization

        torch.nn.utils.parametrizations.orthogonal(
            self,
            name="weight",
            orthogonal_map=self.orthogonal_map,
            use_trivialization=self.use_trivialization,
        )
