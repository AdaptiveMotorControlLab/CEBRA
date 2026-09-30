import torch

import cebra.dynamics.linear as linear_dynamics


def test_linear_dynamics():
    latent_dim = 5
    model = linear_dynamics.Linear(latent_dim=latent_dim, bias=True)

    assert model.weight.shape == (latent_dim, latent_dim)
    assert model.bias.shape == (latent_dim,)

    x = torch.randn(10, latent_dim)
    out = model(x)
    assert out.shape == (10, latent_dim)


def test_orthogonal_linear_dynamics():
    latent_dim = 5
    model = linear_dynamics.OrthogonalLinear(latent_dim=latent_dim, bias=True)

    assert model.weight.shape == (latent_dim, latent_dim)
    assert model.bias.shape == (latent_dim,)

    x = torch.randn(10, latent_dim)
    out = model(x)
    assert out.shape == (10, latent_dim)

    W = model.weight.detach()
    WWT = W @ W.T
    identity = torch.eye(latent_dim)
    assert torch.allclose(WWT, identity, atol=1e-6)


def test_identity_dynamics():
    model = linear_dynamics.Identity()

    x = torch.randn(10, 5)
    out = model(x)
    assert torch.allclose(x, out)
    assert x.shape == out.shape
