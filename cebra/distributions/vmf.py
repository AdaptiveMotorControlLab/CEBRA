"""
Generate multivariate von Mises Fisher samples.
This solution originally appears here:
http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
Also see:
Sampling from vMF on S^2:
    https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    http://www.stat.pitt.edu/sungkyu/software/randvonMisesFisher3.pdf
This code was taken from the following project:
https://github.com/clara-labs/spherecluster
"""
import numpy as np


__all__ = ["sample_vMF", "sample_vMF_sequential"]


def sample_vMF_sequential(mu, kappa, num_samples):
    """Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu \in R^N with concentration kappa.
    """
    if len(mu.shape) == 1:
        mu = mu.reshape(1, -1)

    assert len(mu.shape) == 2
    dim = mu.shape[1]
    # assert len(mu) == num_samples
    result = np.zeros((num_samples, dim))

    for nn in range(num_samples):
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight_sequential(kappa, dim)

        if len(mu) == 1:
            n_mu = mu[0]
        else:
            n_mu = mu[nn]

        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal_to_sequential(n_mu)

        # compute new point
        result[nn, :] = v * np.sqrt(1.0 - w ** 2) + w * n_mu

    return result


def sample_vMF(mu, kappa, num_samples):
    """Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu \in R^N with concentration kappa.
    """
    if len(mu.shape) == 1:
        mu = mu.reshape(1, -1)
        mu = np.repeat(mu, num_samples, 0)

    assert len(mu.shape) == 2
    dim = mu.shape[1]

    # sample offset from center (on sphere) with spread kappa
    w = _sample_weight(kappa, dim, num_samples)

    # sample a point v on the unit sphere that's orthogonal to mu
    v = _sample_orthonormal_to(mu)

    # compute new point
    result = v * np.sqrt(1.0 - w ** 2).reshape(-1, 1) + w.reshape(-1, 1) * mu

    return result


def _sample_weight_sequential(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + dim * np.log(1 - x ** 2)

    while True:
        z = np.random.beta(dim / 2.0, dim / 2.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
            return w


def _sample_weight(kappa, dim, num_samples):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + dim * np.log(1 - x ** 2)

    results = []
    n = 0

    while True:
        z = np.random.beta(dim / 2.0, dim / 2.0, size=num_samples)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = np.random.uniform(low=0, high=1, size=num_samples)

        mask = kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u)
        results.append(w[mask])
        n += sum(mask)

        if n >= num_samples:
            break

    results = np.concatenate(results)[:num_samples]

    return results


def _sample_orthonormal_to_sequential(mu):
    """Sample point on sphere orthogonal to mu."""
    v = np.random.randn(mu.shape[0])
    proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto)


def _sample_orthonormal_to(mu):
    """Sample point on sphere orthogonal to mu."""
    v = np.random.randn(mu.shape[0], mu.shape[1])
    proj_mu_v = (
        mu
        * np.einsum("ij,ij -> i", mu, v).reshape(-1, 1)
        / np.linalg.norm(mu, axis=-1, keepdims=True)
    )
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto, axis=-1, keepdims=True)