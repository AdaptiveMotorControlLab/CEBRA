"""Generate synthetic datasets for benchmarking embedding quality.

References:
    Adapted from pi-VAE: https://github.com/zhd96/pi-vae/blob/main/code/pi_vae.py
"""
import poisson
import third_party.pivae.pivae_code.pi_vae as pi_vae

        length: The length of the simulation or the number of the simulated samples.
        n_dim: The dimension of the observation.
        noise_func: The distribution used for generative process
    lam_true = np.exp(2.2 * np.tanh(mean_true))



    """Apply poisson noise to the input.

    This setup corresponds to the synthetic dataset originally
    considered by Zhou and Wei (NeurIPS 2021) for pi-VAE benchmarking.

    Args:
        x: The rate parameter

    Returns:
        Samples with the specified rate, of same shape as the input.
    """


    """Apply truncated Gaussian noise with unit variance to the input.

    Args:
        x: The mean

    Returns:
        The samples, of same shape as the input.
    """
    return scipy.stats.truncnorm.rvs(0, 1000, loc=x)


@_register_noise
    """Apply (post-truncated) Laplace noise to the input.

    Args:
        x: The mean

    Returns:
        The samples, of same shape as the input.
    """
    return np.clip(np.random.laplace(x), a_min=0, a_max=1000)


@_register_noise
    """Apply uniform noise from [0, 2) to the input.

    Args:
        x: The offset added to the noise samples.

    Returns:
        The samples, of same shape as the input.
    """
    return np.random.uniform(0, 2, x.shape) + x


@_register_noise
    """Apply student-t distributed noise to the input.

    Args:
        x: The mean

    Returns:
        The samples, of same shape as the input.
    """
    return scipy.stats.distributions.t.rvs(2, loc=x)


@_register_noise
    """TODO. Not implemented yet."""
    raise NotImplementedError()
    parser.add_argument("--neurons",
                        type=int,
                        default=100,
    parser.add_argument(
        "--noise",
        type=str,
        choices=list(__noises.keys()),
    parser.add_argument(
        "--scale",
        type=float,
        default=50,
        help=
    )
    parser.add_argument(
        "--time-interval",
        type=float,
        default=3,
        help=
    )

    args = parser.parse_args()

        z_true, u_true, mean_true, lam_true, x = simulate_cont_data_diff_var(
            args.n_samples, args.neurons, func)
        z_true, u_true, mean_true, lam_true = simulate_cont_data_diff_var(
            args.n_samples, args.neurons, None)
            count = neuron._get_counts(refractory_period=args.refractory_period)
        x = x.reshape(lam_true.shape)
