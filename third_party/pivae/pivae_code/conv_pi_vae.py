"""Implementation of conv-piVAE.
Adapted from https://github.com/zhd96/pi-vae/blob/3d31fae18bf9dda5290ac74435296b94eb05f854/code/pi_vae.py.
"""
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras.models import Model
from keras import losses
from keras.layers.core import Lambda
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.activations import softplus
import numpy as np
from keras.callbacks import LearningRateScheduler
from . import pi_vae

eps = 1e-7

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


get_custom_objects().update({"custom_gelu": layers.Activation(custom_gelu)})
squeeze_func = Lambda(lambda x: K.squeeze(x, 1))

def decode_nflow_func_conv(z_input, n_blk, dim_x, time_window, mdl, gen_nodes=None):
    permute_ind = []
    # n_blk = 5;
    for ii in range(n_blk):
        np.random.seed(ii)
        permute_ind.append(
            tf.convert_to_tensor(np.random.permutation(dim_x * time_window))
        )

    output = pi_vae.first_nflow_layer(z_input, dim_x * time_window)
    for ii in range(n_blk):
        output = Lambda(pi_vae.perm_func, arguments={"ind": permute_ind[ii]})(output)
        output = pi_vae.affine_coupling_block(output, gen_nodes)

    if mdl == "poisson":
        softplus_func = Lambda(lambda x: softplus(x))
        output = softplus_func(output)
    output = layers.Reshape((time_window, dim_x), input_shape=(dim_x * time_window,))(
        output
    )
    return output

def encode_func_conv(x_input, gen_nodes, dim_z):
    n_nodes = [32, 32, 32]
    kernels = [3, 3, 3]
    n_layers = len(n_nodes)
    output = x_input
    output = layers.Conv1D(
        filters=32,
        kernel_size=2,
        activation=custom_gelu,
        input_shape=x_input.shape[1:],
    )(x_input)
    for i in range(n_layers):
        _output = layers.Conv1D(
            n_nodes[i],
            kernels[i],
            activation=custom_gelu,
        )(output)
        _skip = layers.Cropping1D(cropping=(1, 1))(output)
        output = layers.Add()([_skip, _output])
    output = layers.Conv1D(
        dim_z,
        3,
        activation=None,
    )(output)
    output = squeeze_func(output)
    return output

def conv_vae_mdl(
    dim_x,
    dim_z,
    dim_u,
    time_window,
    gen_nodes,
    n_blk=None,
    mdl="poisson",
    disc=True,
    learning_rate=5e-4,
):
    ### discrete u, or continuous u (one-hot) as input

    ### input layer
    x_input = layers.Input(shape=(time_window, dim_x))
    z_input = layers.Input(shape=(dim_z,))

    ## conditional distribution of z given u # 2*dim_z if z follows gaussian
    if disc:
        u_input = layers.Input(shape=(1,))
        lam_mean, lam_log_var = pi_vae.z_prior_disc(u_input, dim_z, dim_u)
    else:
        u_input = layers.Input(shape=(dim_u,))
        lam_mean, lam_log_var = pi_vae.z_prior_nn(u_input, 2 * dim_z)
    
    ### encoder model
    z_mean = encode_func_conv(x_input, gen_nodes, dim_z)
    z_log_var = encode_func_conv(x_input, gen_nodes, dim_z)

    post_mean, post_log_var = Lambda(pi_vae.compute_posterior)(
        [z_mean, z_log_var, lam_mean, lam_log_var]
    )
    z_sample = Lambda(pi_vae.sampling)([post_mean, post_log_var])
    encoder = Model(
        inputs=[x_input, u_input],
        outputs=[
            post_mean,
            post_log_var,
            z_sample,
            z_mean,
            z_log_var,
            lam_mean,
            lam_log_var,
        ],
        name="encoder",
    )

    ### decoder model
    if n_blk is not None:  # use nflow
        fire_rate = decode_nflow_func_conv(z_input, n_blk, dim_x, time_window, mdl)

    if mdl == "poisson":
        clip_func = Lambda(lambda x: K.clip(x, min_value=1e-7, max_value=1e7))
        fire_rate = clip_func(fire_rate)

    decoder = Model(inputs=[z_input], outputs=[fire_rate], name="decoder")

    ### define vae and loss func
    ### vae model
    (
        post_mean,
        post_log_var,
        z_sample,
        z_mean,
        z_log_var,
        lam_mean,
        lam_log_var,
    ) = encoder([x_input, u_input])
    fire_rate = decoder([z_sample])
    if mdl == "gaussian":
        one_tensor = layers.Input(tensor=(tf.ones((1, 1))))
        obs_log_var = layers.Dense(
            dim_x, activation="linear", use_bias=False, name="obs_noise"
        )(one_tensor)
        vae = Model(
            inputs=[x_input, u_input, one_tensor],
            outputs=[
                post_mean,
                post_log_var,
                z_sample,
                fire_rate,
                lam_mean,
                lam_log_var,
                z_mean,
                z_log_var,
                obs_log_var,
            ],
            name="vae",
        )
    elif mdl == "poisson":
        vae = Model(
            inputs=[x_input, u_input],
            outputs=[
                post_mean,
                post_log_var,
                z_sample,
                fire_rate,
                lam_mean,
                lam_log_var,
                z_mean,
                z_log_var,
            ],
            name="vae",
        )

    ### objective function
    # min -log p(x|z) + E_q log(q(z))-log(p(z|u))
    # cross entropy
    # q (mean1, var1) p (mean2, var2)
    # E_q log(q(z))-log(p(z|u)) = -0.5*(1-log(var2/var1) - (var1+(mean2-mean1)^2)/var2)
    # E_q(z|x,u) log(q(z|x,u))-log(p(z|u)) = -0.5*(log(2*pi*var2) + (var1+(mean2-mean1)^2)/var2)
    # p(z) = q(z|x) = N(f(x), g(x)) parametrized by nn;

    if mdl == "poisson":
        obs_loglik = K.sum(fire_rate - x_input * tf.math.log(fire_rate), axis=[1, 2])
    elif mdl == "gaussian":
        obs_loglik = K.sum(
            K.square(fire_rate - x_input) / (2 * tf.exp(obs_log_var))
            + (obs_log_var / 2),
            axis=-1,
        )

    kl_loss = (
        1
        + post_log_var
        - lam_log_var
        - ((K.square(post_mean - lam_mean) + K.exp(post_log_var)) / K.exp(lam_log_var))
    )
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(obs_loglik + kl_loss)
    vae.add_loss(vae_loss)
    # learning_rate = 5e-4;
    optimizer = Adam(lr=learning_rate)
    vae.compile(optimizer=optimizer)

    print(vae.summary())
    return vae
