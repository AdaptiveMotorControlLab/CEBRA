#!/usr/bin/env python3
"""
Example script demonstrating how to use CEBRA DCL with the sklearn API.
This reproduces the functionality of dcl_example.py using the sklearn wrapper.
"""

import sklearn.decomposition

import cebra
import cebra.datasets

if __name__ == "__main__":
    device = "mps"

    input_data = cebra.datasets.init("rat-hippocampus-single-achilles")

    latent_dim = 5

    # Create DCL model using sklearn API
    dcl_model = cebra.CEBRA(
        model_architecture="offset10-model-mse",
        output_dimension=latent_dim,
        num_hidden_units=32,
        batch_size=4096,
        learning_rate=0.001,
        max_iterations=500,
        time_offsets=10,
        conditional="time",
        distance="euclidean",
        temperature=1.0,
        device=device,
        verbose=True,
        dynamics_model_architecture="linear",
        full_denominator=True,  # Match the PyTorch API example
    )

    # Fit the model
    dcl_model.fit(input_data.neural.cpu().numpy())

    # Transform to get embeddings
    x_train_emb = dcl_model.transform(input_data.neural.cpu().numpy())

    ica = sklearn.decomposition.FastICA(n_components=2)
    x_train_emb = ica.fit_transform(x_train_emb)

    ax = cebra.plot_embedding(
        x_train_emb,
        embedding_labels=input_data.continuous_index[:, 0].cpu(),
        markersize=10,
        cmap="rainbow")

    ax.figure.savefig("dcl-example-sklearn.png")
