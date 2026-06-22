#!/usr/bin/env python3
"""
Example script demonstrating how to use CEBRA for time contrastive learning.
"""

import sklearn.decomposition
import torch

import cebra
import cebra.datasets

if __name__ == "__main__":

    device = "mps"

    input_data = cebra.datasets.init("rat-hippocampus-single-achilles")

    latent_dim = 5

    neural_model = cebra.models.init(
        name="offset10-model-mse",
        num_neurons=input_data.input_dimension,
        num_units=32,
        num_output=latent_dim,
    ).to(device)

    dynamics_model = cebra.dynamics.init(
        name="linear",
        latent_dim=latent_dim,
        bias=True,
    ).to(device)

    input_data.configure_for(neural_model)

    crit = cebra.models.criterions.FixedEuclideanInfoNCE(
        temperature=1,
        full_denominator=True,
    ).to(device)

    opt = torch.optim.Adam(list(neural_model.parameters()) +
                           list(dynamics_model.parameters()) +
                           list(crit.parameters()),
                           lr=0.001,
                           weight_decay=0)

    solver = cebra.solver.init(
        name="single-session-dcl",
        model=neural_model,
        criterion=crit,
        optimizer=opt,
        tqdm_on=True,
        dynamics_model=dynamics_model,
    ).to(device)

    loader = cebra.data.single_session.ContinuousDataLoader(
        dataset=input_data,
        num_steps=500,
        batch_size=2048,
        batch_size_negatives=10000,
        conditional="time",
        time_offset=10,
    ).to(device)

    solver.fit(loader=loader)

    x_train_emb = solver.transform(input_data.neural)

    ica = sklearn.decomposition.FastICA(n_components=2)
    x_train_emb = ica.fit_transform(x_train_emb.cpu())

    ax = cebra.plot_embedding(
        x_train_emb,
        embedding_labels=input_data.continuous_index[:, 0].cpu(),
        markersize=10,
        cmap="rainbow")

    ax.figure.savefig("dcl-example.png")
