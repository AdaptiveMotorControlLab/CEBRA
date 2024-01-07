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
import matplotlib
import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

import cebra.integrations.plotly as cebra_plotly
import cebra.integrations.sklearn.cebra as cebra_sklearn_cebra


@pytest.mark.parametrize("cmap", ["viridis", "plasma", "inferno", "magma"])
def test_colorscale(cmap):
    cmap = matplotlib.cm.get_cmap(cmap)
    colorscale = cebra_plotly._convert_cmap2colorscale(cmap)
    assert isinstance(colorscale, list)


@pytest.mark.parametrize("output_dimension, idx_order", [(8, (2, 3, 4)),
                                                         (3, (0, 1, 2))])
def test_plot_embedding(output_dimension, idx_order):
    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    y = np.random.uniform(0, 1, (1000,))

    # integration tests
    model = cebra_sklearn_cebra.CEBRA(max_iterations=10,
                                      batch_size=512,
                                      output_dimension=output_dimension)

    model.fit(X)
    embedding = model.transform(X)

    fig = cebra_plotly.plot_embedding_interactive(embedding=embedding,
                                                  embedding_labels=y)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    fig.layout = {}
    fig.data = []

    fig_subplots = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{
                "type": "scatter3d"
            }, {
                "type": "scatter3d"
            }],
            [{
                "type": "scatter3d"
            }, {
                "type": "scatter3d"
            }],
        ],
    )

    fig_subplots = cebra_plotly.plot_embedding_interactive(axis=fig_subplots,
                                                           embedding=embedding,
                                                           embedding_labels=y,
                                                           row=1,
                                                           col=1)

    fig_subplots.data = []
    fig_subplots.layout = {}


def test_discrete_with_legend():
    embedding = np.random.uniform(0, 1, (1000, 3))
    labels = np.random.randint(0, 10, (1000,))

    fig = cebra_plotly.plot_embedding_interactive(embedding,
                                                  labels,
                                                  discrete=True,
                                                  showlegend=True)

    assert len(fig._data_objs) == np.unique(labels).shape[0]
    assert isinstance(fig, go.Figure)


def test_continuous_no_legend():
    embedding = np.random.uniform(0, 1, (1000, 3))
    labels = np.random.uniform(0, 1, (1000,))

    fig = cebra_plotly.plot_embedding_interactive(embedding, labels)

    assert len(fig._data_objs) == 1

    assert isinstance(fig, go.Figure)


def test_continuous_with_legend_raises_error():
    embedding = np.random.uniform(0, 1, (1000, 3))
    labels = np.random.uniform(0, 1, (1000,))

    with pytest.raises(ValueError):
        cebra_plotly.plot_embedding_interactive(embedding,
                                                labels,
                                                discrete=False,
                                                showlegend=True)
