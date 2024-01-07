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
"""Plotly interface to CEBRA."""
from typing import Optional, Tuple, Union

import matplotlib.colors
import numpy as np
import numpy.typing as npt
import plotly.graph_objects
import torch

from cebra.integrations.matplotlib import _EmbeddingPlot


def _convert_cmap2colorscale(cmap: str, pl_entries: int = 11, rdigits: int = 2):
    """Convert matplotlib colormap to plotly colorscale.

    Args:
        cmap: A registered colormap name from matplotlib.
        pl_entries: Number of colors to use in the plotly colorscale.
        rdigits: Number of digits to round the colorscale to.

    Returns:
        pl_colorscale: List of scaled colors to plot the embeddings
    """
    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3] * 255).astype(np.uint8)
    pl_colorscale = [[round(s, rdigits), f"rgb{tuple(color)}"]
                     for s, color in zip(scale, colors)]
    return pl_colorscale


class _EmbeddingInteractivePlot(_EmbeddingPlot):

    def __init__(self, **kwargs):
        self.figsize = kwargs.get("figsize")
        super().__init__(**kwargs)
        self.colorscale = self._define_colorscale(self.cmap)

    def _define_ax(self, axis: Optional[plotly.graph_objects.Figure]):
        """Define the axis of the plot.

        Args:
            axis: Optional axis to create the plot on.

        Returns:
            axis: The axis :py:meth:`plotly.graph_objs._figure.Figure` of the plot.
        """

        if axis is None:
            self.axis = plotly.graph_objects.Figure(
                layout=plotly.graph_objects.Layout(height=100 * self.figsize[0],
                                                   width=100 * self.figsize[1]))

        else:
            self.axis = axis

    def _define_colorscale(self, cmap: str):
        """Specify the cmap for plotting the latent space.

        Args:
            cmap: The Colormap instance or registered colormap name used to map scalar data to colors. It will be ignored if `embedding_labels` is set to a valid RGB(A).


        Returns:
            colorscale: List of scaled colors to plot the embeddings
        """
        colorscale = _convert_cmap2colorscale(matplotlib.cm.get_cmap(cmap))

        return colorscale

    def _plot_3d(self, **kwargs) -> plotly.graph_objects.Figure:
        """Plot the embedding in 3d.

        Returns:
            The axis :py:meth:`plotly.graph_objs._figure.Figure` of the plot.
        """
        showlegend = kwargs.get("showlegend", False)
        discrete = kwargs.get("discrete", False)
        col = kwargs.get("col", None)
        row = kwargs.get("row", None)
        template = kwargs.get("template", "plotly_white")
        data = []

        if not discrete and showlegend:
            raise ValueError("Cannot show legend with continuous labels.")

        idx1, idx2, idx3 = self.idx_order

        if discrete:
            unique_labels = np.unique(self.embedding_labels)
        else:
            unique_labels = [self.embedding_labels]

        for label in unique_labels:
            if discrete:
                filtered_idx = [
                    i for i, x in enumerate(self.embedding_labels) if x == label
                ]
            else:
                filtered_idx = np.arange(self.embedding.shape[0])
            data.append(
                plotly.graph_objects.Scatter3d(x=self.embedding[filtered_idx,
                                                                idx1],
                                               y=self.embedding[filtered_idx,
                                                                idx2],
                                               z=self.embedding[filtered_idx,
                                                                idx3],
                                               mode="markers",
                                               marker=dict(
                                                   size=self.markersize,
                                                   opacity=self.alpha,
                                                   color=label,
                                                   colorscale=self.colorscale,
                                               ),
                                               name=str(label)))

        for trace in data:
            if col is None or row is None:
                self.axis.add_trace(trace)
            else:
                self.axis.add_trace(trace, row=row, col=col)

        self.axis.update_layout(
            template=template,
            showlegend=showlegend,
            title=self.title,
        )

        return self.axis


def plot_embedding_interactive(
    embedding: Union[npt.NDArray, torch.Tensor],
    embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]] = "grey",
    axis: Optional[plotly.graph_objects.Figure] = None,
    markersize: float = 1,
    idx_order: Optional[Tuple[int]] = None,
    alpha: float = 0.4,
    cmap: str = "cool",
    title: str = "Embedding",
    figsize: Tuple[int] = (5, 5),
    dpi: int = 100,
    **kwargs,
) -> plotly.graph_objects.Figure:
    """Plot embedding in a 3D dimensional space.

    This is supposing that the dimensions provided to ``idx_order`` are in the range of the number of
    dimensions of the embedding (i.e., between 0 and :py:attr:`cebra.CEBRA.output_dimension` -1).

    The function makes use of :py:class:`plotly.graph_objects.Scatter` and parameters from that function can be provided
    as part of ``kwargs``.


    Args:
        embedding: A matrix containing the feature representation computed with CEBRA.
        embedding_labels: The labels used to map the data to color. It can be:

            * A vector that is the same sample size as the embedding, associating a value to each of the sample, either discrete or continuous.
            * A string, either `time`, then the labels while color the embedding based on temporality, or a string that can be interpreted as a RGB(A) color, then the embedding will be uniformly display with that unique color.
        axis: Optional axis to create the plot on.
        idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
            embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
            dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
            (e.g., (2, 4, 5)).
        markersize: The marker size.
        alpha: The marker blending, between 0 (transparent) and 1 (opaque).
        cmap: The Colormap instance or registered colormap name used to map scalar data to colors. It will be ignored if `embedding_labels` is set to a valid RGB(A).
        title: The title on top of the embedding.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
        kwargs: Optional arguments to customize the plots. This dictionary includes the following optional arguments:
            -- showlegend: Whether to show the legend or not.
            -- discrete: Whether the labels are discrete or not.
            -- col: The column of the subplot to plot the embedding on.
            -- row: The row of the subplot to plot the embedding on.
            -- template: The template to use for the plot.

            Note: showlegend can be True only if discrete is True.

            See :py:class:`plotly.graph_objects.Scatter` documentation for more
                details on which arguments to use.

    Returns:
        The plotly figure.


    Example:

        >>> import cebra
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 5))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(X, y)
        CEBRA(max_iterations=10)
        >>> embedding = cebra_model.transform(X)
        >>> cebra_time = np.arange(X.shape[0])
        >>> fig = cebra.plot_embedding_interactive(embedding, embedding_labels=cebra_time)

    """
    return _EmbeddingInteractivePlot(
        embedding=embedding,
        embedding_labels=embedding_labels,
        axis=axis,
        idx_order=idx_order,
        markersize=markersize,
        alpha=alpha,
        cmap=cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
    ).plot(**kwargs)
