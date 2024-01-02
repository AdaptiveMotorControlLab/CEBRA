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
"""Matplotlib interface to CEBRA."""
import abc
from collections.abc import Iterable
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sklearn.utils.validation
import torch

from cebra import CEBRA


def _register_colormap():
    """Register colormaps used in CEBRA plotting.

    Currently registered:
        * Logo colors (``cebra``)
    """

    if "cebra" not in matplotlib.colormaps:
        matplotlib.colormaps.register(
            matplotlib.colors.LinearSegmentedColormap.from_list(
                "", ["#4854e9", "#6235e0", "#a045e8", "#bf1bb9", "#d4164f"]),
            name="cebra",
            force=False,
        )


class _BasePlot:
    """Base plotting class.

    Attributes:
        axis: Optional axis to create the plot on.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
    """

    def __init__(self, axis: Optional[matplotlib.axes.Axes], figsize: tuple,
                 dpi: int):
        if axis is None:
            self.fig = plt.figure(figsize=figsize, dpi=dpi)

    @abc.abstractmethod
    def _define_ax(
            self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        raise NotImplementedError()

    @abc.abstractmethod
    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        raise NotImplementedError()


class _TemperaturePlot(_BasePlot):
    """Plot temperature evolution during model training.

    Attributes:
        model: The (trained) CEBRA model.
        color: Line color.
        linewidth: Line width.
        x_label: A boolean that specifies if the x-axis label should be displayed.
        y_label: A boolean that specifies if the y-axis label should be displayed.
        axis: Optional axis to create the plot on.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
    """

    def __init__(
        self,
        model: CEBRA,
        color: str,
        linewidth: float,
        x_label: bool,
        y_label: bool,
        axis: Optional[matplotlib.axes.Axes],
        figsize: tuple,
        dpi: int,
    ):
        super().__init__(axis, figsize, dpi)
        self.ax = self._define_ax(axis)
        sklearn.utils.validation.check_is_fitted(model, "n_features_")
        self.model = model
        self.color = color
        self.linewidth = linewidth
        self.x_label = x_label
        self.y_label = y_label

    def _define_ax(
            self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        """Define the ax on which to generate the plot.

        Args:
            axis: A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            self.ax = self.fig.add_subplot()
        else:
            self.ax = axis
        return self.ax

    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the temperature.

        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
        """
        self.ax.plot(
            self.model.state_dict_["log"]["temperature"],
            c=self.color,
            linewidth=self.linewidth,
            **kwargs,
        )
        if self.y_label:
            self.ax.set_ylabel("Temperature")
        if self.x_label:
            self.ax.set_xlabel("Steps")

        return self.ax


class _LossPlot(_BasePlot):
    """Plot loss evolution during model training.

    Attributes:
        model: The (trained) CEBRA model.
        label: The legend for the trace.
        color: Line color.
        linewidth: Line width.
        x_label: A boolean that specifies if the x-axis label should be displayed.
        y_label: A boolean that specifies if the y-axis label should be displayed.
        axis: Optional axis to create the plot on.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
        kwargs: Optional arguments to customize the plots.
    """

    def __init__(
        self,
        model: CEBRA,
        label: Optional[Union[str, int, float]],
        color: str,
        linewidth: float,
        x_label: bool,
        y_label: bool,
        axis: Optional[matplotlib.axes.Axes],
        figsize: tuple,
        dpi: int,
    ):
        super().__init__(axis, figsize, dpi)
        self.ax = self._define_ax(axis)
        sklearn.utils.validation.check_is_fitted(model, "n_features_")
        self.model = model
        self.label = label
        self.color = color
        self.linewidth = linewidth
        self.x_label = x_label
        self.y_label = y_label

    def _define_ax(
            self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        """Define the ax on which to generate the plot.

        Args:
            axis: A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            self.ax = self.fig.add_subplot()
        else:
            self.ax = axis
        return self.ax

    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the loss.

        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
        """
        self.ax.plot(
            self.model.state_dict_["loss"],
            c=self.color,
            linewidth=self.linewidth,
            label=self.label,
            **kwargs,
        )
        if self.y_label:
            loss = self.model.__dict__["criterion"]
            if loss == "infonce":
                loss = "InfoNCE"
            self.ax.set_ylabel(f"{loss} Loss")
        if self.x_label:
            self.ax.set_xlabel("Steps")

        if self.label is not None:
            self.ax.legend()

        return self.ax


class _EmbeddingPlot(_BasePlot):
    """Plot a CEBRA embedding in a 3D or 2D dimensional space.

    Attributes:
        embedding: A matrix containing the feature representation computed with CEBRA.
        embedding_labels: The labels used to map the data to color. It can be a vector that is the
            same sample size as the embedding, associating a value to each of the sample, either discrete
            or continuous or string, either `time`, then the labels while color the embedding based on
            temporality, or a string that can be interpreted as a RGB(A) color, then the embedding will be
            uniformly display with that unique color.
        ax: Optional axis to create the plot on.
        idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
            embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
            dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
            (e.g., (2, 4, 5)).
        markersize: The marker size.
        alpha: The marker blending, between 0 (transparent) and 1 (opaque).
        cmap: The Colormap instance or registered colormap name used to map scalar data to colors.
            It will be ignored if `embedding_labels` is set to a valid RGB(A).
        title: The title on top of the embedding.
        axis: Optional axis to create the plot on.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.

    """

    def __init__(
        self,
        embedding: Union[npt.NDArray, torch.Tensor],
        embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]],
        idx_order: Optional[Tuple[int]],
        markersize: float,
        alpha: float,
        cmap: str,
        title: str,
        axis: Optional[matplotlib.axes.Axes],
        figsize: tuple,
        dpi: int,
    ):
        super().__init__(axis, figsize, dpi)
        self._define_plot_dim(embedding, idx_order)
        self._define_ax(axis)
        self.embedding = embedding
        self.embedding_labels = embedding_labels
        self.idx_order = self._define_idx_order(idx_order)
        self.markersize = markersize
        self.alpha = alpha
        self.cmap = cmap
        self.title = title

    def _define_plot_dim(
        self,
        embedding: Union[npt.NDArray, torch.Tensor],
        idx_order: Optional[Tuple[int]],
    ):
        """Define the dimension of the embedding plot, either 2D or 3D, by setting ``_is_plot_3d``.

        If the embedding dimension is equal or higher to 3:

            * If ``idx_order`` is not provided, the plot will be 3D by default.
            * If ``idx_order`` is provided, if it has 3 dimensions, the plot will be 3D, if only 2 dimensions
                are provided, the plot will be 2D.

        If the embedding dimension is equal to 2:

            * If ``idx_order`` is not provided, the plot will be 2D by default.
            * If ``idx_order`` is provided, if it has 3 dimensions, the plot will be 3D, if 2 dimensions
                are provided, the plot will be 2D.

        This is supposing that the dimensions provided to ``idx_order`` are in the range of the number of
        dimensions of the embedding (i.e., between 0 and :py:attr:`cebra.CEBRA.output_dimension` -1).

        Args:
            embedding: A matrix containing the feature representation computed with CEBRA.
            idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
                embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
                dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
                (e.g., (2, 4, 5)).
        """
        if (idx_order is None and
                embedding.shape[1] == 2) or (idx_order is not None and
                                             len(idx_order) == 2):
            self._is_plot_3d = False
        elif (idx_order is None and
              embedding.shape[1] >= 3) or (idx_order is not None and
                                           len(idx_order) == 3):
            self._is_plot_3d = True
        else:
            raise ValueError(
                f"Invalid embedding dimension, expects 2D or more, got {self.embedding.shape[1]}"
            )

    def _define_ax(self, axis: Optional[matplotlib.axes.Axes]):
        """Define the ax on which to generate the plot.

        Args:
            axis: A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            if self._is_plot_3d:
                self.ax = self.fig.add_subplot(projection="3d")
            else:
                self.ax = self.fig.add_subplot()
        else:
            self.ax = axis

    def _define_idx_order(self, idx_order: Optional[Tuple[int]]) -> Tuple[int]:
        """Check that the index order has a valid number of dimensions compared to the number of
        dimensions of the embedding.

        Args:
            idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
                embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
                dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
                (e.g., (2, 4, 5)).

        Returns:
            The index order for the corresponding 2D or 3D plot.
        """

        if idx_order is None:
            if self._is_plot_3d:
                idx_order = (0, 1, 2)
            else:
                idx_order = (0, 1)
        else:
            # If the idx_order was provided by the user
            self._check_valid_dimensions(idx_order)
        return idx_order

    def _check_valid_dimensions(self, idx_order: Tuple[int]):
        """Check that provided dimensions are valid.

        The provided dimensions need to be 2 if the plot is set to a 2D plot and 3 if it is set to 3D.
        The dimensions values should be in the range of the embedding dimensionality.

        Args:
            idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
                embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
                dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
                (e.g., (2, 4, 5)).
        """
        # Check size validity
        if self._is_plot_3d and len(idx_order) != 3:
            raise ValueError(
                f"idx_order must contain 3 dimension values, got {len(idx_order)}."
            )
        elif not self._is_plot_3d and len(idx_order) != 2:
            raise ValueError(
                f"idx_order must contain 2 dimension values, got {len(idx_order)}."
            )

        # Check value validity
        for dim in idx_order:
            if dim < 0 or dim > self.embedding.shape[1] - 1:
                raise ValueError(
                    f"List of dimensions to plot is invalid, got {idx_order}, with {dim} invalid."
                    f"Values should be between 0 and {self.embedding.shape[1]}."
                )

    def _plot_3d(self,
                 grey_fig: bool = False,
                 **kwargs) -> matplotlib.axes.Axes:
        """Plot the embedding in 3d.

        Args:
            grey_fig: Set the title and edge color to grey, to be visible on both white and black
                backgrounds.


        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
        """

        idx1, idx2, idx3 = self.idx_order
        self.ax.scatter(
            xs=self.embedding[:, idx1],
            ys=self.embedding[:, idx2],
            zs=self.embedding[:, idx3],
            c=self.embedding_labels,
            cmap=self.cmap,
            alpha=self.alpha,
            s=self.markersize,
            **kwargs,
        )

        self.ax.grid(False)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("w")
        self.ax.yaxis.pane.set_edgecolor("w")
        self.ax.zaxis.pane.set_edgecolor("w")
        self.ax.set_title(self.title, y=1.0, pad=-10)

        if grey_fig:
            self.ax.xaxis.pane.set_edgecolor("grey")
            self.ax.yaxis.pane.set_edgecolor("grey")
            self.ax.zaxis.pane.set_edgecolor("grey")

        return self.ax

    def _plot_2d(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the embedding in 2d.

        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
        """

        idx1, idx2 = self.idx_order
        self.ax.scatter(
            x=self.embedding[:, idx1],
            y=self.embedding[:, idx2],
            c=self.embedding_labels,
            cmap=self.cmap,
            alpha=self.alpha,
            s=self.markersize,
            **kwargs,
        )

        return self.ax

    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the embedding.

        Note:
            To set the entire figure to grey, you should add that snippet of code:

            >>> from matplotlib import rcParams
            >>> rcParams['xtick.color'] = 'grey'
            >>> rcParams['ytick.color'] = 'grey'
            >>> rcParams['axes.labelcolor'] = 'grey'
            >>> rcParams['axes.edgecolor'] = 'grey'
            >>> rcParams['axes.titlecolor'] = 'grey'

        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
        """
        if isinstance(self.embedding_labels, str):
            if self.embedding_labels == "time":
                self.embedding_labels = np.arange(self.embedding.shape[0])
            elif not matplotlib.colors.is_color_like(self.embedding_labels):
                raise ValueError(
                    f"Embedding labels invalid: provide a list of index or a valid str (time or valid colorname), got {self.embedding_labels}."
                )
            self.cmap = None
        elif isinstance(self.embedding_labels, Iterable):
            if len(self.embedding_labels) != self.embedding.shape[0]:
                raise ValueError(
                    f"Invalid embedding labels: the labels vector should have the same number of samples as the embedding, got {len(self.embedding_labels)}, expect {self.embedding.shape[0]}."
                )
            if self.embedding_labels.ndim > 1:
                raise NotImplementedError(
                    f"Invalid embedding labels: plotting does not support multiple sets of labels, got {self.embedding_labels.ndim}."
                )

        if self._is_plot_3d:
            self.ax = self._plot_3d(**kwargs)
        else:
            self.ax = self._plot_2d(**kwargs)
        if isinstance(self.ax, matplotlib.axes._axes.Axes):
            self.ax.set_title(self.title)

        return self.ax


class _ConsistencyPlot(_BasePlot):
    """Plot a consistency matrix from the scores obtained with :py:func:`~.consistency_score`.

    Attributes:
        scores: List of consistency scores obtained by comparing a set of CEBRA embeddings using :py:func:`~.consistency_score`.
        pairs: Optional list of the pairs of datasets/runs whose embeddings were compared in ``scores``.
        datasets: Optional list of the datasets whose embeddings were compared in ``scores``. Each dataset is present once only in the list.
        ax: Optional axis to create the plot on.
        cmap: Color map to use to color the m
        text_color: If None, then the ``scores`` values are not displayed, else, it corresponds to the color
            of the values displayed inside the grid.
        colorbar_label: If None, then the color bar is not shown, else, it defines the corresponding title.
        title: The title on top of the confusion matrix.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.

    """

    def __init__(
        self,
        scores: Union[torch.Tensor, list, npt.NDArray],
        datasets: Optional[Union[list, npt.NDArray]],
        pairs: Optional[Union[list, npt.NDArray]],
        cmap: str,
        text_color: Optional[str],
        colorbar_label: Optional[str],
        title: Optional[str],
        axis: Optional[matplotlib.axes.Axes],
        figsize: tuple,
        dpi: int,
    ):
        super().__init__(axis, figsize, dpi)
        self._define_ax(axis)
        scores = self._check_array(scores)
        # Check the values dimensions
        if scores.ndim >= 2:
            raise ValueError(
                f"Invalid scores dimensions, expect 1D, got {scores.ndim}D.")

        self.labels = self._compute_labels(scores,
                                           pairs=pairs,
                                           datasets=datasets)
        self.scores = self._to_heatmap_format(scores)
        self.cmap = cmap
        self.text_color = text_color
        self.colorbar_label = colorbar_label
        self.title = title

    def _define_ax(
            self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        """Define the ax on which to generate the matrix.

        Args:
            axis: A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            self.ax = self.fig.add_subplot()
        else:
            self.ax = axis
        return self.ax

    def _check_array(
            self, array: Union[torch.Tensor, list, npt.NDArray]) -> npt.NDArray:
        """Check the ``array`` type and either convert to :py:func:`numpy.array` or raise an
        error.

        Args:
            array: Data structure to convert to :py:func:`numpy.array`.

        Returns:
            A :py:func:`numpy.array`.
        """
        if not isinstance(array, np.ndarray):
            if isinstance(array, list):
                array = np.array(array)
            elif isinstance(array, torch.Tensor):
                array = array.numpy()
            else:
                raise ValueError(
                    f"Invalid values format, expect a numpy.array, torch.Tensor or list, got {type(array)}."
                )
        return array

    def _compute_labels(
        self,
        scores: Union[torch.Tensor, list, npt.NDArray],
        pairs: Union[list, npt.NDArray],
        datasets: Union[list, npt.NDArray],
    ) -> npt.NDArray:
        """Define the x- and y-labels to use for the confusion matrix.

        The labels qualify the embeddings that were compared and are selected based on ``pairs`` and ``datasets``.
        Note that as comparison is one-to-one, the confusion matrix is a square and
        the labels are similar on the x- and y-axis. First we check if the number of datasets compared to
        the number of scores is consistent to a between-datasets comparison. As we do not compute
        self-consistency, we expect :math:`#scores = #datasets^2 - #datasets` as well as finding the datasets
        IDs present in ``datasets`` also in ``pairs``. If that is not the case, we check if the number of
        pairs in ``pairs`` is consistent with the number of scores in ``scores``. For that, we expect
        :math:`#scores = #pairs` and the number of unique dataset IDs in ``pairs`` is such that, similarly,
        :math:`#scores = #set(pairs)^2 - #set(pairs)`, with :math:`set(pairs)` the set of unique dataset IDs
        in ``pairs``.

        Args:
            scores: List of consistency scores obtained by comparing a set of CEBRA embeddings using
                :py:func:`~.consistency_score`.
            pairs: Optional list of the pairs of datasets/runs whose embeddings were compared in ``scores``.
            datasets: Optional list of the datasets whose embeddings were compared in ``scores``. Each dataset
                is present once only in the list.

        Returns:
            The labels to use.
        """
        if datasets is None or pairs is None:
            raise ValueError(
                "Missing datasets or pairs, provide both of them to plot consistency, "
                "got either both or one of them  set to None.")
        else:
            datasets = self._check_array(datasets)
            pairs = self.pairs = self._check_array(pairs)

            if len(pairs.shape) == 2:
                compared_items = list(sorted(set(pairs[:, 0])))
            elif len(pairs.shape) == 3:
                compared_items = list(sorted(set(pairs[0, :, 0])))

            # between-datasets comparison
            if len(scores) == len(datasets)**2 - len(datasets) and all(
                    mouse in compared_items for mouse in datasets):
                self.labels = datasets
            # between-runs comparison
            elif len(scores) == len(compared_items)**2 - len(
                    compared_items) and not any(mouse in compared_items
                                                for mouse in datasets):
                self.labels = np.array(compared_items)
            else:
                raise ValueError(
                    f"Shape of the scores, datasets and pairs do not match, got scores:{scores.shape}, datasets:{len(datasets)} and pairs:{len(pairs)}."
                )

        return self.labels

    def _to_heatmap_format(
            self, values: Union[torch.Tensor, list,
                                npt.NDArray]) -> npt.NDArray:
        """Transform ``values`` to a compatible format for :py:func:`matplotlib.pyplot.imshow`.

        Args:
            values: An array of values to transform to a compatible format, meaning, square format,
                of size `len(self.labels)`, with NaNs on the diagonal. We also transform the values
                to percentages.

        Returns:
            A :py:func:`numpy.array` of shape ``(len(self.labels), len(self.labels))``.

        """
        if values.ndim == 1:
            values = np.expand_dims(values, axis=0)

        values = np.concatenate(values)

        pairs = self.pairs

        if pairs.ndim == 3:
            pairs = pairs[0]

        assert len(pairs) == len(values), (self.pairs.shape, len(values))
        score_dict = {tuple(pair): value for pair, value in zip(pairs, values)}

        if self.labels is None:
            n_grid = self.score

        heatmap_values = np.zeros((len(self.labels), len(self.labels)))

        heatmap_values[:] = float("nan")
        for i, label_i in enumerate(self.labels):
            for j, label_j in enumerate(self.labels):
                if i == j:
                    heatmap_values[i, j] = float("nan")
                else:
                    heatmap_values[i, j] = score_dict[label_i, label_j]

        return np.minimum(heatmap_values * 100, 99)

    def _create_text(self):
        """Create the text to add in the confusion matrix grid and the title."""
        if self.text_color is not None:
            for (i, j), z in np.ndenumerate(self.scores):
                if z == z:  # not NaN
                    self.ax.text(
                        j,
                        i,
                        "{:0.1f}".format(z),
                        ha="center",
                        va="center",
                        color=self.text_color,
                    )

        if self.title is not None:
            self.ax.set_title(self.title)

    def _create_colorbar(self, im):
        """Create color bar based on the provided ``colorbar_label``."""
        if self.colorbar_label is not None:
            cbar = self.ax.figure.colorbar(im, ax=self.ax)
            cbar.outline.set_visible(False)
            cbar.ax.set_ylabel(self.colorbar_label, rotation=-90, va="bottom")

    def _create_labels(self):
        """Create and add the labels to the confusion matrix."""
        # Show labels
        if self.labels is not None:
            self.ax.set_xticks(np.arange(self.scores.shape[1]),
                               labels=self.labels)
            self.ax.set_yticks(np.arange(self.scores.shape[0]),
                               labels=self.labels)

        # Remove labels ticks and pivot y-labels
        self.ax.tick_params(left=False,
                            bottom=False,
                            labelleft=True,
                            labelbottom=True)
        plt.setp(self.ax.get_yticklabels(),
                 rotation=90,
                 ha="center",
                 rotation_mode="anchor")
        self.ax.tick_params(axis="y", which="major", pad=10)

    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the consistency matrix.

        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the matrix.
        """
        im = self.ax.imshow(
            self.scores,
            cmap=self.cmap,
            **kwargs,
        )

        self._create_text()
        self._create_colorbar(im)
        self._create_labels()

        # Turn spines off and create white grid.
        self.ax.spines[:].set_visible(False)
        self.ax.tick_params(which="minor", bottom=False, left=False)

        return self.ax


def plot_overview(
    model: CEBRA,
    X: Union[npt.NDArray, torch.Tensor],
    loss_kwargs: dict = {},
    temperature_kwargs: dict = {},
    embedding_kwargs: dict = {},
    figsize: tuple = (15, 4),
    dpi: int = 100,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, Tuple[
        matplotlib.axes.Axes, matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Plot an overview of a trained CEBRA model.

    Args:
        model: The (fitted) CEBRA model to analyze to compute the different plots.
        X: The data matrix on which to compute the embedding to plot.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
        kwargs: Optional arguments to customize the plots.

    Returns:
        The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 5))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(X, y)
        CEBRA(max_iterations=10)
        >>> fig = cebra.plot_overview(cebra_model, X, embedding_kwargs={"embedding_labels":"time"})

    """

    fig = plt.figure(figsize=figsize, dpi=dpi)
    grid = plt.GridSpec(4, 2, wspace=0, hspace=0.2, figure=fig)

    # Plot the loss
    ax1 = fig.add_subplot(grid[:2, 0])
    ax1 = plot(
        model,
        which="loss",
        ax=ax1,
        x_label=False,
        linewidth=1,
        figure=fig,
        **loss_kwargs,
        **kwargs,
    )
    ax2 = fig.add_subplot(grid[2:, 0], sharex=ax1)

    # Plot the temperature
    ax2 = plot(model,
               which="temperature",
               ax=ax2,
               linewidth=1,
               **temperature_kwargs,
               **kwargs)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Compute and plot the embedding of X
    ax3 = fig.add_subplot(grid[:, 1:], projection="3d")
    ax3 = plot(
        model,
        which="embedding",
        X=X,
        ax=ax3,
        markersize=0.01,
        **embedding_kwargs,
        **kwargs,
    )

    return fig, (ax1, ax2, ax3)


def plot(
    model: CEBRA,
    which: str = Literal["loss", "temperature", "embedding"],
    X: Optional[Union[npt.NDArray, torch.Tensor]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot the required information.

    Args:
        model: The (fitted) CEBRA model to analyze to compute the different plots.
        which: The required information to plot.
        X: The data matrix on which to compute the embedding to plot, needed only to display an embedding.
        ax: Optional axis to create the plot on.
        kwargs: Optional arguments to customize the plots or provide the embedding labels.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 5))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(X, y)
        CEBRA(max_iterations=10)
        >>> ax = cebra.plot(cebra_model, which="temperature")

    Returns:
        The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
    """
    if which == "loss":
        ax = plot_loss(model, ax=ax, **kwargs)
    elif which == "temperature":
        ax = plot_temperature(model, ax=ax, **kwargs)
    elif which == "embedding":
        if X is None:
            raise ValueError(
                "No data: a data matrix to compute the embedding on needs to be provided."
            )
        embedding = model.transform(X)

        ax = plot_embedding(embedding=embedding, ax=ax, **kwargs)
    else:
        raise ValueError(
            f"Invalid value for `which`: got {which}, expect loss, temperature or embedding."
        )

    return ax


def plot_temperature(
    model: CEBRA,
    ax: Optional[matplotlib.axes.Axes] = None,
    color: str = "dodgerblue",
    linewidth: int = 1,
    x_label: bool = True,
    y_label: bool = True,
    figsize: tuple = (7, 4),
    dpi: float = 100,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot the evolution of the :py:attr:`~.CEBRA.temperature` hyperparameter during model training.

    Note:
        It will vary only when using :py:attr:`~.CEBRA.temperature_mode`, else the :py:attr:`~.CEBRA.temperature` stays constant.

    The function makes use of :py:func:`matplotlib.pyplot.plot` and parameters from that function can be provided
    as part of ``kwargs``.

    Args:
        model: The (trained) CEBRA model.
        ax: Optional axis to create the plot on.
        color: Line color.
        linewidth: Line width.
        x_label: A boolean that specifies if the x-axis label should be displayed.
        y_label: A boolean that specifies if the y-axis label should be displayed.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
        kwargs: Optional arguments to customize the plots. See :py:func:`matplotlib.pyplot.plot` documentation for more
            details on which arguments to use.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 5))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(X, y)
        CEBRA(max_iterations=10)
        >>> ax = cebra.plot_temperature(cebra_model)

    Returns:
        The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.

    """

    return _TemperaturePlot(
        model=model,
        axis=ax,
        color=color,
        linewidth=linewidth,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        dpi=dpi,
    ).plot(**kwargs)


def plot_loss(
    model: CEBRA,
    label: Optional[Union[str, int, float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    color: str = "magenta",
    linewidth: int = 1,
    x_label: bool = True,
    y_label: bool = True,
    figsize: tuple = (7, 4),
    dpi: float = 100,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot the evolution of loss during model training.

    The function makes use of :py:func:`matplotlib.pyplot.plot` and parameters from that function can be provided
    as part of ``kwargs``.

    Args:
        model: The (trained) CEBRA model.
        label: The legend for the loss trace.
        ax: Optional axis to create the plot on.
        color: Line color.
        linewidth: Line width.
        x_label: A boolean that specifies if the x-axis label should be displayed.
        y_label: A boolean that specifies if the y-axis label should be displayed.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
        kwargs: Optional arguments to customize the plots. See :py:func:`matplotlib.pyplot.plot` documentation for more
            details on which arguments to use.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> X = np.random.uniform(size=(100, 50))
        >>> y = np.random.uniform(size=(100, 5))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(X, y)
        CEBRA(max_iterations=10)
        >>> ax = cebra.plot_loss(cebra_model)

    Returns:
        The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.

    """

    return _LossPlot(
        model=model,
        label=label,
        axis=ax,
        color=color,
        linewidth=linewidth,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        dpi=dpi,
    ).plot(**kwargs)


def plot_embedding(
    embedding: Union[npt.NDArray, torch.Tensor],
    embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]] = "grey",
    ax: Optional[matplotlib.axes.Axes] = None,
    idx_order: Optional[Tuple[int]] = None,
    markersize: float = 0.05,
    alpha: float = 0.4,
    cmap: str = "cool",
    title: str = "Embedding",
    figsize: Tuple[int] = (5, 5),
    dpi: float = 100,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot embedding in a 3D or 2D dimensional space.

    If the embedding dimension is equal or higher to 3:

        * If ``idx_order`` is not provided, the plot will be 3D by default.
        * If ``idx_order`` is provided, if it has 3 dimensions, the plot will be 3D, if only 2 dimensions are provided, the plot will be 2D.

    If the embedding dimension is equal to 2:

        * If ``idx_order`` is not provided, the plot will be 2D by default.
        * If ``idx_order`` is provided, if it has 3 dimensions, the plot will be 3D, if 2 dimensions are provided, the plot will be 2D.

    This is supposing that the dimensions provided to ``idx_order`` are in the range of the number of
    dimensions of the embedding (i.e., between 0 and :py:attr:`cebra.CEBRA.output_dimension` -1).

    The function makes use of :py:func:`matplotlib.pyplot.scatter` and parameters from that function can be provided
    as part of ``kwargs``.


    Args:
        embedding: A matrix containing the feature representation computed with CEBRA.
        embedding_labels: The labels used to map the data to color. It can be:

            * A vector that is the same sample size as the embedding, associating a value to each of the sample, either discrete or continuous.
            * A string, either `time`, then the labels while color the embedding based on temporality, or a string that can be interpreted as a RGB(A) color, then the embedding will be uniformly display with that unique color.
        ax: Optional axis to create the plot on.
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
        kwargs: Optional arguments to customize the plots. See :py:func:`matplotlib.pyplot.scatter` documentation for more
            details on which arguments to use.

    Returns:
        The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 5))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(X, y)
        CEBRA(max_iterations=10)
        >>> embedding = cebra_model.transform(X)
        >>> ax = cebra.plot_embedding(embedding, embedding_labels='time')

    """
    return _EmbeddingPlot(
        embedding=embedding,
        embedding_labels=embedding_labels,
        axis=ax,
        idx_order=idx_order,
        markersize=markersize,
        alpha=alpha,
        cmap=cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
    ).plot(**kwargs)


def plot_consistency(
    scores: Union[npt.NDArray, torch.Tensor, list],
    pairs: Optional[Union[npt.NDArray, list]] = None,
    datasets: Optional[Union[npt.NDArray, list]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: str = "binary",
    text_color: str = "white",
    colorbar_label: Optional[str] = "Consistency score (%)",
    title: Optional[str] = None,
    figsize: Tuple[int] = (5, 5),
    dpi: float = 100,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot the consistency matrix from the consistency scores obtained with :py:func:`~.consistency_score`.

    To display the consistency map, ``datasets`` and ``pairs`` must be provided.
    The function is implemented so that it will first try to display the labels as if the
    scores were computed following a between-sujects comparison.
    If only one dataset is present or if the number of datasets doesn't fit with
    the number of scores, then the function will display the labels as if the scores were computed following a
    between-runs comparison. If the number of pairs still doesn't fit with that comparison, then an error is
    raised, asking the user to make sure that the formats of the scores, datasets and pairs are valid.

    .. tip::
        The safer way to use that function is to directly use as inputs the outputs from :py:func:`~.consistency_score`.

    The function makes use of :py:func:`matplotlib.pyplot.imshow` and parameters from that function can be provided as part of ``kwargs``.
    For instance, we recommend that you bound the color bar using ``vmin`` and ``vmax``. The scores are percentages between 0 and 100.

    Args:
        scores: List of consistency scores obtained by comparing a set of CEBRA embeddings using :py:func:`~.consistency_score`.
        pairs: List of the pairs of datasets/runs whose embeddings were compared in ``scores``.
        datasets: List of the datasets whose embeddings were compared in ``scores``. Each dataset is present once only in the list.
        ax: Optional axis to create the plot on.
        cmap: Color map to use to color the m
        text_color: If None, then the ``scores`` values are not displayed, else, it corresponds to the color
            of the values displayed inside the grid.
        colorbar_label: If None, then the color bar is not shown, else, it defines the corresponding title.
        title: The title on top of the confusion matrix.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
        kwargs: Optional arguments to customize the plots. See :py:func:`matplotlib.pyplot.imshow` documentation for more
            details on which arguments to use.

    Returns:
        The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> embedding1 = np.random.uniform(0, 1, (1000, 5))
        >>> embedding2 = np.random.uniform(0, 1, (1000, 8))
        >>> labels1 = np.random.uniform(0, 1, (1000, ))
        >>> labels2 = np.random.uniform(0, 1, (1000, ))
        >>> dataset_ids = ["achilles", "buddy"]
        >>> # between-datasets consistency, by aligning on the labels
        >>> scores, pairs, datasets = cebra.sklearn.metrics.consistency_score(embeddings=[embedding1, embedding2], labels=[labels1, labels2], dataset_ids=dataset_ids, between="datasets")
        >>> ax = cebra.plot_consistency(scores, pairs, datasets, vmin=0, vmax=100)

    """

    return _ConsistencyPlot(
        scores=scores,
        datasets=datasets,
        pairs=pairs,
        axis=ax,
        cmap=cmap,
        text_color=text_color,
        colorbar_label=colorbar_label,
        title=title,
        figsize=figsize,
        dpi=dpi,
    ).plot(**kwargs)


from cebra.helper import requires_package_version


@requires_package_version(matplotlib, "3.6")
def compare_models(
    models: List[CEBRA],
    labels: Optional[List[str]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    color: Optional[str] = None,
    cmap: str = "cebra",
    linewidth: int = 1,
    x_label: bool = True,
    y_label: bool = True,
    figsize: tuple = (7, 4),
    dpi: float = 100,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot the evolution of loss during model training.

    The function makes use of :py:func:`matplotlib.pyplot.plot` and parameters from that function can be provided
    as part of ``kwargs``.

    Args:
        models: A list of trained CEBRA models.
        labels: A list of labels, associated to each model to be used in the legend of the plot.
        ax: Optional axis to create the plot on.
        color: Line color. If not ``None``, then all the traces for all models will be of the same color.
        cmap: Color map from which to sample uniformly the quantitative list of colors for each trace.
            If ``color`` is different from then that parameter is ignored.
        linewidth: Line width.
        x_label: A boolean that specifies if the x-axis label should be displayed.
        y_label: A boolean that specifies if the y-axis label should be displayed.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
        kwargs: Optional arguments to customize the plots. See :py:func:`matplotlib.pyplot.plot` documentation for more
            details on which arguments to use.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 5))
        >>> output_dimensions = [3, 8, 12, 32]
        >>> models, labels = [], []
        >>> for output_dimension in output_dimensions:
        ...     cebra_model = cebra.CEBRA(max_iterations=10,
        ...                               output_dimension=output_dimension)
        ...     cebra_model = cebra_model.fit(X, y)
        ...     models.append(cebra_model)
        ...     labels.append(f"Output dimension: {output_dimension}")
        >>> ax = cebra.compare_models(models, labels)

    Returns:
        The axis of the generated plot. If no ``ax`` argument was specified, it will be created
        by the function and returned here.
    """

    if not isinstance(models, list):
        raise ValueError(f"Invalid list of models, got {type(models)}.")

    for model in models:
        if not isinstance(model, CEBRA):
            raise ValueError(
                f"Invalid list of models, it should only contain CEBRA models, got {model}, of type {type(model)}"
            )
    n_models = len(models)

    # check the color of the traces
    if color is None:
        cebra_map = plt.get_cmap(cmap)
        colors = matplotlib.colors.ListedColormap(
            cebra_map.resampled(n_models)(np.arange(n_models))).colors
    else:
        colors = [color for i in range(n_models)]

    # check the labels
    if labels is None:
        labels = [None for i in range(n_models)]
    else:
        if not isinstance(labels, list):
            raise ValueError(f"Invalid list of labels, got {type(labels)}.")
        if len(labels) != len(models):
            raise ValueError(
                f"Invalid list of labels, it should be the same length as the list of models,"
                f"got {len(labels)}, expected {len(models)}.")
        for label in labels:
            if not (isinstance(label, str) or isinstance(label, int) or
                    isinstance(label, float)):
                raise ValueError(
                    f"Invalid list of labels, it should only contain strs/ints/floats, got {label}, of type {type(label)}"
                )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for color, model, label in zip(colors, models, labels):
        ax = plot_loss(
            model,
            color=color,
            label=label,
            linewidth=linewidth,
            x_label=x_label,
            y_label=y_label,
            dpi=dpi,
            ax=ax,
            **kwargs,
        )

    return ax


_register_colormap()
