from collections.abc import Iterable
from typing import List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.cm


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


_register_colormap()
