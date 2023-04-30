#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
"""Scikit-Learn interface for CEBRA

The implementation follows the guide on `Developing scikit-learn estimators`_.

Note:
    The main class of this module, :py:class:`CEBRA`, is also available under the
    top-level package name as :py:class:`cebra.CEBRA` and automatically imported
    with :py:mod:`cebra`.

.. _Developing scikit-learn estimators:
    https://scikit-learn.org/stable/developers/develop.html
"""

from cebra.integrations.sklearn import cebra
from cebra.integrations.sklearn import dataset
from cebra.integrations.sklearn import decoder
from cebra.integrations.sklearn import metrics
from cebra.integrations.sklearn import utils
