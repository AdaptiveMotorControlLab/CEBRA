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
"""Integration of CEBRA with common machine learning and visualization libraries.

This package contains a growing collection of interfaces to other Python packages.
There is no clear limit (yet) of what can go into it. The current examples include
interfaces (implemented or planned) to `scikit-learn <https://scikit-learn.org/stable/>`_,
`streamlit <https://streamlit.io/>`_, `deeplabcut <http://www.mackenziemathislab.org/deeplabcut>`_,
`matplotlib <https://matplotlib.org/>`_ and `threejs <https://threejs.org/>`_ and `plotly <https://plotly.com/>`_.

Integrations can be used for data visualization, for providing easier interfaces to using CEBRA
for a particular userbase, or any other helpful function that requires a dependency to a larger
third-party package.
"""
