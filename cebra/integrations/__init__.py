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
"""Integration of CEBRA into common machine learning libraries.

This package contains a growing collection of interfaces to other Python packages.
There is no clear limit (yet) of what can go into it. The current examples include 
interfaces (implemented or planned) to `scikit-learn <https://scikit-learn.org/stable/>`_,
`streamlit <https://streamlit.io/>`_, `deeplabcut <http://www.mackenziemathislab.org/deeplabcut>`_,
`matplotlib <https://matplotlib.org/>`_ and `threejs <https://threejs.org/>`_.

Integrations can be used for data visualization, for providing easier interfaces to using CEBRA
for a particular userbase, or any other helpful function that requires a dependency to a larger
third-party package.

See our CEBRA `live demo <https://stes.io/c>`_.
"""
