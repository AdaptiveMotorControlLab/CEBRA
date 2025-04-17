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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import cebra  # noqa: E402


def get_years(start_year=2021):
    year = datetime.datetime.now().year
    if year > start_year:
        return f"{start_year} - {year}"
    else:
        return f"{year}"


# -- Project information -----------------------------------------------------
project = "cebra"
copyright = f"""{get_years(2021)}"""
author = "See AUTHORS.md"
# The full version, including alpha/beta/rc tags
release = cebra.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

#https://github.com/spatialaudio/nbsphinx/issues/128#issuecomment-1158712159
html_js_files = [
    "require.min.js",  # Add to your _static
    "custom.js",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.todo",
    "nbsphinx",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx.ext.doctest",
    "sphinx_gallery.load_style",
]

coverage_show_missing_items = True
panels_add_bootstrap_css = False

# NOTE(stes): All configuration options for the napoleon package
#   The package is used to configure rendering of Google-style
#   docstrings used throughout the CEBRA package.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = True
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = True
# napoleon_use_admonition_for_notes = True
# napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
# napoleon_use_keyword = True
napoleon_use_param = True
# napoleon_use_rtype = True
# napoleon_preprocess_types = False
# napoleon_type_aliases = None
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None)
}

# Config is documented here: https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r">>> |\$ |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "torch", "nlb_tools", "tqdm", "h5py", "pandas", "matplotlib", "plotly"
]
# autodoc_typehints = "none"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "**/todo", "**/src", "cebra-figures/figures.rst", "cebra-figures/*.rst",
    "*/cebra-figures/*.rst", "demo_notebooks/README.rst"
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_context = {
    "default_mode": "light",
    "switcher": {
        "version_match":
            "latest",  # Adjust this dynamically per version
        "versions": [
            ("latest", "/latest/"),
            ("v0.2.0", "/v0.2.0/"),
            ("v0.3.0", "/v0.3.0/"),
            ("v0.4.0", "/v0.4.0/"),
            ("v0.5.0rc1", "/v0.5.0rc1/"),
        ],
    },
    "navbar_start": ["version-switcher",
                     "navbar-logo"],  # Place the dropdown above the logo
}

# More info on theme options:
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html
html_theme_options = {
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/AdaptiveMotorControlLab/CEBRA",
            "icon": "fab fa-github",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/cebraAI",
            "icon": "fab fa-twitter",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cebra/",
            "icon": "fab fa-python",
        },
        {
            "name": "How to cite CEBRA",
            "url": "https://doi.org/10.1038/s41586-023-06031-6#citeas",
            "icon": "fas fa-graduation-cap",
        },
    ],
    "external_links": [
        # {"name": "Mathis Lab", "url": "http://www.mackenziemathislab.org/"},
    ],
    "collapse_navigation": False,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "navbar_align": "content",
    "show_prev_next": False,
}

html_context = {"default_mode": "dark"}
html_favicon = "_static/img/logo_small.png"
html_logo = "_static/img/logo_large.png"

# Remove the search field for now
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"],
}

# Disable links for embedded images
html_scaled_image_link = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# See discussion here: https://github.com/sphinx-doc/sphinx/issues/6895#issuecomment-570759798
# Right now, only python module types are problematic, in cebra.registry
nitpick_ignore = [
    ("py:class", "module"),
]

nbsphinx_thumbnails = {
    "demo_notebooks/CEBRA_best_practices":
        "_static/thumbnails/cebra-best.png",
    "demo_notebooks/Demo_primate_reaching":
        "_static/thumbnails/ForelimbS1.png",
    "demo_notebooks/Demo_hippocampus":
        "_static/thumbnails/encodingSpace.png",
    "demo_notebooks/Demo_Allen":
        "_static/thumbnails/DecodingVideos.png",
    "demo_notebooks/Demo_conv-pivae":
        "_static/thumbnails/TechconvpiVAE.png",
    "demo_notebooks/Demo_hippocampus_multisession":
        "_static/thumbnails/TechMultiSession.png",
    "demo_notebooks/Demo_learnable_temperature":
        "_static/thumbnails/TechLearningTemp.png",
    "demo_notebooks/Demo_primate_reaching_mse_loss":
        "_static/thumbnails/TechMSE.png",
    "demo_notebooks/Demo_synthetic_exp":
        "_static/thumbnails/SyntheticBenchmark.png",
    "demo_notebooks/Demo_consistency":
        "_static/thumbnails/consistency.png",
    "demo_notebooks/Demo_decoding":
        "_static/thumbnails/decoding.png",
    "demo_notebooks/Demo_hypothesis_testing":
        "_static/thumbnails/hypothesis.png",
    "demo_notebooks/Demo_cohomology":
        "_static/thumbnails/cohomology.png",
    "demo_notebooks/Demo_openscope_databook":
        "_static/thumbnails/openScope_demo.png",
    "demo_notebooks/Demo_dandi_NeuroDataReHack_2023":
        "_static/thumbnails/dandi_demo_monkey.png",
    "demo_notebooks/Demo_dandi_NeuroDataReHack_2023":
        "_static/thumbnails/xCEBRA.png",
}

rst_prolog = r"""

.. |Default:| raw:: html

    <div class="default-value-section"> <span class="default-value-label">Default:</span>

"""

# Download link for the notebook, see
# https://nbsphinx.readthedocs.io/en/0.3.0/prolog-and-epilog.html

# fmt: off
# flake8: noqa: E501
nbsphinx_prolog = r"""

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        {% if 'demo_notebooks' in env.docname %}
        You can download and run the notebook locally or run it with Google Colaboratory:

        :raw-html:`<a href="/docs/{{ env.docname }}.ipynb"><img alt="Download jupyter notebook" src="https://img.shields.io/badge/download-jupyter%20notebook-bf1bb9" style="vertical-align:text-bottom"></a>`
        :raw-html:`<a href="https://colab.research.google.com/github/AdaptiveMotorControlLab/CEBRA-demos/blob/main/{{ env.doc2path(env.docname, base=None)|basename }}"><img alt="Run on Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>`
        {% else %}
        You can download and run the notebook locally:

        :raw-html:`<a href="/docs/{{ env.docname }}.ipynb"><img alt="Download jupyter notebook" src="https://img.shields.io/badge/download-jupyter%20notebook-bf1bb9" style="vertical-align:text-bottom"></a>`
        {% endif %}

----
"""
# fmt: on
# flake8: enable=E501
