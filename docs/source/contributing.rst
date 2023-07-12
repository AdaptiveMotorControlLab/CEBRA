Contribution Guide
==================

CEBRA is an actively developed package and we welcome community development
and involvement. We are happy to receive code extensions, bug fixes, documentation
updates, etc, but please sign the `Contributor License Agreement (CLA) <https://forms.gle/SYbceYvot64ngNxJ9>`_
and note that it was signed in your pull request.

Development setup
^^^^^^^^^^^^^^^^^

Development should be done inside the provided docker environment.
All essential commands are included in the project ``Makefile``.

To start an interactive console, run:

.. code:: bash

    $ make interact

We use ``pytest`` for running tests. The full test suite can be run with:

.. code:: bash

    $ make test

A faster version of the test suite, only running one iteration of each longer tests, can be run with:

.. code:: bash

    $ make test_fast

To investigate the last failed test and re-run it, use:

.. code:: bash

    $ make test_debug

Docs are placed in ``docs/`` and can be built using ``sphinx``, by running:

.. code:: bash

    $ make docs

Code is formatted using `Google code style <https://google.github.io/styleguide/pyguide.html>`_, but with 4 spaces.
The specification is in ``.style.yapf`` and ``.style.yapf``.
We use ``yapf`` for automated formatting and ``isort`` for import statement formatting.

Formatting the whole code base can be done with

.. code:: bash

    $ make format

For in-depth information on how to adapt and contribute to CEBRA, please refer to the full documentation.

Quick testing
^^^^^^^^^^^^^

Upon development, you'll write new tests to assess the quality of your contribution. If those tests are lenghy (e.g., lots of iterations to run), an option for "quick testing"
upon ``git push`` is available.To mark a test as "slow",

* Write the "slow" test function (name begins with ``test_``) in the file of your choice in the ``tests/`` folder.

* Decorate it with our custom ``@parametrize_slow`` decorator. For that, provide the sets of parameter to run for the "fast" and "slow" versions as arguments. Usually, the "fast" version corresponds to the first iteration of the "slow" version. The type of the arguments should be an iterable.

* We also have a special decorator ``@parametrize_with_checks_slow``, which replaces the ``sklearn.utils.estimator_checks.parametrize_with_checks`` pytest specific decorator and checks if an estimator provided as an argument adheres to the scikit-learn convention.

* More concrete examples are available in ``tests/test_sklearn.py``.

.. code:: python

    from _utils import parametrize_slow

    @parametrize_slow(
        arg_names="param_a, param_b",
        fast_arguments=[(1, 2)],
        slow_arguments=[(1, 2), (3, 4), (5, 6)],
    )
    def test_example(param_a, param_b):
        # testing things ...


You can now skip the slower test version by running pytest in the ``--runfast`` mode.
Upon commit or pull request (PR), the slower tests will be automatically skipped. To run them before merging a PR, you have to add the
label ``ready to merge`` to the PR. The tests will automatically launch. To rerun, remove and re-add the label.

**Working in "development mode"**

To implement changes to the CEBRA package from your system and use them without having to rebuild the Python package, the ``-e`` or ``--editable`` option
can be used with ``pip`` by running the following, in the root of your project directory:

.. code:: bash

    $ pip install -e .

It will link the package to the local location, basically meaning any changes to the local package will reflect directly in your environment.

Adding a Demo Jupyter Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New demo notebooks should be placed in ``demo_notebooks/`` folder.
Then the new notebook needs to be added to the Demo Notebooks ``toctree`` in ``demo_notebooks/README.rst``.
For that, extend the ``toctree`` (at the end of the file) using the following template:

.. code:: bash

    <notebook_title> <demo_notebooks/<notebook_name>.ipynb>

**Example:**

.. code:: bash

    Rat hippocampus <demo_notebooks/Demo_hippocampus.ipynb>


Building the Python package (information for maintainers only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To prepare or build a new release, please follow the following steps.

If the version changes, adjust the version in ``cebra.__version__`` directly. No additional update to the package
file is needed. CEBRA uses `Semantic Versioning <https://semver.org/spec/v2.0.0.html>` to denote versions.

Enter the build environment and build the package:

.. code:: bash

    host $ make interact
    docker $ make build
    # ... outputs ...
    Successfully built cebra-X.X.XaX-py2.py3-none-any.whl

The built package can be found in ``dist/`` and can be installed locally with

.. code:: bash

    pip install dist/cebra-X.X.XaX-py2.py3-none-any.whl

**Please do not distribute this package prior to the public release of the CEBRA repository, because it also
contains parts of the source code.**
