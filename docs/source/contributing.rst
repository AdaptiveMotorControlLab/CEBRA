Contribution Guide
==================

Development setup
^^^^^^^^^^^^^^^^^

Development should be done inside the provided docker environment.
All essential commands are included in the project ``Makefile``.

.. code:: bash

    $ make interact


.. code:: bash

    $ make test


.. code:: bash

    $ make test_debug


.. code:: bash

    $ make docs

Code is formatted using `Google code style <https://google.github.io/styleguide/pyguide.html>`_, but with 4 spaces.
The specification is in ``.style.yapf`` and ``.style.yapf``.
We use ``yapf`` for automated formatting and ``isort`` for import statement formatting.

Formatting the whole code base can be done with 

.. code:: bash

    $ make format

For in-depth information on how to adapt and contribute to CEBRA, please refer to the full documentation.

    $ pip install -e .
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
