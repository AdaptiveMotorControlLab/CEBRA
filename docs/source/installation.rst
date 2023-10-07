Installation Guide
==================

System Requirements
-------------------

CEBRA is written in Python (3.8+) and PyTorch. CEBRA is most effective when used with a GPU, but CPU-only support is provided. We provide instructions to run CEBRA on your system directly.  The instructions below were tested on different compute setups with Ubuntu 18.04 or 20.04, using Nvidia GTX 2080, A4000, and V100 cards. Other setups are possible (including Windows), as long as CUDA 10.2+ support is guaranteed.

- Software dependencies and operating systems:
    - Linux or MacOS
- Versions software has been tested on:
    - Linux (Ubuntu 20.04, 18.04, MacOS 12.1-5)
- Required non-standard hardware
    - GPU (optional)


Installation Guide
------------------

We outline installation instructions for different systems.
CEBRA will be installed via ``pip install cebra``.

CEBRA's dependencies can be installed using ``pip`` or ``conda`` and
we outline different options below.

.. tabs::

    .. tab:: Google Colab

        CEBRA can also be installed and run on Google Colaboratory.
        Please see the ``open in colab`` button at the top of each `demo notebook <https://cebra.ai/docs/demos.html>`_ for examples.

        If you are starting with a new notebook, simply run

        .. code:: bash

            ! pip install cebra

        In the first cell.


    .. tab:: Supplied conda (CEBRA)

        A ``conda`` environment for running CEBRA is provided in the ``conda`` sub-directory.
        To built the env, please run from the CEBRA repo root directory:

        .. code:: bash

            $ conda env create -f conda/cebra.yml

    .. tab:: conda

        Conda users should currently use ``pip`` for installation. The missing dependencies will be installed in the install process. A fresh conda environment can be created using

        .. code:: bash

            $ conda create -n cebra python==3.8
            $ conda activate cebra

        .. rubric:: Install PyTorch separately

        It is recommended to install PyTorch manually given your system setup. To select the right version, head to
        the "Install PyTorch" instructions in the official `PyTorch Docs`_. Select your desired PyTorch build, operating system,
        select ``conda`` as your package manager and ``Python`` as the language. Select your compute platform (either a CUDA version or
        CPU only). Then, use the command to install the PyTorch package. Below are a few possible examples (as of 23/8/22):

        .. code:: bash

            # CPU only version of pytorch, using the latest version
            $ conda install pytorch cpuonly -c pytorch

        .. code:: bash

            # GPU version of pytorch for CUDA 11.3
            $ conda install pytorch cudatoolkit=11.3 -c pytorch

        .. code:: bash

            # CPU only version of pytorch, using the pytorch LTS version
            $ conda install pytorch cpuonly -c pytorch-lts

        .. rubric:: Install CEBRA using ``pip``

        Once PyTorch is set up, the remaining dependencies can be installed via ``pip``. Select the correct feature
        set based on your usecase:

        * Regular usage

        .. code:: bash

            $ pip install cebra

        * 🚀 For more advanced users, CEBRA has different extra install options that you can select based on your usecase:

            * ``[integrations]``: This will install (experimental) support for our streamlit and jupyter integrations.
            * ``[docs]``: This will install additional dependencies for building the package documentation.
            * ``[dev]``: This will install additional dependencies for development, unit and integration testing,
              code formatting, etc. Install this extension if you want to work on a pull request.
            * ``[demos]``: This will install additional dependencies for running our demo notebooks.
            * ``[datasets]``: This extension will install additional dependencies to use the pre-installed datasets
              in ``cebra.datasets``.

        * Inference and development tools only

        .. code:: bash

            $ pip install '.[dev]'

        * Full feature set

        .. code:: bash

            $ pip install '.[dev,docs,integrations,demos,datasets]'

        Note that, similarly to that last command, you can select the specific install options of interest based on their description above and on your usecase.

        .. note::
            On windows systems, you will need to drop the quotation marks and install via ``pip install .[dev]``.

    .. tab:: pip

        .. note::
            Consider using a `virtual environment`_ when installing the package via ``pip``.

        *(Optional)* Create the virtual environment by running

        .. code:: bash

            $ virtualenv .env && source .env/bin/activate

        We recommend that you install ``PyTorch`` before CEBRA by selecting the correct version in the `PyTorch Docs`_. Select your desired PyTorch build, operating
        system, select ``pip`` as your package manager and ``Python`` as the language. Select your compute platform (either a
        CUDA version or CPU only). Then, use the command to install the PyTorch package. See the ``conda`` tab for examples.

        Then you can install  CEBRA, by running one of these lines, depending on your usage, in the root directory.

        * For **regular usage**, the PyPi package can be installed using

        .. code:: bash

            $ pip install cebra

        * For a full install, run

        .. code:: bash

            $ pip install 'cebra[dev,integrations,datasets]'

        Note that, similarly to that last command, you can select the specific install options of interest based on their description above and on your usecase.

..



.. Post-Installation
.. -----------------

.. After installing CEBRA using any of the guides above, please verify the installation by running the test suite.

.. .. code:: bash

..     $ make test

.. No tests should fail.
.. If this is the case, the installation was successful.


Installation Troubleshooting
----------------------------

If yopu have issues installing CEBRA, we recommend carefully checking the `traceback`_ which can help you look on `stackoverflow`_ or the popular-in-life-sciences, `Image Forum`_ for similar issues. If you cannot find a solution, please do post an issue on GitHub!

Advanced Installation for Schneider, Lee, Mathis 2023 paper experiments
-----------------------------------------------------------------------

If you want to install the additional dependencies required to run comparisons with other algorithms, please see the following:

.. tabs::
   .. tab:: Supplied conda (paper reproduction)

        We provide a ``conda`` environment with the full requirements needed to reproduce the first CEBRA paper (although we
        recommend using Docker). Namely, you can run CEBRA, piVAE, tSNE and UMAP within this conda env. It is *NOT* needed if you only want to use CEBRA.

        * For all platforms except MacOS with M1/2 chipsets, create the full environment using ``cebra_paper.yml``, by running the following from the CEBRA repo root directory:

            .. code:: bash

                $ conda env create -f conda/cebra_paper.yml

        * If you are a MacOS M1 or M2 user and want to reproduce the paper, use the ``cebra_paper_m1.yml`` instead. You'll need to install tensorflow. For that, use `miniconda3 <https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_ and follow the setup instructions for tensorflow listed in the `Apple developer docs <https://developer.apple.com/metal/tensorflow-plugin/>`_. In the Terminal, run the following commands:

            .. code:: bash

                wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-MacOSX-arm64.sh -O ~/miniconda.sh
                bash ~/miniconda.sh -b -p $HOME/miniconda
                source ~/miniconda/bin/activate
                conda init zsh

            Then, you can build the full environment from the root directory:

            .. code:: bash

                $ conda env create -f conda/cebra_paper_m1.yml


.. _PyTorch Docs: https://pytorch.org/
.. _virtual environment: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment
.. _traceback: https://realpython.com/python-traceback/
.. _stackoverflow: https://stackoverflow.com/
.. _Image Forum: https://forum.image.sc/
