Installation Guide
==================

System Requirements
-------------------


- Software dependencies and operating systems:
    - Linux or MacOS
- Versions software has been tested on:
- Required non-standard hardware
    - GPU (optional)


Installation Guide
------------------

We outline installation instructions for different systems. 


    * ``[integrations]``: This will install (experimental) support for our streamlit and jupyter integrations.
    * ``[docs]``: This will install additional dependencies for building the package documentation.
    * ``[dev]``: This will install additional dependencies for development, unit and integration testing,
      code formatting, etc. Install this extension if you want to work on a pull request.
    * ``[datasets]``: This extension will install additional dependencies to use the pre-installed datasets

.. tabs::


        Conda users should currently use ``pip`` for installation. The missing dependencies will be installed in the install process. A fresh conda environment can be created using 

        .. code:: bash

            $ conda create -n cebra python==3.8
            $ conda activate cebra

        It is recommended to install PyTorch manually given your system setup. To select the right version, head to
        CPU only). Then, use the command to install the PyTorch package. Below are a few possible examples (as of 23/8/22):

        .. code:: bash

            # CPU only version of pytorch, using the latest version
            $ conda install pytorch cpuonly -c pytorch
            # GPU version of pytorch for CUDA 11.3
            $ conda install pytorch cudatoolkit=11.3 -c pytorch
            # CPU only version of pytorch, using the pytorch LTS version
            $ conda install pytorch cpuonly -c pytorch-lts

        Once PyTorch is set up, the remaining dependencies can be installed via ``pip``. Select the correct feature

        .. code:: bash
            $ pip install '.[dev]'

        .. note::

    .. tab:: Docker

        It is possible to start a full development environment by running ``make interact``.
        Alternatively, the container can be build locally. Refer to ``make docker`` in the ``Makefile``.

        .. code:: bash

            $ make docker 
            $ make interact 
            $ make test
        Several arguments can be used to configure the docker container.
        ``docker-10.1-runtime-ubuntu18.04``, ``docker-10.2-runtime-ubuntu18.04`` and ``docker-11.1-runtime-ubuntu20.04``, but more images can be easily added by modifying the Dockerfile.

        A particular version can be built and run by executing

        .. code:: bash

            $ make docker-11.1-runtime-ubuntu20.04
            $ make interact-11.1-runtime-ubuntu20.04

        All images are based on the official `Nvidia CUDA Docker images <https://hub.docker.com/r/nvidia/cuda>`_.

    .. tab:: pip

        .. note::
        .. code:: bash
            





        .. code:: bash

            $ pip install .


        .. code:: bash

            $ pip install -e '.[dev,docs,integrations,datasets]'

..









Installation Troubleshooting
----------------------------


.. _PyTorch Docs: https://pytorch.org/
.. _virtual environment: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment
