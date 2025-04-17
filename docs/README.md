# CEBRA documentation

This directory contains the documentation for CEBRA.

To build the docs, head to *the root folder of the repository* and run:

```bash
./build_docs.sh
```

This will build the docker container in [Dockerfile](Dockerfile) and run the `make docs` command from the root repo.
The exact requirements for building the docs are now listed in [requirements.txt](requirements.txt).

For easier local development, docs are not using `sphinx-autobuild` and will by default be served at [http://127.0.0.1:8000](http://127.0.0.1:8000).
