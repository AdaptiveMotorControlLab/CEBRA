#!/bin/bash

# Re-install the package. By running './reinstall.sh'
#
# Note that CEBRA uses the build
# system specified in
# PEP517 https://peps.python.org/pep-0517/ and
# PEP518 https://peps.python.org/pep-0518/
# and hence there is no setup.py file.

set -e # abort on error

pip uninstall -y cebra

# Get version info after uninstalling --- this will automatically get the
# most recent version based on the source code in the current directory.
# $(tools/get_cebra_version.sh)
VERSION=0.3.0
echo "Upgrading to CEBRA v${VERSION}"

# Upgrade the build system (PEP517/518 compatible)
python3 -m pip install virtualenv
python3 -m pip install --upgrade build
python3 -m build --sdist --wheel .

# Reinstall the package with most recent version
pip install --upgrade --no-cache-dir "dist/cebra-${VERSION}-py2.py3-none-any.whl[datasets,integrations]"
