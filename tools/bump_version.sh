#!/bin/bash
# Bump the CEBRA version to the specified value.
# Edits all relevant files at once.
#
# Usage:
# tools/bump_version.sh 0.3.1rc1

version=$1
if [ -z ${version} ]; then
    >&1 echo "Specify a version number."
    >&1 echo "Usage:"
    >&1 echo "tools/bump_version.sh <semantic version>"
    exit 1
fi

# Determine the correct sed command based on the OS
# On macOS, the `sed` command requires an empty string argument after `-i` for in-place editing.
# On Linux and other Unix-like systems, the `sed` command only requires `-i` for in-place editing.
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SED_CMD="sed -i .bkp -e"
else
    # Linux and other Unix-like systems
    SED_CMD="sed -i -e"
fi

# python cebra version
$SED_CMD "s/__version__ = .*/__version__ = \"${version}\"/" cebra/__init__.py

# reinstall script in root
$SED_CMD "s/VERSION=.*/VERSION=${version}/" reinstall.sh

# Makefile
$SED_CMD "s/CEBRA_VERSION := .*/CEBRA_VERSION := ${version}/" Makefile

# Arch linux PKGBUILD
$SED_CMD "s/pkgver=.*/pkgver=${version}/" PKGBUILD

# Dockerfile
$SED_CMD "s/ENV WHEEL=cebra-.*\.whl/ENV WHEEL=cebra-${version}-py2.py3-none-any.whl/" Dockerfile

# Remove backup files
if [[ "$OSTYPE" == "darwin"* ]]; then
    rm cebra/__init__.py.bkp reinstall.sh.bkp Makefile.bkp PKGBUILD.bkp Dockerfile.bkp
fi
