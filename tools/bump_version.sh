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
fi

# python cebra version
sed -i "s/__version__ = .*/__version__ = \"${version}\"/" \
    cebra/__init__.py

# reinstall script in root
sed -i "s/VERSION=.*/VERSION=${version}/" \
    reinstall.sh

# Makefile
sed -i "s/CEBRA_VERSION := .*/CEBRA_VERSION := ${version}/" \
    Makefile

# Arch linux PKGBUILD 
sed -i "s/pkgver=.*/pkgver=${version}/" \
    PKGBUILD 

# Dockerfile
sed -i "s/ENV WHEEL=cebra-.*\.whl/ENV WHEEL=cebra-${version}-py2.py3-none-any.whl/" \
    Dockerfile 
