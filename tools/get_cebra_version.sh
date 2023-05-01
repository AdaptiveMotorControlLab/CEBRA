#!/bin/bash
# Parses the cebra version

sed -ne 's/.*__version__.*"\([0-9a-z\.]\+\)"/\1/p' cebra/__init__.py
