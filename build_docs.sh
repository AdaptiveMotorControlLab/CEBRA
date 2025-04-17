#!/bin/bash

docker build -t cebra-docs -f docs/Dockerfile .
docker run -u $(id -u):$(id -g) \
  -p 127.0.0.1:8000:8000 \
  -v $(pwd):/app \
  -v /tmp/.cache/pip:/.cache/pip \
  -v /tmp/.cache/sphinx:/.cache/sphinx \
  -v /tmp/.cache/matplotlib:/.cache/matplotlib \
  -v /tmp/.cache/fontconfig:/.cache/fontconfig \
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib \
  -w /app \
  --env HOST=0.0.0.0 \
  --env PORT=8000 \
  -it cebra-docs \
  make docs
