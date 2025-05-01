#!/bin/bash

DOCKER_BUILDKIT=1 docker build --output type=local,dest=. .
