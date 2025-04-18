#!/bin/bash
# Build, test and push cebra container.

set -e

# Parse command line arguments
RUN_FULL_TESTS=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --full-tests)
      RUN_FULL_TESTS=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z $(git status --porcelain) ]]; then
  TAG=$(git rev-parse --short HEAD)
else
  TAG=dev
fi

BASENAME=cebra
DOCKERNAME=$BASENAME:$TAG
LATEST=$BASENAME:latest
echo Building $DOCKERNAME

#docker login <your registry>

docker build \
--build-arg UID=$(id -u) \
--build-arg GID=$(id -g) \
--build-arg GIT_HASH=$(git rev-parse HEAD) \
	-t $DOCKERNAME .
docker tag $DOCKERNAME $LATEST

# Determine whether to run full tests or not
if [[ "$RUN_FULL_TESTS" == "true" ]]; then
  echo "Running full test suite including tests that require datasets"
else
  echo "Running tests that don't require datasets"
fi

docker run \
  --gpus 2 \
  ${extra_kwargs[@]} \
  -v ${CEBRA_DATADIR:-./data}:/data \
  --env CEBRA_DATADIR=/data \
  --network host \
  -it $DOCKERNAME python -m pytest --ff -x $([ "$RUN_FULL_TESTS" != "true" ] && echo '-m "not requires_dataset"') --doctest-modules ./docs/source/usage.rst tests cebra

#docker push $DOCKERNAME
#docker push $LATEST
