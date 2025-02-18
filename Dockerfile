## EXPERIMENT BASE CONTAINER
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cebra-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
	&& apt-get install --no-install-recommends -yy git python3 python3-pip python-is-python3 \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --upgrade pip


## GIT repository
FROM ubuntu AS repo

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
	&& apt-get install --no-install-recommends -yy git python3 python3-pip python-is-python3 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY . /repo
RUN git status --porcelain

WORKDIR /target
RUN git clone --filter=tree:0 --depth=1 file:///repo/.git /target
RUN git log

## CEBRA WHEEL
FROM python:3.9 AS wheel

RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir build virtualenv

WORKDIR /build
COPY --from=repo /target .
RUN make dist

## CEBRA CONTAINER
FROM cebra-base

# install the cebra wheel
ENV WHEEL=cebra-0.5.0rc1-py3-none-any.whl
WORKDIR /build
COPY --from=wheel /build/dist/${WHEEL} .
RUN pip install --no-cache-dir ${WHEEL}'[dev,integrations,datasets]'
RUN rm -rf /build

# add the repository
WORKDIR /app
COPY --from=repo /target .

ENV PYTHONPATH=/app

ARG UID
ARG GID
RUN groupadd -g $GID appgroup
RUN useradd -u $UID -g $GID -ms "/bin/bash" appuser
RUN chown -R appuser:appgroup /app
USER appuser

ARG GIT_HASH
RUN if [ "$(git rev-parse HEAD)" != "${GIT_HASH}" ]; then \
	git rev-parse HEAD; \
	echo ${GIT_HASH}; \
	exit 1; \
    fi
RUN git status --porcelain || exit 1
