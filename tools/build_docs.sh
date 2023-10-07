#!/bin/bash
# Locally build the documentation and display it in a webserver.

set -xe

git_checkout_or_pull() {
    local repo=$1
    local target_dir=$2
    # TODO(stes): theoretically we could also auto-update the repo,
    # I commented this out for now to avoid interference with local
    # dev/changes
    #if [ -d "$target_dir" ]; then
    #    cd "$target_dir"
    #    git pull --ff-only origin main
    #    cd -
    #else
    if [ ! -d "$target_dir" ]; then
        git clone "$repo" "$target_dir"
    fi
}

checkout_cebra_figures() {
    git_checkout_or_pull git@github.com:AdaptiveMotorControlLab/cebra-figures.git docs/source/cebra-figures
}

checkout_assets() {
    git_checkout_or_pull git@github.com:AdaptiveMotorControlLab/cebra-assets.git assets
}

checkout_cebra_demos() {
    git_checkout_or_pull git@github.com:AdaptiveMotorControlLab/cebra-demos.git docs/source/demo_notebooks
}

setup_python() {
    python -m pip install --upgrade pip setuptools wheel
    sudo apt-get install -y pandoc
    pip install torch --extra-index-url=https://download.pytorch.org/whl/cpu
    pip install '.[docs]'
}

build_docs() {
    cp -r assets/* .
    export SPHINXOPTS="-W --keep-going -n"
    (cd docs && PYTHONPATH=.. make page)
}

serve() {
    python -m http.server 8080 --b 0.0.0.0 -d docs/build/html
}

main() {
    build_docs
    serve
}

if [[ "$1" == "--build" ]]; then
    main
fi

docker build -t cebra-docs -f - . << "EOF"
FROM python:3.9

RUN python -m pip install --upgrade pip setuptools wheel \
    && apt-get update -y && apt-get install -y pandoc git

RUN pip install torch --extra-index-url=https://download.pytorch.org/whl/cpu \
    && pip install 'cebra[docs]' && pip uninstall -y cebra

EOF

checkout_cebra_figures
checkout_assets
checkout_cebra_demos

docker run \
    -p 127.0.0.1:8080:8080 \
    -u $(id -u):$(id -g) \
    -v .:/app -w /app \
    --tmpfs /.config --tmpfs /.cache \
    -it cebra-docs \
    ./tools/build_docs.sh --build
