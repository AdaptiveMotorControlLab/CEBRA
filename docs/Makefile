# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= --keep-going -n
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help page page_rebuild figures

# Build the API documentation using sphinx
html:
	PYTHONPATH=.. $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build multiple versions
html_versions:
	for v in latest v0.2.0 v0.3.0 v0.4.0; do \
		PYTHONPATH=.. $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/$$v"; \
	done
# Remove the current temp folder and page build
clean:
	rm -rf build
	rm -rf page

# Checkout the source repository for CEBRA figures. Note that this requires SSH access
# and might prompt you for an SSH key.
source/cebra-figures:
	cd $(dir $(realpath $(firstword $(MAKEFILE_LIST)))) && git clone --depth 1 git@github.com:AdaptiveMotorControlLab/cebra-figures.git source/cebra-figures

source/demo_notebooks:
	cd $(dir $(realpath $(firstword $(MAKEFILE_LIST)))) && git clone --depth 1 git@github.com:AdaptiveMotorControlLab/cebra-demos.git source/demo_notebooks

source/demo_notebooks:
	git clone --depth 1 git@github.com:AdaptiveMotorControlLab/cebra-demos.git source/demo_notebooks

# Update the figures. Note that this might prompt you for an SSH key
figures: source/cebra-figures
	cd source/cebra-figures &&	git pull --ff-only origin main

demos: source/demo_notebooks
	cd source/demo_notebooks &&	git pull --ff-only origin main

source/assets:
	cd $(dir $(realpath $(firstword $(MAKEFILE_LIST)))) && git clone --depth 1 git@github.com:AdaptiveMotorControlLab/cebra-assets.git source/assets

assets: source/assets
	cd source/assets && git pull --ff-only origin main
	cp -r source/assets/docs/* .
	#rm -rf source/assets

# Build the page with pre-built figures
page: source/cebra-figures source/demo_notebooks html
	mkdir -p page/
	mkdir -p page/docs
	mkdir -p page/staging/docs
	cp -a root/. page/
	cp -r build/html/* page/docs/
	cp -r build/html/* page/staging/docs/

# Do a full rebuild, starting from a clean build and page directory, checking that the
# figure repo is up-to-date, and starting the page build
page_rebuild: clean figures page
