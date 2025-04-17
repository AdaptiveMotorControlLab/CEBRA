CEBRA_VERSION := 0.6.0rc1

dist:
	python3 -m pip install virtualenv
	python3 -m pip install --upgrade build twine
	python3 -m build --wheel --sdist

build: dist

archlinux:
	mkdir -p dist/arch
	cp PKGBUILD dist/arch
	cp dist/cebra-${CEBRA_VERSION}.tar.gz dist/arch
	(cd dist/arch; makepkg --skipchecksums -f)

# NOTE(stes): Ensure that no old tempfiles are present. Ideally, move this into
# the test pipeline directly.
clean_test:
	rm -rf auxiliary_behavior_data.h5 data.npz neural_data.npz saved_models grid_search_models

test: clean_test
	python -m pytest --ff --doctest-modules -m "not requires_dataset" tests ./docs/source/usage.rst cebra

doctest: clean_test
	python -m pytest --ff --doctest-modules -vvv ./docs/source/usage.rst

docker:
	./tools/build_docker.sh

test_parallel: clean_test
	python -m pytest -n auto --ff -m "not requires_dataset"  tests

test_parallel_debug: clean_test
	python -m pytest -n auto -x --ff -m "not requires_dataset"  tests

test_all: clean_test
	python -m pytest --ff --ignore cebra/grid_search.py tests cebra

test_fast: clean_test
	python -m pytest --ff --ignore cebra/grid_search.py -m "not requires_dataset" tests cebra --runfast

# Run failed test firsts, using a single worker (for debugging)
test_debug: clean_test
	python -m pytest -vvv -x --ff -m "not requires_dataset" tests

test_benchmark: clean_test
	python -m pytest --ff -m "not requires_dataset" tests --runbenchmark

interrogate:
	interrogate \
		--ignore-property-decorators \
		--ignore-init-method \
		--verbose \
		--ignore-semiprivate \
		--ignore-private \
		--ignore-magic \
		--omit-covered-files \
		-f 90 \
		cebra

# Build documentation using sphinx
docs:
	(cd docs && PYTHONPATH=.. make page)

docs-touch:
	find docs/source -iname '*.rst' -exec touch {} \;
	(cd docs && PYTHONPATH=.. make page)

docs-strict:
	(cd docs && PYTHONPATH=.. SPHINXOPTS='-n' make page)

# Serve the docs
serve_docs:
	python -m http.server 8080 --b 127.0.0.1 -d docs/build/html

# Serve the entire page
serve_page:
	python -m http.server 8080 --b 127.0.0.1 -d docs/page

# Format code in the main package and docs
format:
	yapf -i -p -r cebra
	yapf -i -p -r tests
	yapf -i -p -r docs/source/conf.py
	# TODO(stes) Add back once upstream issue
	# https://github.com/PyCQA/docformatter/issues/119
	# is resolved.
	# docformatter --config pyproject.toml -i cebra
	# docformatter --config pyproject.toml -i tests
	isort cebra/
	isort tests/

codespell:
	codespell cebra/ tests/ docs/source/*.rst *.md -L "nce, nd"

check_for_binary:
	./tools/check_for_binary_files.sh

# Show code report (pylint and coverage)
report: check_docker format .coverage .pylint
	cat .pylint
	coverage report

.PHONY: dist build docker archlinux clean_test test doctest test_parallel \
	test_parallel_debug test_all test_fast test_debug test_benchmark \
	interrogate docs docs-touch docs-strict serve_docs serve_page \
	format codespell check_for_binary
