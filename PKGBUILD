# Maintainer: Steffen Schneider <stes@hey.com>
pkgname=python-cebra
_pkgname=cebra
pkgver=0.5.0
pkgrel=1
pkgdesc="Consistent Embeddings of high-dimensional Recordings using Auxiliary variables"
url="https://cebra.ai"
depends=(
    python-joblib
    python-scikit-learn
    python-scipy
    python-pytorch
    python-tqdm
    python-joblib
    python-scipy
    python-literate-dataclasses
)
makedepends=(
    python-pytest-benchmark
)
optdepends=(
    python-werkzeug
    python-seaborn
    python-matplotlib
    python-h5py
    python-argparse
    python-plotly
)
license=('custom')
arch=('any')
source=("${_pkgname}-${pkgver}.tar.gz")
md5sums=('d51281b2cfc7257020c99e7a794a7fc4')

_site_packages=$(python -c 'import sys; print(sys.path[-1])')

build() {
    cd $srcdir/${_pkgname}-${pkgver}
	python3 -m build --wheel
}

package() {
    cd $srcdir/${_pkgname}-${pkgver}
    pip install --ignore-installed --no-deps --root="${pkgdir}" dist/${_pkgname}-${pkgver}-py3-none-any.whl
    find ${pkgdir} -iname __pycache__ -exec rm -r {} \; 2>/dev/null || echo
    install -Dm 644 LICENSE.md $pkgdir/usr/share/licenses/${pkgname}/LICENSE
}

#check() {
#    PYTHONPATH=${pkgdir}/${_site_packages} \
#      python -m pytest \
#      -c $srcdir/${_pkgname}-${pkgver}/pytest.ini \
#      --ff -m "not requires_dataset" \
#      $srcdir/${_pkgname}-${pkgver}/tests
#}
