#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
import numpy as np
import pytest

import cebra.integrations.sklearn.decoder as cebra_sklearn_decoder


def test_imports():
    import cebra

    assert hasattr(cebra, "KNNDecoder")
    assert hasattr(cebra, "L1LinearRegressor")


def _iterate_decoders():
    decoders = []
    for param in cebra_sklearn_decoder.KNNDecoder.iter_hyperparams():
        decoders.append(
            cebra_sklearn_decoder.KNNDecoder(n_neighbors=param["n_neighbors"],
                                             metric=param["metric"]))
    for param in cebra_sklearn_decoder.L1LinearRegressor.iter_hyperparams():
        decoders.append(
            cebra_sklearn_decoder.L1LinearRegressor(alpha=param["alpha"]))
    return decoders


@pytest.mark.parametrize("decoder", _iterate_decoders())
def test_sklearn_decoder(decoder):
    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    y_c = np.random.uniform(0, 1, (1000))
    y_d = np.random.randint(0, 10, (1000,))
    y_c_dim = np.random.uniform(0, 1, (1000, 5))
    y_str = np.array(["test" for i in range(1000)])
    y_d_short = np.random.randint(0, 10, (500,))

    # continuous target
    decoder.fit(X, y_c)
    pred = decoder.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.dtype in (np.float32, np.float64)

    score = decoder.score(X, y_c)
    assert isinstance(score, float)

    # discrete target
    decoder.fit(X, y_d)
    pred = decoder.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.dtype in (np.int32, np.int64, np.float32, np.float64)

    score = decoder.score(X, y_d)
    assert isinstance(score, float)

    # multi-dim continuous target
    decoder.fit(X, y_c_dim)
    pred = decoder.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.dtype in (np.float32, np.float64)

    score = decoder.score(X, y_c_dim)
    assert isinstance(score, float)

    # multi-dim discrete and continuous target
    multi_y = np.concatenate([y_c_dim, np.expand_dims(y_d, axis=1)], axis=1)
    decoder.fit(X, multi_y)
    pred = decoder.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.dtype in (np.float32, np.float64)

    score = decoder.score(X, multi_y)
    assert isinstance(score, float)

    # invalid targets
    with pytest.raises(NotImplementedError, match="Invalid.*type"):
        decoder.fit(X, y_str)
    with pytest.raises(ValueError, match="Invalid.*shape"):
        decoder.fit(X, y_d_short)
