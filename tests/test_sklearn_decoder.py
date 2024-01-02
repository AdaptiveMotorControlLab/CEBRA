#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import pytest
import torch

import cebra.helper
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
    assert np.issubdtype(pred.dtype, np.floating)

    score = decoder.score(X, y_c)
    assert isinstance(score, float)

    # torch
    decoder.fit(torch.Tensor(X), torch.Tensor(y_c))
    pred = decoder.predict(torch.Tensor(X))
    assert isinstance(pred, np.ndarray)
    assert np.issubdtype(pred.dtype, np.floating)

    # discrete target
    decoder.fit(X, y_d)
    pred = decoder.predict(X)
    assert isinstance(pred, np.ndarray)
    assert np.issubdtype(pred.dtype, np.integer) or np.issubdtype(
        pred.dtype, np.floating)

    score = decoder.score(X, y_d)
    assert isinstance(score, float)

    # torch
    decoder.fit(torch.Tensor(X), torch.Tensor(y_d))
    pred = decoder.predict(torch.Tensor(X))
    assert isinstance(pred, np.ndarray)
    assert np.issubdtype(pred.dtype, np.integer) or np.issubdtype(
        pred.dtype, np.floating)

    # multi-dim continuous target
    decoder.fit(X, y_c_dim)
    pred = decoder.predict(X)
    assert isinstance(pred, np.ndarray)
    assert np.issubdtype(pred.dtype, np.floating)

    score = decoder.score(X, y_c_dim)
    assert isinstance(score, float)

    # multi-dim discrete and continuous target
    multi_y = np.concatenate([y_c_dim, np.expand_dims(y_d, axis=1)], axis=1)
    decoder.fit(X, multi_y)
    pred = decoder.predict(X)
    assert isinstance(pred, np.ndarray)
    assert np.issubdtype(pred.dtype, np.floating)

    score = decoder.score(X, multi_y)
    assert isinstance(score, float)

    # invalid targets
    with pytest.raises(NotImplementedError, match="Invalid.*type"):
        decoder.fit(X, y_str)
    with pytest.raises(ValueError, match="Invalid.*shape"):
        decoder.fit(X, y_d_short)


def test_dtype_checker():
    assert cebra.helper._is_floating(torch.Tensor([4.5]))
    assert cebra.helper._is_integer(torch.LongTensor([4]))
    assert cebra.helper._is_floating(np.array([4.5]))
    assert cebra.helper._is_integer(np.array([4]))
