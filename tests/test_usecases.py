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
"""Integrations tests using various usecases via the sklearn API.

All attempts to reproduce bugs in the sklearn API should be added to this
file. Existing tests should **not** be changed to ensure backward compatibility.

If breaking changes are introduced, add decorators that skip tests based on the
cebra version we test against.
"""

import itertools
import pickle

import numpy as np
import pytest
import torch

import cebra
import cebra.integrations.sklearn.dataset as cebra_sklearn_dataset


def _default_kwargs():
    return dict(device="cpu", max_iterations=10, verbose=False)


def _make_data():
    spikes = np.random.normal(0, 1, size=(1000, 10))
    behavior = np.random.normal(0, 1, size=(1000, 5))
    return spikes, behavior


def _make_data_torch():
    spikes = torch.randn(size=(1000, 10))
    behavior = torch.randn(size=(1000, 5))
    return spikes, behavior


def _run_test(model, behavior=False):
    neural, index = _make_data()
    print(index)
    if behavior:
        model.fit(neural, index)
    else:
        model.fit(neural)
    embedding = model.transform(neural)
    assert len(embedding) == len(neural)
    assert len(embedding.shape) == 2


def test_minimal():
    model = cebra.CEBRA("offset10-model",
                        output_dimension=2,
                        time_offsets=10,
                        **_default_kwargs())
    _run_test(model)
    _run_test(model, behavior=True)


def test_full():
    model = cebra.CEBRA("offset10-model",
                        output_dimension=3,
                        time_offsets=(10,),
                        conditional="time_delta",
                        batch_size=300,
                        **_default_kwargs())
    _run_test(model)
    _run_test(model, behavior=True)


def test_hybrid():
    model = cebra.CEBRA("offset10-model",
                        output_dimension=3,
                        time_offsets=(10,),
                        conditional="time_delta",
                        batch_size=300,
                        hybrid=True,
                        **_default_kwargs())
    _run_test(model, behavior=True)
    with pytest.raises(ValueError):
        _run_test(model, behavior=False)


_args = [
    dict(
        model_architecture="offset1-model",
        time_offsets=10,
        batch_size=200,
        learning_rate=3e-4,
        num_hidden_units=64,
        output_dimension=8,
        temperature=1,
    ),
    dict(
        model_architecture="offset1-model",
        batch_size=200,
        learning_rate=3e-4,
        num_hidden_units=64,
        temperature=1,
    ),
    dict(model_architecture="offset1-model", time_offsets=10),
]
_additions = [
    dict(),
]
_args = list(
    dict(**arg, **addition)
    for arg, addition in itertools.product(_args, _additions))


@pytest.mark.parametrize(
    "leave_out,args",
    [(leave_out, args) for args in _args for leave_out in args.keys()])
def test_leave_arg_out(leave_out, args):
    model = cebra.CEBRA(**{k: v for k, v in args.items() if k != leave_out},
                        **_default_kwargs())
    _run_test(model)


def test_defaults():
    model = cebra.CEBRA(max_iterations=10)
    _run_test(model)


def test_dataset():
    X = np.zeros((100, 3))
    y = np.zeros((100, 3))
    cebra_sklearn_dataset.SklearnDataset(X, y=())
    with pytest.raises(ValueError):
        cebra_sklearn_dataset.SklearnDataset(X, y=None)
    with pytest.raises(ValueError):
        cebra_sklearn_dataset.SklearnDataset(X, (y,), device="foo")
    with pytest.raises(ValueError):
        cebra_sklearn_dataset.SklearnDataset(X=None, y=(y,))

    dataset = cebra_sklearn_dataset.SklearnDataset(X, y=(y,))
    assert dataset.input_dimension == X.shape[1]
    assert torch.allclose(dataset.continuous_index, torch.from_numpy(y).float())
    assert dataset.continuous_index_dimensions == y.shape[1]
    assert dataset.total_index_dimensions == y.shape[1]


def test_incompatible():
    X, y = _make_data()

    # hybrid training is not possible without passing
    # and index
    model = cebra.CEBRA(hybrid=True)
    with pytest.raises(ValueError):
        model.fit(X)

    # multisession training is not possible without passing
    # an index
    model = cebra.CEBRA()
    with pytest.raises(RuntimeError):
        model.fit([X, X])


# @pytest.mark.timeout(3)
# def test_no_cuda():
#    # Then initializing a new model can be done by running:
#    #if torch.cuda.is_available():
#    #    pytest.skip("Test only useful when running on CPU.")
#    model = cebra.CEBRA(
#        "offset1-model",
#        **_default_kwargs()
#    )
#    _run_test(model)
