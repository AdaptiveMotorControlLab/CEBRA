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
import itertools
import tempfile
import warnings

import _util
import numpy as np
import pytest
import sklearn.utils.estimator_checks
import torch

import cebra.data as cebra_data
import cebra.integrations.sklearn.cebra as cebra_sklearn_cebra
import cebra.integrations.sklearn.dataset as cebra_sklearn_dataset
import cebra.models

if torch.cuda.is_available():
    _DEVICES = "cpu", "cuda"
else:
    _DEVICES = ("cpu",)


def test_imports():
    import cebra

    assert hasattr(cebra, "CEBRA")


def test_sklearn_dataset():
    X = np.zeros((100, 5), dtype="float32")
    yc = np.zeros((100, 5), dtype="float32")
    yd = np.zeros((100,), dtype="int")

    # cannot create datasets with more than one 1D discrete index
    with pytest.raises(ValueError):
        cebra_sklearn_dataset.SklearnDataset(X, (yd, yd), device="cpu")
    with pytest.raises(ValueError):
        cebra_sklearn_dataset.SklearnDataset(X, (np.stack([yd, yd], axis=1),),
                                             device="cpu")

    # need to pass iterable as type ...
    with pytest.raises(TypeError):
        cebra_sklearn_dataset.SklearnDataset(X, yc, device="cpu")
    with pytest.raises(TypeError):
        cebra_sklearn_dataset.SklearnDataset(X, yd, device="cpu")
    with pytest.raises(TypeError):
        cebra_sklearn_dataset.SklearnDataset(X, [yd], device="cpu")
    with pytest.raises(TypeError):
        cebra_sklearn_dataset.SklearnDataset(X, [[yd]], device="cpu")
    with pytest.raises(TypeError):
        cebra_sklearn_dataset.SklearnDataset(X, [[[yd]]], device="cpu")
    with pytest.raises(ValueError):
        cebra_sklearn_dataset.SklearnDataset(X, ((yd,),), device="cpu")

    # ... but any iterable needs to work
    arg = ()
    cebra_sklearn_dataset.SklearnDataset(X, arg, device="cpu")
    arg = (yd,)
    cebra_sklearn_dataset.SklearnDataset(X, arg, device="cpu")
    arg = (yc,)
    cebra_sklearn_dataset.SklearnDataset(X, arg, device="cpu")
    arg = (yc, yc, yd)
    cebra_sklearn_dataset.SklearnDataset(X, arg, device="cpu")

    # checking data input and indexing ops
    for labels in [(), (yc,), (yd,), (yc, yd)]:
        data = cebra_sklearn_dataset.SklearnDataset(X, labels, device="cpu")
        assert data.input_dimension == X.shape[1]
        assert len(data) == len(X)
        assert data[torch.arange(10)].shape == X[:10, ..., np.newaxis].shape
        assert torch.allclose(data[torch.arange(10)],
                              torch.from_numpy(X[:10, ..., np.newaxis]))

    # check indexing
    for is_cont, is_disc in [[True, False]] * 2:
        arg = []
        if is_cont:
            arg.append(yc)
        if is_disc:
            arg.append(yd)
        arg = tuple(arg)
        data = cebra_sklearn_dataset.SklearnDataset(X, arg, device="cpu")
        assert (data.continuous_index is not None) == is_cont
        assert (data.discrete_index is not None) == is_disc

    # multisession
    num_sessions = 3

    sessions = []
    for i in range(num_sessions):
        sessions.append(cebra_sklearn_dataset.SklearnDataset(X, (yc,)))
    data = cebra_data.datasets.DatasetCollection(*sessions)
    assert data.num_sessions == num_sessions
    for i in range(num_sessions):
        assert data.get_input_dimension(i) == X.shape[1]
        assert len(data.get_session(i)) == len(X)

    with pytest.raises(ValueError):
        cebra_data.datasets.DatasetCollection(())
    with pytest.raises(ValueError):
        cebra_data.datasets.DatasetCollection(sessions)

    sessions = []
    for i in range(num_sessions):
        sessions.append(cebra_sklearn_dataset.SklearnDataset(X, (yd,)))
    with pytest.raises(NotImplementedError):
        cebra_data.datasets.DatasetCollection(*sessions)


@_util.parametrize_slow(
    arg_names="is_cont,is_disc,is_full,is_multi,is_hybrid",
    fast_arguments=list(
        itertools.islice(itertools.product(*[[False, True]] * 5), 1)),
    slow_arguments=list(itertools.product(*[[False, True]] * 5)),
)
def test_init_loader(is_cont, is_disc, is_full, is_multi, is_hybrid):
    if is_multi:
        # TODO(celia): change to a MultiDemoDataset class when discrete/mixed index implemented
        class __Dataset(cebra.datasets.MultiContinuous):
            neural = torch.zeros((50, 10), dtype=torch.float)
            continuous_index = torch.zeros((50, 10), dtype=torch.float)
            discrete_index = torch.zeros((50,), dtype=torch.int)

    else:

        class __Dataset(cebra.datasets.DemoDataset):
            neural = torch.zeros((50, 10), dtype=torch.float)
            continuous_index = torch.zeros((50, 10), dtype=torch.float)
            discrete_index = torch.zeros((50,), dtype=torch.int)

    shared_kwargs = dict(num_steps=5, dataset=__Dataset())
    extra_kwargs = dict(batch_size=512, time_offsets=10, delta=0.01)

    try:
        loader, solver = cebra_sklearn_cebra._init_loader(
            is_cont=is_cont,
            is_disc=is_disc,
            is_full=is_full,
            is_multi=is_multi,
            is_hybrid=is_hybrid,
            shared_kwargs=shared_kwargs,
            extra_kwargs=extra_kwargs,
        )
        assert isinstance(loader, cebra.data.Loader)
        assert isinstance(solver, str)
    except Exception as e:
        with pytest.raises((NotImplementedError, ValueError)):
            raise e


def iterate_models():
    # architecture checks
    for model_architecture, device, distance in itertools.product(
            cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
            _DEVICES,
        ["euclidean", "cosine"],
    ):
        yield cebra_sklearn_cebra.CEBRA(
            model_architecture=model_architecture,
            pad_before_transform=
            True,  # NOTE(stes) needs to be true, otherwise not sklearn compatible.
            device=device,
            distance=distance,
            time_offsets=5,
            max_iterations=5,
            batch_size=10,
        )

    # parameter checks
    for (
            model_architecture,
            device,
            distance,
            temperature_mode,
            min_temperature,
            temperature,
    ) in itertools.product(
        [
            "offset10-model", "offset10-model-mse", "offset1-model",
            "resample-model"
        ],
            _DEVICES,
        ["euclidean", "cosine"],
        ["auto", "constant"],
        [None, 0.1],
        [0.1, 1.0],
    ):
        yield cebra_sklearn_cebra.CEBRA(
            model_architecture=model_architecture,
            pad_before_transform=
            True,  # NOTE(stes) needs to be true, otherwise not sklearn compatible.
            device=device,
            distance=distance,
            min_temperature=min_temperature,
            temperature_mode=temperature_mode,
            time_offsets=5,
            max_iterations=5,
            batch_size=10,
        )


@_util.parametrize_with_checks_slow(
    fast_arguments=list(itertools.islice(iterate_models(), 1)),
    slow_arguments=list(iterate_models()),
)
def test_api(estimator, check):
    num_retries = 1
    if check.func == sklearn.utils.estimator_checks.check_fit_idempotent:
        pytest.skip("CEBRA is non-deterministic.")
    if (check.func == sklearn.utils.estimator_checks.
            check_methods_sample_order_invariance):
        num_retries = 1000
    # if estimator.model_architecture == 'offset5-model':
    #    if check.func == sklearn.utils.estimator_checks.check_methods_sample_order_invariance:
    #        pytest.skip(
    #            "Output of fully convolutional models is not permutation invariant."
    #        )
    #    if check.func == sklearn.utils.estimator_checks.check_methods_subset_invariance:
    #        pytest.skip(
    #            "Output of fully convolutional models is not subset invariant.")
    if len(estimator._compute_offset()) > 10:
        pytest.skip(f"Model architecture {estimator.model_architecture} "
                    f"requires longer input sizes than 20 samples.")

    success = True
    exception = None
    num_successful = 0
    total_runs = 0
    for _ in range(num_retries):
        total_runs += 1
        try:
            check(estimator)
            num_successful += 1
        except AssertionError as e:
            exception = e
        if total_runs > num_retries // 10:
            if num_successful > 0:
                break
    if num_successful == 0:
        raise exception
    if exception is not None:
        warnings.warn(
            UserWarning(
                f"At least one repeat of the test raised an assertion error. "
                f"Test has a success probability of {num_successful}/{total_runs} = {100. * num_successful/total_runs:.2f}%."
            ))


@_util.parametrize_slow(
    arg_names="model_architecture,device",
    fast_arguments=list(
        itertools.islice(
            itertools.product(
                cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
                _DEVICES),
            2,
        )),
    slow_arguments=list(
        itertools.product(
            cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
            _DEVICES)),
)
def test_sklearn(model_architecture, device):
    output_dimension = 4
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture,
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=5,
        device=device,
        output_dimension=output_dimension,
        batch_size=42,
        verbose=True,
    )

    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    X_s2 = np.random.uniform(0, 1, (800, 30))
    X_s3 = np.random.uniform(0, 1, (1000, 30))
    y_c1 = np.random.uniform(0, 1, (1000, 5))
    y_c1_s2 = np.random.uniform(0, 1, (800, 5))
    y_c2 = np.random.uniform(0, 1, (1000, 2))
    y_c2_s2 = np.random.uniform(0, 1, (800, 2))
    y_d = np.random.randint(0, 10, (1000,))
    y_d_s2 = np.random.randint(0, 10, (800,))

    # time contrastive
    cebra_model.fit(X)
    assert cebra_model.num_sessions is None
    embedding = cebra_model.transform(X)
    assert isinstance(embedding, np.ndarray)

    # continuous behavior contrastive
    cebra_model.fit(X, y_c1, y_c2)
    assert cebra_model.num_sessions is None

    embedding = cebra_model.transform(X)
    assert isinstance(embedding, np.ndarray)
    embedding = cebra_model.transform(torch.Tensor(X))
    assert isinstance(embedding, np.ndarray)
    embedding = cebra_model.transform(X, session_id=0)
    assert isinstance(embedding, np.ndarray)

    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = cebra_model.transform(X, session_id=2)
    with pytest.raises(ValueError, match="Invalid.*labels"):
        cebra_model.fit(X, [y_c1, y_c1_s2])
    with pytest.raises(ValueError, match="Invalid.*samples"):
        cebra_model.fit(X_s3, y_c1_s2)

    # tensor input
    cebra_model.fit(torch.Tensor(X), torch.Tensor(y_c1), torch.Tensor(y_c2))

    # discrete behavior contrastive
    cebra_model.fit(X, y_d)
    embedding = cebra_model.transform(X)
    assert isinstance(embedding, np.ndarray)

    # mixed
    cebra_model.fit(X, y_c1, y_c2, y_d)
    embedding = cebra_model.transform(X)
    assert isinstance(embedding, np.ndarray)

    # multi-session behavior contrastive
    cebra_model.fit([X, X_s2], [y_c1, y_c1_s2])
    assert cebra_model.num_sessions == 2

    embedding = cebra_model.transform(X, session_id=0)
    assert isinstance(embedding, np.ndarray)
    embedding = cebra_model.transform(torch.Tensor(X), session_id=0)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (X.shape[0], output_dimension)
    embedding = cebra_model.transform(X_s2, session_id=1)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (X_s2.shape[0], output_dimension)

    with pytest.raises(ValueError, match="shape"):
        embedding = cebra_model.transform(X_s2, session_id=0)
    with pytest.raises(ValueError, match="shape"):
        embedding = cebra_model.transform(X, session_id=1)
    with pytest.raises(RuntimeError, match="No.*session_id"):
        embedding = cebra_model.transform(X)
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = cebra_model.transform(X, session_id=2)
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = cebra_model.transform(X, session_id=-1)

    # multi-session tensor inputs
    cebra_model.fit(
        [torch.Tensor(X), torch.Tensor(X_s2)],
        [torch.Tensor(y_c1), torch.Tensor(y_c1_s2)],
    )

    # multi-session behavior contrastive, more than two sessions
    cebra_model.fit([X, X_s2, X], [y_c1, y_c1_s2, y_c1])
    assert cebra_model.num_sessions == 3

    embedding = cebra_model.transform(X, session_id=0)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (X.shape[0], output_dimension)
    embedding = cebra_model.transform(X_s2, session_id=1)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (X_s2.shape[0], output_dimension)
    embedding = cebra_model.transform(X, session_id=2)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (X.shape[0], output_dimension)

    with pytest.raises(ValueError, match="shape"):
        embedding = cebra_model.transform(X_s2, session_id=0)
    with pytest.raises(ValueError, match="shape"):
        embedding = cebra_model.transform(X_s2, session_id=2)
    with pytest.raises(ValueError, match="shape"):
        embedding = cebra_model.transform(X, session_id=1)
    with pytest.raises(RuntimeError, match="No.*session_id"):
        embedding = cebra_model.transform(X)
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = cebra_model.transform(X, session_id=3)

    with pytest.raises(RuntimeError, match="No.*label"):
        cebra_model.fit([X, X_s2])
    with pytest.raises(ValueError, match="Invalid.*sessions"):
        cebra_model.fit([X, X_s2], y_c1)
    with pytest.raises(ValueError, match="Invalid.*sessions"):
        cebra_model.fit(X, [y_c1, y_c2])
    with pytest.raises(ValueError, match="Invalid.*sessions"):
        cebra_model.fit([X, X, X_s2], [y_c1, y_c2])
    with pytest.raises(ValueError, match="Invalid.*sessions"):
        cebra_model.fit([X, X_s2], [y_c1, y_c1, y_c2])
    with pytest.raises(NotImplementedError, match="discrete"):
        cebra_model.fit([X, X_s2], [y_d, y_d_s2])
    with pytest.raises(ValueError, match="Invalid.*samples"):
        cebra_model.fit([X, X_s2], [y_c1_s2, y_c1_s2])

    if cebra_model.pad_before_transform:
        assert embedding.shape == (len(X), 4)
    else:
        assert embedding.shape == (len(X) -
                                   len(cebra_model.model_.get_offset()) + 1, 4)

    for key in [
            "model", "optimizer", "loss", "decode", "criterion", "version",
            "log"
    ]:
        assert key in cebra_model.state_dict_, cebra_model.state_dict_.keys()


def test_adapt_model():
    adaptation_param_key = "net.O.weight", "net.0.bias"
    output_dimension = 4
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture="offset1-model",
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=5,
        max_adapt_iterations=1,
        output_dimension=output_dimension,
        batch_size=42,
        verbose=True,
    )

    X = np.random.uniform(0, 1, (1000, 50))
    X_s2 = np.random.uniform(0, 1, (800, 30))

    cebra_model.fit(X)
    before_adapt = cebra_model.state_dict_["model"]
    cebra_model._adapt_model(X_s2)
    after_adapt = cebra_model.state_dict_["model"]

    assert before_adapt.keys() == after_adapt.keys()
    for key in before_adapt.keys():
        if key in adaptation_param_key:
            assert (before_adapt[key].shape != after_adapt[key].shape
                   ) or not torch.allclose(before_adapt[key], after_adapt[key])
        else:
            assert torch.allclose(before_adapt[key], after_adapt[key])


@_util.parametrize_slow(
    arg_names="model_architecture,device",
    fast_arguments=list(
        itertools.islice(
            itertools.product(
                cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
                _DEVICES),
            2,
        )),
    slow_arguments=list(
        itertools.product(
            cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
            _DEVICES)),
)
def test_partial_fit(model_architecture, device):
    max_iterations = 10
    partial_max_iterations = 5

    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture,
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=max_iterations,
        device=device,
        output_dimension=4,
        batch_size=42,
        verbose=True,
    )

    X = np.random.uniform(0, 1, (1000, 50))
    X_s2 = np.random.uniform(0, 1, (800, 30))
    y_c1 = np.random.uniform(0, 1, (1000, 5))
    y_c1_s2 = np.random.uniform(0, 1, (800, 5))

    # Single session training
    # Fit the model
    cebra_model.fit(X)
    assert len(cebra_model.state_dict_["loss"]) == max_iterations
    for k in cebra_model.state_dict_["log"].keys():
        assert len(cebra_model.state_dict_["log"][k]) == max_iterations

    # Assert that fitting the model again resets the fitting
    cebra_model.fit(X)
    assert len(cebra_model.state_dict_["loss"]) == max_iterations
    for k in cebra_model.state_dict_["log"].keys():
        assert len(cebra_model.state_dict_["log"][k]) == max_iterations

    cebra_partial_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture,
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=partial_max_iterations,
        device=device,
        output_dimension=4,
        batch_size=42,
        verbose=True,
    )

    # Partially fit the model
    cebra_partial_model.partial_fit(X)
    assert len(
        cebra_partial_model.state_dict_["loss"]) == partial_max_iterations
    for k in cebra_partial_model.state_dict_["log"].keys():
        assert len(
            cebra_partial_model.state_dict_["log"][k]) == partial_max_iterations

    # Assert that partially fitting the model again iterates over the previously fitted model
    cebra_partial_model.partial_fit(X)
    assert len(cebra_partial_model.state_dict_["loss"]) == max_iterations
    for k in cebra_partial_model.state_dict_["log"].keys():
        assert len(cebra_partial_model.state_dict_["log"][k]) == max_iterations

    # Multi session training
    # Fit the model
    cebra_model.fit([X, X_s2], [y_c1, y_c1_s2])
    assert len(cebra_model.state_dict_["loss"]) == max_iterations
    for k in cebra_model.state_dict_["log"].keys():
        assert len(cebra_model.state_dict_["log"][k]) == max_iterations

    # Assert that fitting the model again resets the fitting
    cebra_model.fit([X, X_s2], [y_c1, y_c1_s2])
    assert len(cebra_model.state_dict_["loss"]) == max_iterations
    for k in cebra_model.state_dict_["log"].keys():
        assert len(cebra_model.state_dict_["log"][k]) == max_iterations

    cebra_partial_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture,
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=partial_max_iterations,
        device=device,
        output_dimension=4,
        batch_size=42,
        verbose=True,
    )

    # Partially fit the model
    cebra_partial_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture,
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=partial_max_iterations,
        device=device,
        output_dimension=4,
        batch_size=42,
        verbose=True,
    )

    cebra_partial_model.partial_fit([X, X_s2], [y_c1, y_c1_s2])
    assert len(
        cebra_partial_model.state_dict_["loss"]) == partial_max_iterations
    for k in cebra_partial_model.state_dict_["log"].keys():
        assert len(
            cebra_partial_model.state_dict_["log"][k]) == partial_max_iterations

    # Assert that partially fitting the model again iterates over the previously fitted model
    cebra_partial_model.partial_fit([X, X_s2], [y_c1, y_c1_s2])
    assert len(cebra_partial_model.state_dict_["loss"]) == max_iterations
    for k in cebra_partial_model.state_dict_["log"].keys():
        assert len(cebra_partial_model.state_dict_["log"][k]) == max_iterations


@_util.parametrize_slow(
    arg_names="model_architecture,device",
    fast_arguments=list(
        itertools.islice(
            itertools.product(
                cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
                _DEVICES),
            2,
        )),
    slow_arguments=list(
        itertools.product(
            cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
            _DEVICES)),
)
def test_sklearn_adapt(model_architecture, device):
    num_hidden_units = 32
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture,
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=5,
        max_adapt_iterations=1,
        device=device,
        output_dimension=4,
        num_hidden_units=num_hidden_units,
        batch_size=42,
        verbose=True,
    )

    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    X_s2 = np.random.uniform(0, 1, (800, 30))
    X_s3 = np.random.uniform(0, 1, (1000, 30))
    y_c1 = np.random.uniform(0, 1, (1000, 5))
    y_c1_s2 = np.random.uniform(0, 1, (800, 5))
    y_c2 = np.random.uniform(0, 1, (1000, 2))
    y_c2_s2 = np.random.uniform(0, 1, (800, 2))
    y_d = np.random.randint(0, 10, (1000,))
    y_d_s2 = np.random.randint(0, 10, (800,))

    def check_first_layer_dim(model, X):
        params = model.state_dict_["model"]
        params_keys = list(params.keys())
        assert params[params_keys[0]].shape[1] == X.shape[1]

    cebra_model.fit(X)
    check_first_layer_dim(cebra_model, X)

    cebra_model.fit(X_s2, adapt=True)
    check_first_layer_dim(cebra_model, X_s2)
    embedding = cebra_model.transform(X_s2)
    assert isinstance(embedding, np.ndarray)

    cebra_model.fit(X, y_c1, y_c2, adapt=True)
    check_first_layer_dim(cebra_model, X)
    embedding = cebra_model.transform(X)
    assert isinstance(embedding, np.ndarray)
    embedding = cebra_model.transform(X, session_id=0)
    assert isinstance(embedding, np.ndarray)

    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = cebra_model.transform(X, session_id=2)
    with pytest.raises(ValueError, match="Invalid.*sessions"):
        cebra_model.fit(X_s2, [y_c1_s2, y_c2_s2], adapt=True)
    with pytest.raises(ValueError, match="Invalid.*samples"):
        cebra_model.fit(X_s3, y_c1_s2, adapt=True)

    cebra_model.fit(X_s2, y_d_s2, adapt=True)
    check_first_layer_dim(cebra_model, X_s2)
    embedding = cebra_model.transform(X_s2)
    assert isinstance(embedding, np.ndarray)

    cebra_model.fit(X, y_c1, y_c2, y_d, adapt=True)
    check_first_layer_dim(cebra_model, X)
    embedding = cebra_model.transform(X)
    assert isinstance(embedding, np.ndarray)

    with pytest.raises(NotImplementedError, match=".*multisession.*"):
        cebra_model.fit([X, X_s2], [y_c1, y_c1_s2], adapt=True)


@_util.parametrize_slow(
    arg_names="model_architecture,device,pad_before_transform",
    fast_arguments=list(
        itertools.islice(
            itertools.product(
                cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
                _DEVICES,
                [True, False],
            ),
            1,
        )),
    slow_arguments=list(
        itertools.product(
            cebra_sklearn_cebra.CEBRA.supported_model_architectures(),
            _DEVICES,
            [True, False],
        )),
)
def test_sklearn_full(model_architecture, device, pad_before_transform):
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture,
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=5,
        device=device,
        pad_before_transform=pad_before_transform,
        output_dimension=4,
        batch_size=None,
        verbose=True,
    )

    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    y_c1 = np.random.uniform(0, 1, (1000, 5))
    y_c2 = np.random.uniform(0, 1, (1000, 2))
    y_d = np.random.randint(0, 10, (1000,))

    # time contrastive
    cebra_model.fit(X)
    embedding = cebra_model.transform(X)

    # continuous behavior contrastive
    cebra_model.fit(X, y_c1, y_c2)
    embedding = cebra_model.transform(X)

    # Check for https://github.com/stes/neural_cl/issues/153
    for target_type in [int, "float16", "int32", "int8", "uint8"]:
        embedding_disc_1 = cebra_model.transform(
            X.astype(target_type).astype("float32"))
        embedding_disc_2 = cebra_model.transform(X.astype(target_type))
        assert np.allclose(embedding_disc_1,
                           embedding_disc_2,
                           rtol=1e-2,
                           atol=1e-2), target_type
        assert embedding_disc_1.dtype == "float32"
        assert embedding_disc_2.dtype == "float32"

    assert isinstance(embedding, np.ndarray)
    if cebra_model.pad_before_transform:
        assert embedding.shape == (len(X), 4)
    else:
        assert embedding.shape == (len(X) -
                                   len(cebra_model.model_.get_offset()) + 1, 4)


@pytest.mark.parametrize("model_architecture,device",
                         [("resample-model", "cpu"),
                          ("resample5-model", "cpu")])
def test_sklearn_resampling_model(model_architecture, device):
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture,
        time_offsets=10,
        learning_rate=3e-4,
        max_iterations=5,
        device=device,
        output_dimension=4,
        batch_size=128,
        verbose=True,
    )

    # example dataset
    X = torch.tensor(np.random.uniform(0, 1, (1000, 50)))
    y_c1 = torch.tensor(np.random.uniform(0, 1, (1000, 5)))

    cebra_model.fit(X, y_c1)
    output = cebra_model.transform(X)
    assert output.shape == (250, 4)


@pytest.mark.parametrize("model_architecture,device",
                         [("resample1-model", "cpu")])
def test_sklearn_resampling_model_not_yet_supported(model_architecture, device):
    cebra_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture, max_iterations=5)

    # example dataset
    X = torch.tensor(np.random.uniform(0, 1, (1000, 50)))
    y_c1 = torch.tensor(np.random.uniform(0, 1, (1000, 5)))

    with pytest.raises(ValueError):
        cebra_model.fit(X, y_c1)
        output = cebra_model.transform(X)


def _iterate_actions():

    def do_nothing(model):
        return model

    def fit_model(model):
        X = np.linspace(-1, 1, 1000)[:, None]
        model.fit(X)
        return model

    return [do_nothing, fit_model]


def _assert_same_state_dict(first, second):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], torch.Tensor):
            assert torch.allclose(first[key], second[key]), key
        elif isinstance(first[key], dict):
            _assert_same_state_dict(first[key], second[key]), key
        else:
            assert first[key] == second[key]


def _assert_equal(original_model, loaded_model):
    assert original_model.get_params() == loaded_model.get_params()

    def check_fitted(model):
        """Check if a model is fitted.

        Args:
            model: The model to assess.

        Returns:
            True if fitted.
        """
        return hasattr(model, "n_features_")

    assert check_fitted(loaded_model) == check_fitted(original_model)

    if check_fitted(loaded_model):
        _assert_same_state_dict(original_model.state_dict_,
                                loaded_model.state_dict_)
        X = np.random.normal(0, 1, (100, 1))
        assert np.allclose(loaded_model.transform(X),
                           original_model.transform(X))


@pytest.mark.parametrize("action", _iterate_actions())
def test_save_and_load(action):
    model_architecture = "offset10-model"
    original_model = cebra_sklearn_cebra.CEBRA(
        model_architecture=model_architecture, max_iterations=5)
    original_model = action(original_model)
    with tempfile.NamedTemporaryFile(mode="w+b", delete=True) as savefile:
        original_model.save(savefile.name)
        loaded_model = cebra_sklearn_cebra.CEBRA.load(savefile.name)
    _assert_equal(original_model, loaded_model)
