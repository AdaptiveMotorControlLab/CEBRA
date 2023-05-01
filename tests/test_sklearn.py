import itertools
import tempfile
import warnings

import numpy as np
import pytest
import torch

import cebra.models

if torch.cuda.is_available():
else:


def test_sklearn_dataset():
    X = np.zeros((100, 5), dtype="float32")
    yc = np.zeros((100, 5), dtype="float32")
    yd = np.zeros((100,), dtype="int")

    # cannot create datasets with more than one 1D discrete index
    with pytest.raises(ValueError):
    with pytest.raises(ValueError):

    # need to pass iterable as type ...
    with pytest.raises(TypeError):
    with pytest.raises(TypeError):
    with pytest.raises(TypeError):
    with pytest.raises(TypeError):
    with pytest.raises(TypeError):
    with pytest.raises(ValueError):

    # ... but any iterable needs to work
    arg = ()
    arg = (yd,)
    arg = (yc,)
    arg = (yc, yc, yd)

    # checking data input and indexing ops
    for labels in [(), (yc,), (yd,), (yc, yd)]:
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
        assert (data.continuous_index is not None) == is_cont
        assert (data.discrete_index is not None) == is_disc



    shared_kwargs = dict(num_steps=5, dataset=__Dataset())
    extra_kwargs = dict(batch_size=512, time_offsets=10, delta=0.01)

    try:
            is_cont=is_cont,
            is_disc=is_disc,
            is_full=is_full,
            is_multi=is_multi,
            is_hybrid=is_hybrid,
            shared_kwargs=shared_kwargs,
        assert isinstance(loader, cebra.data.Loader)
        assert isinstance(solver, str)
    except Exception as e:
        with pytest.raises((NotImplementedError, ValueError)):
            raise e

def iterate_models():
    # architecture checks
    for model_architecture, device, distance in itertools.product(
            model_architecture=model_architecture,
            pad_before_transform=
            True,  # NOTE(stes) needs to be true, otherwise not sklearn compatible.
            device=device,
            distance=distance,
            time_offsets=5,
            max_iterations=5,

    # parameter checks
        [
            "offset10-model", "offset10-model-mse", "offset1-model",
            "resample-model"
            model_architecture=model_architecture,
            pad_before_transform=
            True,  # NOTE(stes) needs to be true, otherwise not sklearn compatible.
            device=device,
            distance=distance,
            min_temperature=min_temperature,
            temperature_mode=temperature_mode,
            time_offsets=5,
            max_iterations=5,

    num_retries = 1
        pytest.skip("CEBRA is non-deterministic.")
        num_retries = 1000
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


    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    y_c1 = np.random.uniform(0, 1, (1000, 5))
    y_c2 = np.random.uniform(0, 1, (1000, 2))
    y_d = np.random.randint(0, 10, (1000,))

    # time contrastive

    # continuous behavior contrastive

    # discrete behavior contrastive

    # mixed

    assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (len(X), 4)
    else:

    for key in [
    ]:


    # example dataset
    X = np.random.uniform(0, 1, (1000, 50))
    y_c1 = np.random.uniform(0, 1, (1000, 5))
    y_c2 = np.random.uniform(0, 1, (1000, 2))
    y_d = np.random.randint(0, 10, (1000,))

    # time contrastive

    # continuous behavior contrastive

    # Check for https://github.com/stes/neural_cl/issues/153
    for target_type in [int, "float16", "int32", "int8", "uint8"]:
            X.astype(target_type).astype("float32"))
        assert np.allclose(embedding_disc_1,
                           embedding_disc_2,
                           rtol=1e-2,
                           atol=1e-2), target_type
        assert embedding_disc_1.dtype == "float32"
        assert embedding_disc_2.dtype == "float32"

    assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (len(X), 4)
    else:
@pytest.mark.parametrize("model_architecture,device",
                         [("resample-model", "cpu"),
                          ("resample5-model", "cpu")])

@pytest.mark.parametrize("model_architecture,device",
                         [("resample1-model", "cpu")])
def test_sklearn_resampling_model_not_yet_supported(model_architecture, device):

    # example dataset
    X = torch.tensor(np.random.uniform(0, 1, (1000, 50)))
    y_c1 = torch.tensor(np.random.uniform(0, 1, (1000, 5)))

    with pytest.raises(ValueError):
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
        _assert_same_state_dict(original_model.state_dict_,
                                loaded_model.state_dict_)
        X = np.random.normal(0, 1, (100, 1))
        assert np.allclose(loaded_model.transform(X),
                           original_model.transform(X))


@pytest.mark.parametrize("action", _iterate_actions())
def test_save_and_load(action):
    model_architecture = "offset10-model"
    original_model = action(original_model)
    with tempfile.NamedTemporaryFile(mode="w+b", delete=True) as savefile:
        original_model.save(savefile.name)
