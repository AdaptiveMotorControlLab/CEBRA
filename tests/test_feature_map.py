"""Tests for :py:mod:`cebra.attribution.feature_map`: readout, feature attribution maps and helpers."""
import numpy as np
import pytest
import torch

import cebra.attribution
import cebra.data
import cebra.models
import cebra.solver
from cebra.attribution import feature_map
from cebra.models import Model
from cebra.models.multiobjective import \
    SubspaceMultiobjectiveConvolutionalModel


@pytest.mark.filterwarnings("ignore:The readout is fitted on every row")
def test_fit_feature_readout_recovers_weights():
    rng = np.random.default_rng(0)
    T, D, F = 200, 6, 3
    embedding = rng.normal(size=(T, D))
    W_true = rng.normal(size=(D, F))
    features = embedding @ W_true  # noiseless, no intercept

    rows = np.arange(T)
    W = feature_map.fit_feature_readout(embedding,
                                        features,
                                        alpha=1e-8,
                                        train_index=rows)
    assert W.shape == (D, F)
    np.testing.assert_allclose(W, W_true, atol=1e-4)

    # 1-d features are one feature
    W_1d = feature_map.fit_feature_readout(embedding,
                                           features[:, 0],
                                           alpha=1e-8,
                                           train_index=rows)
    assert W_1d.shape == (D, 1)

    with pytest.raises(ValueError):
        feature_map.fit_feature_readout(embedding[:-1],
                                        features,
                                        train_fraction=0.5)


def test_fit_feature_readout_train_fraction_and_index():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    T, D, F = 50, 4, 2
    embedding = rng.normal(size=(T, D))
    features = rng.normal(size=(T, F))
    n_train = 45  # round(50 * 0.9)

    ref = Ridge(alpha=2.0).fit(embedding[:n_train], features[:n_train]).coef_.T
    W_frac = feature_map.fit_feature_readout(embedding,
                                             features,
                                             alpha=2.0,
                                             train_fraction=0.9)
    W_idx = feature_map.fit_feature_readout(embedding,
                                            features,
                                            alpha=2.0,
                                            train_index=np.arange(n_train))
    mask = np.arange(T) < n_train
    W_mask = feature_map.fit_feature_readout(embedding,
                                             features,
                                             alpha=2.0,
                                             train_index=mask)
    for W in (W_frac, W_idx, W_mask):
        np.testing.assert_allclose(W, ref, atol=1e-10)

    # Rows outside the training subset do not influence the fit
    garbage = embedding.copy()
    garbage[n_train:] = 1e3 * rng.normal(size=(T - n_train, D))
    W_garbage = feature_map.fit_feature_readout(garbage,
                                                features,
                                                alpha=2.0,
                                                train_fraction=0.9)
    np.testing.assert_allclose(W_garbage, ref, atol=1e-10)

    # every row selected (train_fraction=1.0, or a train_index covering all rows, also with
    # duplicates or as a full mask): fitted, with a warning that nothing is held out
    all_rows = Ridge(alpha=2.0).fit(embedding, features).coef_.T
    for kw in (dict(train_fraction=1.0), dict(train_index=np.arange(T)),
               dict(train_index=np.ones(T, dtype=bool))):
        with pytest.warns(UserWarning, match="no row is held out"):
            W_all = feature_map.fit_feature_readout(embedding,
                                                    features,
                                                    alpha=2.0,
                                                    **kw)
        np.testing.assert_allclose(W_all, all_rows, atol=1e-10)


def test_fit_feature_readout_default_first_80_percent(capsys):
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(3)
    embedding = rng.normal(size=(50, 4))
    features = rng.normal(size=(50, 2))
    W = feature_map.fit_feature_readout(embedding, features, alpha=2.0)
    ref = Ridge(alpha=2.0).fit(embedding[:40], features[:40]).coef_.T
    np.testing.assert_allclose(W, ref, atol=1e-10)
    assert "first 80% of rows" in capsys.readouterr().out

    # an explicit choice prints nothing
    feature_map.fit_feature_readout(embedding, features, train_fraction=0.5)
    feature_map.fit_feature_readout(embedding,
                                    features,
                                    train_index=np.arange(10))
    assert capsys.readouterr().out == ""


def test_fit_feature_readout_subset_errors():
    rng = np.random.default_rng(2)
    embedding = rng.normal(size=(20, 3))
    features = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match="not both"):
        feature_map.fit_feature_readout(embedding,
                                        features,
                                        train_fraction=0.5,
                                        train_index=np.arange(5))
    with pytest.raises(ValueError, match="empty"):
        feature_map.fit_feature_readout(embedding,
                                        features,
                                        train_index=np.array([], dtype=int))
    with pytest.raises(ValueError, match="empty"):
        feature_map.fit_feature_readout(embedding,
                                        features,
                                        train_fraction=0.01)
    with pytest.raises(ValueError):
        feature_map.fit_feature_readout(embedding,
                                        features,
                                        train_index=np.zeros(7, dtype=bool))
    with pytest.raises(ValueError):
        feature_map.fit_feature_readout(embedding, features, train_fraction=1.5)


class MLPEncoder(Model):
    """Per-sample encoder without offset."""

    def __init__(self, num_input, num_output):
        super().__init__(num_input=num_input, num_output=num_output)
        self.net = torch.nn.Sequential(torch.nn.Linear(num_input, 8),
                                       torch.nn.Tanh(),
                                       torch.nn.Linear(8, num_output))

    def forward(self, x):
        return self.net(x)

    def get_offset(self):
        return None


def _regularized(model):
    """Single-session solver trained with the Jacobian regularizer (not fitted here)."""
    return cebra.solver.RegularizedSolver(model=model,
                                          criterion=cebra.models.InfoNCE(),
                                          optimizer=torch.optim.Adam(
                                              model.parameters()))


def _two_session_dataset(n_time=40, n_neurons=(3, 5)):
    # Session 0: labels cycle through 0..4 (repeated trials). Session 1: same labels, shuffled
    # order, so the nearest-label match of every reference sample has several ties.
    rng = np.random.default_rng(0)
    labels = [torch.tensor(np.arange(n_time) % 5, dtype=torch.float32)[:, None]]
    labels.append(
        torch.tensor(rng.permutation(n_time) % 5, dtype=torch.float32)[:, None])
    assert all(int((labels[1] == v).sum()) > 1 for v in range(5))  # ties exist
    inputs = [
        torch.arange(n_time * n, dtype=torch.float32).reshape(n_time, n)
        for n in n_neurons
    ]
    datasets = [
        cebra.data.TensorDataset(inputs[i],
                                 continuous=labels[i],
                                 offset=cebra.data.Offset(0, 1))
        for i in range(2)
    ]
    return cebra.data.UnifiedDataset(*datasets), inputs, labels


class ConvTanhEncoder(cebra.models.Model, cebra.models.ConvolutionalModelMixin):
    """Convolutional-offset encoder with a nonlinearity, so the Jacobian varies over time."""

    def __init__(self, num_input, num_output, left=2, right=1):
        super().__init__(num_input=num_input, num_output=num_output)
        self.offset = cebra.data.Offset(left, right)
        self.linear = torch.nn.Linear(num_input * (left + right), num_output)

    def forward(self, x):  # x: (batch, N, L)
        return torch.tanh(self.linear(0.01 * x.flatten(1)))

    def get_offset(self):
        return self.offset


def test_num_valid_rows():
    """One output per window of left + right samples; per-sample models give one per sample."""
    model = ConvTanhEncoder(4, 3, left=2, right=1)  # window of 3 samples
    assert feature_map._num_valid_rows(model, 20) == 18  # outputs at samples 2..19
    assert feature_map._num_valid_rows(MLPEncoder(4, 3), 20) == 20


def _reference_row_jacobian(model, neural, row):
    """Autograd Jacobian of the embedding of one valid row, averaged over the window (D, N)."""
    left, right = model.get_offset().left, model.get_offset().right
    window = (neural[row:row + left + right].T.detach().clone()
             )  # (N, L), model input layout
    jac = torch.autograd.functional.jacobian(lambda w: model(w[None])[0],
                                             window)
    return jac.mean(-1).detach().double().numpy()  # (D, N)


def test_per_row_maps_follow_model_window():
    # Row r is sample left + r, its window is neural[r : r + left + right]
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    model = ConvTanhEncoder(num_input=4, num_output=5, left=2, right=1)
    solver = _regularized(model)
    neural = torch.randn(20, 4)
    n_valid = 20 - 2 - 1 + 1
    W = rng.normal(size=(5, 2))

    per_row = feature_map.feature_attribution_map(solver,
                                                  neural,
                                                  W,
                                                  average=False,
                                                  batch_size=7)
    assert per_row.shape == (n_valid, 4, 2)
    for row in range(n_valid):
        expected = _reference_row_jacobian(model, neural, row).T @ W
        np.testing.assert_allclose(per_row[row], expected, atol=1e-6)


def test_per_row_batch_size_invariance_and_mean():
    torch.manual_seed(0)
    rng = np.random.default_rng(4)
    model = ConvTanhEncoder(num_input=4, num_output=5, left=2, right=1)
    solver = _regularized(model)
    neural = torch.randn(40, 4)
    W = rng.normal(size=(5, 2))

    rows = {
        b:
            feature_map.feature_attribution_map(solver,
                                                neural,
                                                W,
                                                average=False,
                                                batch_size=b) for b in (1, 7, 64)
    }
    assert rows[1].shape == (38, 4, 2)
    np.testing.assert_allclose(rows[1], rows[7], atol=1e-6)
    np.testing.assert_allclose(rows[1], rows[64], atol=1e-6)

    mean = feature_map.feature_attribution_map(solver, neural, W, batch_size=7)
    assert mean.shape == (4, 2)
    np.testing.assert_allclose(mean, rows[7].mean(axis=0), atol=1e-6)


def test_single_session_output_slice_and_split_outputs():
    torch.manual_seed(2)
    rng = np.random.default_rng(6)
    module = ConvTanhEncoder(num_input=4, num_output=6, left=2, right=1)
    model = SubspaceMultiobjectiveConvolutionalModel(
        module,
        feature_ranges=[slice(0, 3), slice(3, 6)],
        renormalize=True,
        split_outputs=True,
    )
    solver = _regularized(model)
    neural = torch.randn(30, 4)
    n_valid = 30 - 2 - 1 + 1
    sl = slice(3, 6)
    W_slice = rng.normal(size=(3, 3))  # readout on the embedding columns 3:6

    # Reference: Jacobian of the wrapper's renormalised embedding (split_outputs off), rows 3:6
    model.split_outputs = False
    expected = np.stack([
        _reference_row_jacobian(model, neural, row)[sl].T @ W_slice
        for row in range(n_valid)
    ])
    model.split_outputs = True

    per_row = feature_map.feature_attribution_map(solver,
                                                  neural,
                                                  W_slice,
                                                  output_slice=sl,
                                                  average=False,
                                                  batch_size=5)
    np.testing.assert_allclose(per_row, expected, atol=1e-6)
    mean = feature_map.feature_attribution_map(solver,
                                               neural,
                                               W_slice,
                                               output_slice=sl)
    np.testing.assert_allclose(mean, expected.mean(axis=0), atol=1e-6)
    assert model.split_outputs is True


def test_single_session_readout_shape_check():
    rng = np.random.default_rng(7)
    module = ConvTanhEncoder(num_input=4, num_output=6, left=2, right=1)
    model = SubspaceMultiobjectiveConvolutionalModel(
        module,
        feature_ranges=[slice(0, 3), slice(3, 6)],
        renormalize=True,
        split_outputs=True,
    )
    solver = _regularized(model)
    neural = torch.randn(30, 4)

    # without output_slice, W must have all 6 output rows
    feature_map.feature_attribution_map(solver, neural, rng.normal(size=(6, 2)))
    with pytest.raises(ValueError, match="rows"):
        feature_map.feature_attribution_map(solver, neural,
                                            rng.normal(size=(3, 2)))
    # with output_slice, W must have len(output_slice) rows
    with pytest.raises(ValueError, match="rows"):
        feature_map.feature_attribution_map(solver,
                                            neural,
                                            rng.normal(size=(6, 2)),
                                            output_slice=slice(3, 6))
    with pytest.raises(ValueError):
        feature_map.feature_attribution_map(solver, neural, np.ones(6))
    assert model.split_outputs is True


def _unified_solver(n_neurons=(3, 5), num_output=4):
    model = ConvTanhEncoder(num_input=sum(n_neurons),
                            num_output=num_output,
                            left=2,
                            right=1)
    return cebra.solver.UnifiedSolverxCEBRA(
        model=model,
        criterion=cebra.models.InfoNCE(),
        optimizer=torch.optim.Adam(model.parameters()),
        feature_ranges=[slice(0, num_output)],
        regularizer=cebra.models.jacobian_regularizer.JacobianReg(),
        renormalize=True)


def test_dispatch_unified_matches_aligned_inputs():
    torch.manual_seed(3)
    _, inputs, labels = _two_session_dataset(n_time=40, n_neurons=(3, 5))
    solver = _unified_solver()
    W = np.random.default_rng(7).normal(size=(4, 2))
    kwargs = dict(labels=labels,
                  session_id=0,
                  random_ties=True,
                  seed=1,
                  batch_size=9)

    mean = feature_map.feature_attribution_map(solver, inputs, W, **kwargs)
    rows = feature_map.feature_attribution_map(solver,
                                               inputs,
                                               W,
                                               average=False,
                                               **kwargs)
    assert mean.shape == (8, 2)
    assert rows.shape == (38, 8, 2)  # 40 - left - right + 1
    # the mean of the per-row maps == the averaged map == the map of the model on the aligned inputs
    np.testing.assert_allclose(rows.mean(axis=0), mean, atol=1e-6)
    aligned = solver.aligned_inputs(inputs, labels, 0, random_ties=True, seed=1)
    ref = feature_map.feature_attribution_map(_regularized(solver.model),
                                              aligned,
                                              W,
                                              batch_size=9)
    np.testing.assert_allclose(mean, ref, atol=1e-6)

    # circular reference session: one row per reference sample
    offset = solver.model.get_offset()
    indices = torch.arange(-offset.left,
                           40 + offset.right - 1) % 40  # window wraps around
    wrapped = feature_map.feature_attribution_map(solver,
                                                  inputs,
                                                  W,
                                                  indices=indices,
                                                  average=False,
                                                  **{
                                                      **kwargs, "batch_size": 64
                                                  })
    assert wrapped.shape == (40, 8, 2)

    with pytest.raises(ValueError, match="labels"):
        feature_map.feature_attribution_map(solver, inputs, W)
    with pytest.raises(ValueError, match="session_id"):
        feature_map.feature_attribution_map(solver, inputs, W, labels=labels)
    with pytest.raises(ValueError, match="rows"):
        feature_map.feature_attribution_map(solver,
                                            inputs,
                                            np.ones((5, 2)),
                                            labels=labels,
                                            session_id=0)


def test_dispatch_single_session_solver():
    torch.manual_seed(4)
    model = ConvTanhEncoder(num_input=4, num_output=5, left=2, right=1)
    solver = _regularized(model)
    neural = torch.randn(20, 4)
    W = np.random.default_rng(0).normal(size=(5, 2))

    mean = feature_map.feature_attribution_map(solver, neural.numpy(), W)
    assert mean.shape == (4, 2)

    # input samples can be selected with indices
    sub = feature_map.feature_attribution_map(solver,
                                              neural,
                                              W,
                                              indices=np.arange(10),
                                              average=False)
    assert sub.shape == (8, 4, 2)

    with pytest.raises(ValueError, match="unified"):
        feature_map.feature_attribution_map(solver, neural, W, labels=[neural])
    with pytest.raises(ValueError, match="list"):
        feature_map.feature_attribution_map(solver, [neural], W)


def test_unregularized_models_are_rejected():
    model = ConvTanhEncoder(num_input=4, num_output=5, left=2, right=1)
    neural = torch.randn(20, 4)
    W = np.ones((5, 2))
    unregularized = [
        model,  # plain module
        torch.nn.Linear(4, 5),
        cebra.solver.SingleSessionSolver(model=model,
                                         criterion=cebra.models.InfoNCE(),
                                         optimizer=torch.optim.Adam(
                                             model.parameters())),
        cebra.solver.UnifiedSolver(model=model,
                                   criterion=cebra.models.InfoNCE(),
                                   optimizer=torch.optim.Adam(
                                       model.parameters())),
    ]
    for obj in unregularized:
        with pytest.raises(TypeError, match="Jacobian regularizer"):
            feature_map.feature_attribution_map(obj, neural, W)
    with pytest.raises(TypeError, match="sklearn"):
        feature_map.feature_attribution_map(cebra.CEBRA(), neural, W)


def test_jacobian_in_eval_mode_and_mode_restored():
    torch.manual_seed(0)

    class DropoutEncoder(Model):

        def __init__(self):
            super().__init__(num_input=4, num_output=3)
            self.net = torch.nn.Sequential(torch.nn.Linear(4, 32),
                                           torch.nn.Dropout(0.5),
                                           torch.nn.Linear(32, 3))
            self.frozen = torch.nn.Dropout(
                0.1)  # a submodule already in eval mode

        def forward(self, x):
            return self.net(x)

        def get_offset(self):
            return None

    model = DropoutEncoder()
    solver = _regularized(model)
    neural = torch.randn(10, 4)
    W = np.random.default_rng(0).normal(size=(3, 2))

    model.eval()
    expected = feature_map.feature_attribution_map(solver, neural, W)
    model.train()
    model.frozen.eval()
    got = feature_map.feature_attribution_map(solver, neural, W)
    np.testing.assert_allclose(got, expected)
    assert model.training and model.net[1].training and not model.frozen.training

    # in train mode the dropout masks would change the outputs (and so the map)
    model.train()
    assert not torch.allclose(model(neural), model(neural))


def test_short_input_raises():
    model = ConvTanhEncoder(num_input=4, num_output=5, left=2, right=1)
    W = np.ones((5, 2))
    with pytest.raises(ValueError, match="too few"):
        feature_map.feature_attribution_map(_regularized(model),
                                            torch.randn(2, 4), W)
    _, inputs, labels = _two_session_dataset(n_time=40, n_neurons=(3, 5))
    solver = _unified_solver()
    with pytest.raises(ValueError, match="too few"):
        feature_map.feature_attribution_map(solver,
                                            inputs,
                                            np.ones((4, 2)),
                                            labels=labels,
                                            session_id=0,
                                            indices=[0, 1])


def test_unified_random_ties_seeded_and_global_rng_untouched():
    _, inputs, labels = _two_session_dataset(n_time=40, n_neurons=(3, 5))
    solver = _unified_solver()
    W = np.random.default_rng(2).normal(size=(4, 2))

    def run(**kw):
        return feature_map.feature_attribution_map(solver,
                                                   inputs,
                                                   W,
                                                   labels=labels,
                                                   session_id=0,
                                                   **kw)

    torch_state, numpy_state = torch.random.get_rng_state(), np.random.get_state()[1].copy()
    a = run(random_ties=True, seed=3)
    np.testing.assert_allclose(a, run(random_ties=True, seed=3))
    assert not np.allclose(a, run(random_ties=True, seed=4))
    assert torch.equal(torch_state, torch.random.get_rng_state())
    np.testing.assert_array_equal(numpy_state, np.random.get_state()[1])
    np.testing.assert_allclose(run(), run(random_ties=False))


def test_public_api_exports():
    assert set(feature_map.__all__) == {
        "fit_feature_readout", "feature_attribution_map", "average_feature_maps"
    }
    for name in feature_map.__all__:
        func = getattr(feature_map, name)
        assert getattr(cebra.attribution, name) is func
        assert ">>>" in func.__doc__, f"{name} has no example (run by --doctest-modules)"


@pytest.mark.parametrize("conv", [False, True])
def test_feature_map_methods_agree_with_cebra_attribution(conv):
    # M_t = J_t^T W with J_t from CEBRA's "jacobian-based" method (and, for a model without an
    # offset, the Captum backend "neuron-gradient"); every `method` option gives the same map.
    torch.manual_seed(0)
    rng = np.random.default_rng(8)
    model = ConvTanhEncoder(4, 5, left=2, right=1) if conv else MLPEncoder(4, 5)
    solver = _regularized(model)
    neural = torch.randn(30, 4)
    W = rng.normal(size=(5, 2))

    jacobian = cebra.attribution.init(
        "jacobian-based",
        model=model,
        input_data=neural.clone().requires_grad_(True),
        output_dimension=5).compute_attribution_map(selected_keys=["jf"])["jf"]
    expected = np.einsum("tdn,df->tnf", jacobian, W)  # J_t^T W, (T_valid, N, F)
    if not conv:
        gradient = cebra.attribution.init(
            "neuron-gradient",
            model=model,
            input_data=neural.clone().requires_grad_(True),
            output_dimension=5).compute_attribution_map()["neuron-gradient"]
        np.testing.assert_allclose(np.einsum("tdn,df->tnf", gradient, W),
                                   expected,
                                   rtol=1e-5,
                                   atol=1e-6)

    for method in feature_map.JACOBIAN_METHODS:
        per_row = feature_map.feature_attribution_map(solver,
                                                      neural,
                                                      W,
                                                      average=False,
                                                      batch_size=7,
                                                      method=method)
        np.testing.assert_allclose(per_row, expected, rtol=1e-5, atol=1e-6)
        mean = feature_map.feature_attribution_map(solver, neural, W, method=method)
        np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="method must be one of"):
        feature_map.feature_attribution_map(solver,
                                            neural,
                                            W,
                                            method="neuron-gradient")
