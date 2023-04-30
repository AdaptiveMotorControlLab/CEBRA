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
import pytest
import torch

import cebra.data
import cebra.io


def parametrize_device(func):
    _devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
    return pytest.mark.parametrize("device", _devices)(func)


class LoadSpeed:

    def __init__(self, loader):
        self.loader = loader

    def __call__(self):
        n = 0
        for batch in self.loader:
            n += 1
        assert batch.reference.device.type == self.loader.device
        assert n == len(self.loader)


class RandomDataset(cebra.data.SingleSessionDataset):

    def __init__(self, N=100, d=5, device="cpu"):
        super().__init__(device=device)
        self._cindex = torch.randint(0, 5, (N, d), device=device).float()
        self._dindex = torch.randint(0, 5, (N,), device=device).long()
        self.neural = self._data = torch.randn((N, d), device=device)

    @property
    def input_dimension(self):
        return self._data.shape[1]

    def __len__(self):
        return len(self._data)

    @property
    def continuous_index(self):
        return self._cindex

    @property
    def discrete_index(self):
        return self._dindex

    def __getitem__(self, index):
        return self._data[index]


def test_offset():
    offset = cebra.data.Offset(5, 4)
    assert offset.left == 5
    assert offset.right == 4
    assert offset.left_slice == slice(0, 5)
    assert len(offset) == 5 + 4

    offset = cebra.data.Offset(0, 4)
    assert offset.left == 0
    assert offset.right == 4
    assert offset.left_slice == slice(0, 0)
    assert len(offset) == 4

    offset = cebra.data.Offset(5)
    assert offset.left == 5
    assert offset.right == 5
    assert offset.left_slice == slice(0, 5)
    assert offset.right_slice == slice(-5, None)
    assert len(offset) == 5 * 2

    with pytest.raises(ValueError, match="Invalid.*right"):
        offset = cebra.data.Offset(5, 0)
    with pytest.raises(ValueError, match="Invalid.*right"):
        offset = cebra.data.Offset(0, 0)
    with pytest.raises(ValueError, match="Invalid.*number"):
        offset = cebra.data.Offset(5, 5, 5)
    with pytest.raises(ValueError, match="Invalid.*bounds"):
        offset = cebra.data.Offset(-2, 4)
    with pytest.raises(ValueError, match="Invalid.*bounds"):
        offset = cebra.data.Offset(4, -2)


def _assert_dataset_on_correct_device(loader, device):
    assert hasattr(loader, "dataset")
    assert hasattr(loader, "device")
    assert isinstance(loader.dataset, cebra.io.HasDevice)
    assert loader.dataset.neural.device.type == device


def test_demo_data():
    if not torch.cuda.is_available():
        pytest.skip("Test only possible with CUDA.")

    dataset = RandomDataset(N=100, device="cuda")
    assert dataset.neural.device.type == "cuda"
    dataset.to("cpu")
    assert dataset.neural.device.type == "cpu"


def _assert_device(first, second):

    def _to_str(val):
        if isinstance(val, torch.device):
            return val.type
        return val

    assert _to_str(first) == _to_str(second)


@parametrize_device
@pytest.mark.parametrize(
    "data_name, loader_initfunc",
    [
        ("demo-discrete", cebra.data.DiscreteDataLoader),
        ("demo-continuous", cebra.data.ContinuousDataLoader),
        ("demo-mixed", cebra.data.MixedDataLoader),
    ],
)
def test_device(data_name, loader_initfunc, device):
    if not torch.cuda.is_available():
        pytest.skip("Test only possible with CUDA.")

    swap = {"cpu": "cuda", "cuda": "cpu"}
    other_device = swap.get(device)
    dataset = RandomDataset(N=100, device=other_device)

    loader = loader_initfunc(dataset, num_steps=10, batch_size=32)
    loader.to(device)
    assert loader.dataset == dataset
    _assert_device(loader.device, device)
    _assert_device(loader.dataset.device, device)

    _assert_device(loader.get_indices(10).reference.device, device)


@parametrize_device
@pytest.mark.parametrize("prior", ("uniform", "empirical"))
def test_discrete(prior, device, benchmark):
    dataset = RandomDataset(N=100, device=device)
    loader = cebra.data.DiscreteDataLoader(
        dataset=dataset,
        num_steps=10,
        batch_size=8,
        prior=prior,
    )
    _assert_dataset_on_correct_device(loader, device)
    load_speed = LoadSpeed(loader)
    benchmark(load_speed)


@parametrize_device
@pytest.mark.parametrize("conditional", ("time", "time_delta"))
def test_continuous(conditional, device, benchmark):
    dataset = RandomDataset(N=100, d=5, device=device)
    loader = cebra.data.ContinuousDataLoader(
        dataset=dataset,
        num_steps=10,
        batch_size=8,
        conditional=conditional,
    )
    _assert_dataset_on_correct_device(loader, device)
    load_speed = LoadSpeed(loader)
    benchmark(load_speed)


def _check_attributes(obj, is_list=False):
    if is_list:
        for obj_ in obj:
            _check_attributes(obj_, is_list=False)
    elif isinstance(obj, cebra.data.Batch) or isinstance(
            obj, cebra.data.BatchIndex):
        assert hasattr(obj, "positive")
        assert hasattr(obj, "negative")
        assert hasattr(obj, "reference")
    else:
        raise TypeError()


@parametrize_device
@pytest.mark.parametrize(
    "data_name, loader_initfunc",
    [
        ("demo-discrete", cebra.data.DiscreteDataLoader),
        ("demo-continuous", cebra.data.ContinuousDataLoader),
        ("demo-mixed", cebra.data.MixedDataLoader),
    ],
)
def test_singlesession_loader(data_name, loader_initfunc, device):
    data = cebra.datasets.init(data_name)
    data.to(device)
    loader = loader_initfunc(data, num_steps=10, batch_size=32)
    _assert_dataset_on_correct_device(loader, device)

    index = loader.get_indices(100)
    _check_attributes(index)

    for batch in loader:
        _check_attributes(batch)
        assert len(batch.positive) == 32


def test_multisession_cont_loader():
    data = cebra.datasets.MultiContinuous(nums_neural=[3, 4, 5],
                                          num_behavior=5,
                                          num_timepoints=100)
    loader = cebra.data.ContinuousMultiSessionDataLoader(
        data,
        num_steps=10,
        batch_size=32,
    )

    # Check the sampler
    assert hasattr(loader, "sampler")
    ref_idx = loader.sampler.sample_prior(1000)
    assert len(ref_idx) == 3  # num_sessions
    for session in range(3):
        assert ref_idx[session].max() < 100
    pos_idx, idx, idx_rev = loader.sampler.sample_conditional(ref_idx)

    assert pos_idx is not None
    assert idx is not None
    assert idx_rev is not None

    batch = next(iter(loader))

    def _mix(array, idx):
        shape = array.shape
        n, m = shape[:2]
        mixed = array.reshape(n * m, -1)[idx]
        print(mixed.shape, array.shape, idx.shape)
        return mixed.reshape(shape)

    def _process(batch, feature_dim=1):
        """Given list_i[(N,d_i)] batch, return (#session, N, feature_dim) tensor"""
        return torch.stack(
            [b.reference.flatten(1).mean(dim=1, keepdims=True) for b in batch],
            dim=0).repeat(1, 1, feature_dim)

    assert batch[0].reference.shape == (32, 3, 10)
    assert batch[1].reference.shape == (32, 4, 10)
    assert batch[2].reference.shape == (32, 5, 10)

    dummy_prediction = _process(batch, feature_dim=6)
    assert dummy_prediction.shape == (3, 32, 6)
    _mix(dummy_prediction, batch[0].index)


@parametrize_device
@pytest.mark.parametrize(
    "data_name, loader_initfunc",
    [
        # ('demo-discrete-multisession', cebra.data.DiscreteMultiSessionDataLoader),
        ("demo-continuous-multisession",
         cebra.data.ContinuousMultiSessionDataLoader)
    ],
)
def test_multisession_loader(data_name, loader_initfunc, device):
    # TODO change number of timepoints across the sessions

    data = cebra.datasets.init(data_name)
    kwargs = dict(num_steps=10, batch_size=32)
    loader = loader_initfunc(data, **kwargs)

    index = loader.get_indices(100)
    print(index[0])
    print(type(index))
    _check_attributes(index, is_list=False)

    for batch in loader:
        _check_attributes(batch, is_list=True)
        for session_batch in batch:
            assert len(session_batch.positive) == 32
