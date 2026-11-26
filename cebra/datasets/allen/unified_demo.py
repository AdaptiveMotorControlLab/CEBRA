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
"""Data helpers for the Unified CEBRA attribution demo on the Allen dataset.

Loads the metadata cache and the event trains, and builds the per-session inputs of the unified model.
Numpy only at import time; ``cebra.data`` and ``scipy`` are imported lazily by the functions that need them.

Cache file: ``allen_demo_cache.npz``, archived on figshare (see :py:data:`CACHE_URL`). Contents:
  (no events: they are read from the processed Allen .mat files, see events_from_visual_drift)
  neuron_* (N,)               session index (model order), session id, cell class, area, mouse, depth.
  session_* (234,)            per session; session_n_neurons sums to N.
  frame_features (900, 768)   DINO ViT-B features, the alignment labels of the unified model.
  movie_features (900, 6)     z-scored movie features (readout target); names in movie_feature_names.
  meta_json                   provenance and model hyper-parameters (see AllenDemo.meta).

Typical use in a notebook::

    from cebra.datasets.allen.unified_demo import load_allen_demo, session_tensor_datasets, TRAIN_REPEATS
    demo = load_allen_demo(visual_drift_dir)       # events from the processed Allen .mat files
    datasets = session_tensor_datasets(demo, TRAIN_REPEATS, device=device)
    dataset = cebra.data.UnifiedxCEBRADataset(*datasets)   # the trainer's class, 234 sessions
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import urllib.request
import zipfile
from typing import List, Sequence, Tuple

import numpy as np

from cebra.datasets import get_datapath

# figshare article holding allen_demo_cache.npz and the reference checkpoints:
# https://figshare.com/s/ae8099185f5e2f5e8185 (downloaded as one zip archive)
CACHE_URL = "https://ndownloader.figshare.com/articles/29275358?private_link=ae8099185f5e2f5e8185"
DEFAULT_CACHE = pathlib.Path(get_datapath("allen/allen_demo_cache.npz"))
NEURON_TYPE_DIR = {"Exc": "calcium_excitatory", "Vip": "calcium_inhibitory", "Pvalb": "calcium_inhibitory",
                   "Sst": "calcium_inhibitory"}
TRAIN_REPEATS = tuple(range(9))   # repeats the unified model was trained on
VALID_REPEAT = 9                  # held-out repeat, used for every score


@dataclasses.dataclass
class AllenDemo:
    events: np.ndarray                 # (repeats, frames, N) uint8; 10 repeats, 0-8 train, 9 held out
    neuron_session_index: np.ndarray   # (N,) model-order session index
    neuron_session_id: np.ndarray
    neuron_cell_class: np.ndarray
    neuron_area: np.ndarray
    neuron_mouse: np.ndarray
    neuron_depth_um: np.ndarray
    neuron_mat_column: np.ndarray
    session_id: np.ndarray
    session_area: np.ndarray
    session_cell_class: np.ndarray
    session_n_neurons: np.ndarray
    session_neuron_start: np.ndarray
    session_mouse: np.ndarray
    session_depth_um: np.ndarray
    frame_features: np.ndarray         # (frames, 768) float32
    movie_features: np.ndarray         # (frames, 6) float64, z-scored
    movie_feature_names: List[str]
    meta: dict

    @property
    def n_neurons(self) -> int:
        return int(self.events.shape[2])

    @property
    def n_sessions(self) -> int:
        return int(len(self.session_n_neurons))

    @property
    def n_frames(self) -> int:
        return int(self.events.shape[1])


_ARRAY_FIELDS = [f.name for f in dataclasses.fields(AllenDemo)
                 if f.name not in ("movie_feature_names", "meta")]


def load_allen_demo(visual_drift_dir=None, path=DEFAULT_CACHE) -> AllenDemo:
    """Read the metadata cache, add the events, and check that the parts agree with each other.

    visual_drift_dir: folder of the processed Allen data (the ``data`` folder with ``calcium_excitatory/`` and
    ``calcium_inhibitory/``; the notebook downloads it from figshare). The events are read from its .mat files
    (events_from_visual_drift). It may be None only if the cache file itself has an ``events`` array.
    """
    with np.load(pathlib.Path(path), allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    if visual_drift_dir is not None:
        raw["events"] = events_from_visual_drift(raw, visual_drift_dir)
    elif "events" not in raw:
        raise ValueError(f"{path} holds no events: pass visual_drift_dir, the folder with calcium_excitatory/ "
                         "and calcium_inhibitory/ of the processed Allen data")
    missing = [k for k in _ARRAY_FIELDS + ["movie_feature_names", "meta_json"] if k not in raw]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    demo = AllenDemo(
        **{k: raw[k] for k in _ARRAY_FIELDS},
        movie_feature_names=[str(x) for x in raw["movie_feature_names"]],
        meta=json.loads(str(raw["meta_json"])),
    )
    n = demo.n_neurons
    if int(demo.session_n_neurons.sum()) != n:
        raise ValueError("session_n_neurons does not sum to the number of neurons")
    if demo.events.ndim != 3 or demo.events.dtype != np.uint8:
        raise ValueError("events must be (repeats, frames, neurons) uint8")
    if demo.frame_features.shape[0] != demo.n_frames or demo.movie_features.shape[0] != demo.n_frames:
        raise ValueError("frame_features / movie_features must have one row per frame")
    if len(demo.movie_feature_names) != demo.movie_features.shape[1]:
        raise ValueError("movie_feature_names does not match movie_features")
    return demo


def split_sessions(neural: np.ndarray, n_neurons_per_session: Sequence[int]) -> List[np.ndarray]:
    """Split the last (neuron) axis into per-session arrays, in model order.

    neural: (T, N) or (R, T, N). Returns views of shape (T, n_i) or (R, T, n_i).
    """
    counts = [int(n) for n in n_neurons_per_session]
    if sum(counts) != neural.shape[-1]:
        raise ValueError(f"sessions have {sum(counts)} neurons, array has {neural.shape[-1]}")
    out, start = [], 0
    for n in counts:
        out.append(neural[..., start:start + n])
        start += n
    return out


def repeat_frames(demo: AllenDemo, repeats: Sequence[int] = TRAIN_REPEATS) -> Tuple[np.ndarray, np.ndarray]:
    """Stack the chosen repeats in time.

    Returns neural (len(repeats) * 900, N) uint8 and labels (len(repeats) * 900, 768) float32.
    The labels are frame_features tiled once per repeat, the same as the trainer's repeat(10, 1).
    """
    reps = list(repeats)
    neural = demo.events[reps].reshape(-1, demo.n_neurons)
    labels = np.tile(demo.frame_features, (len(reps), 1)).astype(np.float32)
    return neural, labels


def session_arrays(demo: AllenDemo, repeats: Sequence[int] = TRAIN_REPEATS):
    """Per-session (neural (T, n_i) uint8, labels (T, 768) float32) pairs, in model order."""
    neural, labels = repeat_frames(demo, repeats)
    return [(x, labels) for x in split_sessions(neural, demo.session_n_neurons)]


def session_tensor_datasets(demo: AllenDemo, repeats: Sequence[int] = TRAIN_REPEATS, device: str = "cpu"):
    """One cebra.data.TensorDataset per session (234), with continuous = DINO labels.

    Pass the list to cebra.data.UnifiedxCEBRADataset(*datasets) (the class the trainer uses).
    Offset is left at its default; the model's configure_for() sets it.
    """
    import cebra.data   # lazy import: the rest of this module needs numpy only
    return [
        cebra.data.TensorDataset(neural=x.astype(np.float32), continuous=y, device=device)
        for x, y in session_arrays(demo, repeats)
    ]


def download_cache(url: str = CACHE_URL, path=DEFAULT_CACHE, overwrite: bool = False) -> pathlib.Path:
    """Fetch allen_demo_cache.npz from the figshare archive to `path` (skipped if it exists)."""
    path = pathlib.Path(path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = path.with_suffix(".zip.part")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=600) as r, open(archive_path, "wb") as f:
            shutil.copyfileobj(r, f)
        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError(f"Expected a zip archive from {url}.")
        with zipfile.ZipFile(archive_path) as archive:
            members = [m for m in archive.namelist() if pathlib.PurePosixPath(m).name == path.name]
            if len(members) != 1:
                raise RuntimeError(f"Expected one {path.name} in the archive from {url}, found {len(members)}.")
            with archive.open(members[0]) as source, open(tmp, "wb") as destination:
                shutil.copyfileobj(source, destination)
        tmp.replace(path)
    finally:
        archive_path.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return path


def events_from_visual_drift(meta, visual_drift_dir) -> np.ndarray:
    """Event trains of the demo neurons, read from the processed Allen .mat files.

    meta: an AllenDemo or the cache arrays (needs session_id, session_area, session_cell_class,
    session_n_neurons, neuron_mat_column). Each session file is
    ``<visual_drift_dir>/<calcium_excitatory|calcium_inhibitory>/<area>/<session id>.mat``; its
    ``filtered_traces_days_events[0, 0]`` holds (neurons, 10 repeats x 900 frames) binary events for natural movie
    one of session A. The neurons of the model are the columns ``neuron_mat_column`` of each file, in model order.

    Returns (10, 900, N) uint8, i.e. ``AllenDemo.events``.
    """
    import scipy.io   # lazy: only needed for this route
    get = (lambda k: getattr(meta, k)) if isinstance(meta, AllenDemo) else (lambda k: meta[k])
    root = pathlib.Path(visual_drift_dir)
    n_per = [int(n) for n in get("session_n_neurons")]
    columns = np.asarray(get("neuron_mat_column"))
    out, start = [], 0
    for sid, area, cls, n in zip(get("session_id"), get("session_area"), get("session_cell_class"), n_per):
        path = root / NEURON_TYPE_DIR[str(cls)] / str(area) / f"{int(sid)}.mat"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: is {root} the visual_drift 'data' folder "
                                    "(with calcium_excitatory/ and calcium_inhibitory/)?")
        m = np.asarray(scipy.io.loadmat(str(path))["filtered_traces_days_events"][0, 0])
        if m.shape[1] % 900:
            raise ValueError(f"{path}: expected 10 x 900 frames per neuron, got {m.shape[1]}")
        cols = columns[start:start + n]
        out.append(m[cols].T.reshape(-1, 900, n))
        start += n
    return np.concatenate(out, axis=2).astype(np.uint8)


def wrapped_window_indices(start: int, end: int, length: int, offset) -> np.ndarray:
    """Input samples that give one model output per reference sample of a looping stimulus.

    For a stimulus shown back to back (e.g. movie repeats), the model window of the first and last
    samples wraps around: for reference samples ``start`` to ``end - 1``, returns the samples
    ``start - left`` to ``end + right - 2`` modulo ``length``. ``offset`` has ``left`` and ``right``
    (a CEBRA ``model.get_offset()``).
    """
    left, right = int(offset.left), int(offset.right)
    if not 0 <= start < end:
        raise ValueError(f"Need 0 <= start < end, got start={start}, end={end}.")
    if int(length) <= 0:
        raise ValueError(f"length must be positive, got {length}.")
    return np.arange(start - left, end + right - 1) % int(length)
