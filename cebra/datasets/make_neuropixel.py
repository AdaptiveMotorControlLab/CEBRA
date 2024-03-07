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
"""Generate pseudomouse Neuropixels data.

This script generates the pseudomouse Neuropixels data for each visual cortical area from original Allen ENuropixels Brain observatory 1.1 NWB data.
We followed the units filtering used in the AllenSDK package.

References:
    *Siegle, Joshua H., et al. "Survey of spiking in the mouse visual system reveals functional hierarchy." Nature 592.7852 (2021): 86-92.
    *https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html

"""
import argparse
import glob

import h5py
import joblib as jl
import numpy as np
import numpy.typing as npt
import pandas as pd


def _filter_units(
    unit_ids: npt.NDArray[np.int64],
    isi_violations: npt.NDArray[np.float64],
    amplitude_cutoff: npt.NDArray[np.float64],
    presence_ratio: npt.NDArray[np.float64],
    quality: npt.NDArray[object],
    amp_thr: float = 0.1,
    ratio_thr: float = 0.95,
    isi_thr: float = 0.5,
):
    """Filter recording units with the conditions defined by the arguments.

    The units above `amp_thr`, `ratio_thr` and below `isi_thr` with the 'good' quality are returned.

    Args:
        unit_ids: Array of unit ids.
        isi_violations: The isi violation levels of each unit.
        amplitude_cutoff: The amplitude cutoff of each unit.
        prsence_ratio: The presence_ratio of each unit.
        quality: The recording quality of each unit.
        amp_thr: The amplitude threshold to filter.
        ratio_thr: The presence ratio threshold to filter.
        isi_thr: The isi level threshold to filter.

    See Also:
        https://github.com/AllenInstitute/AllenSDK/blob/a73b9dbf65d3b03d5c244fad0b271b141afebce1/allensdk/brain_observatory/ecephys/__init__.py
        https://github.com/AllenInstitute/AllenSDK/blob/a73b9dbf65d3b03d5c244fad0b271b141afebce1/allensdk/brain_observatory/ecephys/ecephys_session_api/ecephys_nwb_session_api.py

    """

    quality = np.array([q for q in quality])
    filtered_unit_idx = ((isi_violations <= isi_thr) &
                         (presence_ratio >= ratio_thr) &
                         (amplitude_cutoff <= amp_thr) & (quality == "good"))

    return filtered_unit_idx


def _spikes_by_units(spike_times: npt.NDArray[np.float64],
                     spike_time_index: npt.NDArray[np.int64]):
    """Make array of spike times of each unit.

    The spike times belong to each spike unit are filtered and return the list of spike times of each unit.

    Args:
        spike_times: Spiking times of multiple units.
        spike_time_index: The index of the recording unit corresponding to the recorded spike.

    """

    units = []
    for n, t in enumerate(spike_time_index):
        if n == 0:
            unit = spike_times[:t]
        elif n != len(spike_time_index) - 1:
            unit = spike_times[spike_time_index[n - 1]:t]
        else:
            unit = spike_times[t:]
        units.append(unit)

    return units


def _get_area(
    area: npt.NDArray[str],
    peak_channel_id: npt.NDArray[np.int64],
    electrode_id: npt.NDArray[np.int64],
):
    """Read the area where each unit was recording.

    The corresponding recording visual cortical areas of the recording units are returned.

    Args:
        area: The list of areas registered for each peak channel.
        peak_channel_id: The list of peak channel ids.
        electrode_id: The list of electrode ids.

    """

    units = np.empty(len(peak_channel_id), dtype="object")
    for n, i in enumerate(electrode_id):
        units[peak_channel_id == i] = area[n]

    return units


def _get_movie1(start: float, stop: float, unit_spikes):
    """Get spike timings during the movie1 stimulus block.

    The spike times of each unit during the movie1 stimulus block are returned.

    Args:
        start: The start time point of the movie1 stimulus block.
        stop: The end time point of the movie1 stimulus block.
        unit_spikes: The list of spike times of a single unit.

    """

    if len(unit_spikes) == 0:
        return np.array([])
    start_index = (start < unit_spikes).argmax()
    end_index = (unit_spikes > stop).argmax()

    return unit_spikes[start_index:end_index]


def _spike_counts(bin_edges: npt.NDArray[np.float64], units: list):
    """Get spike counts of defined bins.

    Given spike times of each unit is placed in the defined bins to return spike counts.

    Args:
        bin_edges: The bins to count the spike counts.
        units: The list of spiking time array of the recording units

    """

    spike_matrix = np.zeros((len(bin_edges) - 1, len(units)))
    for i, unit_spikes in enumerate(units):
        hist, bin_edges_hist = np.histogram(unit_spikes, bin_edges)
        spike_matrix[:, i] = hist

    return spike_matrix


def read_neuropixel(
    path: str = "/shared/neuropixel/*/*.nwb",
    cortex: str = "VISp",
    sampling_rate: float = 120.0,
):
    """Load 120Hz Neuropixels data recorded in the specified cortex during the movie1 stimulus.

    The Neuropixels recordin is filtered and transformed to spike counts in a bin size specified by the sampling rat.

    Args:
        path: The wildcard file path where the neuropixels .nwb files are located.
        cortex: The cortex where the neurons were recorded. Choose from VISp, VISal, VISrl, VISl, VISpm, VISam.
        sampling_rate: The sampling rate for spike counts to process the raw data.

    """

    files = glob.glob(path)
    sessions = {}
    len_recording = []
    session_frames = []
    for f in files:
        with h5py.File(f, "r") as d:
            print("read one session and filter for area and quality")
            if "brain" in d["general/stimulus"][...].item():
                area_list = d[
                    "general/extracellular_ephys/electrodes/location"][...]
                start_time = d[
                    "intervals/natural_movie_one_presentations/start_time"][...]
                end_time = d[
                    "intervals/natural_movie_one_presentations/stop_time"][...]
                timeseries = d[
                    "intervals/natural_movie_one_presentations/timeseries"][...]
                timeseries_index = d[
                    "intervals/natural_movie_one_presentations/timeseries_index"][
                        ...]
                session_no = d["identifier"][...].item()
                spike_time_index = d["units/spike_times_index"][...]
                spike_times = d["units/spike_times"][...]
                ids = d["units/id"][...]
                amplitude_cutoff = d["units/amplitude_cutoff"][...]
                presence_ratio = d["units/presence_ratio"][...]
                isi_violations = d["units/isi_violations"][...]
                quality = d["units/quality"][...]

                peak_channel_id = d["units/peak_channel_id"][...]
                electrode_id = d["general/extracellular_ephys/electrodes/id"][
                    ...]
                unit_spikes = _spikes_by_units(spike_times, spike_time_index)
                filtered_quality = _filter_units(ids, isi_violations,
                                                 amplitude_cutoff,
                                                 presence_ratio, quality)
                unit_areas = _get_area(area_list, peak_channel_id, electrode_id)
                units_in_movie1 = []

                for unit in unit_spikes:
                    spikes_in_movie1 = _get_movie1(start_time[0],
                                                   end_time[8999], unit)
                    units_in_movie1.append(spikes_in_movie1)

                filtered_unit = []

                for area, quality, unit in zip(unit_areas, filtered_quality,
                                               units_in_movie1):
                    if area == cortex and quality:
                        filtered_unit.append(unit)
                bin_edges = np.arange(start_time[0], end_time[8999],
                                      1 / sampling_rate)
                movie_frame = np.digitize(
                    bin_edges, start_time[:9000], right=False) - 1
                spike_matrix = _spike_counts(bin_edges, filtered_unit)
                len_recording.append(spike_matrix.shape[0])
                sessions[session_no] = spike_matrix
                session_frames.append(movie_frame % 900)
    print("Build pseudomouse")
    for session_key in sessions.keys():
        sessions[session_key] = sessions[session_key][:sampling_rate * 10 * 30]
    for i in range(len(session_frames)):
        session_frames[i] = np.tile(np.repeat(np.arange(900), 4), 10)

    return sessions, session_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-rate", default=120, type=float)
    parser.add_argument("--cortex", default="VISp", type=str)
    parser.add_argument("--save-path",
                        default="/data/allen_movie1_neuropixel/VISp/",
                        type=str)
    args = parser.parse_args()
    sessions_dic, session_frames = read_neuropixel(
        cortex=args.cortex, sampling_rate=args.sampling_rate)
    pseudo_mice = np.concatenate([v for v in sessions_dic.values()], axis=1)
    pseudo_mice_frames = session_frames[0]

    jl.dump({
        "neural": sessions_dic,
        "frames": session_frames
    },
            Path(args.save_path) /
            f"neuropixel_sessions_{int(args.sampling_rate)}_filtered.jl")
    jl.dump(
        {
            "neural": pseudo_mice,
            "frames": pseudo_mice_frames
        },
        Path(args.save_path) /
        f"neuropixel_pseudomouse_{int(args.sampling_rate)}_filtered.jl",
    )
