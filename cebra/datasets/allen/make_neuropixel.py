"""Generate pseudomouse Neuropixels data.

This script generates the pseudomouse Neuropixels data for each visual cortical area from original Allen ENuropixels Brain observatory 1.1 NWB data.
We followed the units filtering used in the AllenSDK package.

References:
    *Siegle, Joshua H., et al. "Survey of spiking in the mouse visual system reveals functional hierarchy." Nature 592.7852 (2021): 86-92.
    *https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html

"""

import argparse
import glob
import os

import h5py
import joblib as jl
import numpy as np
import numpy.typing as npt
import pandas as pd

from cebra.datasets import get_datapath


    """Filter recording units with the conditions defined by the arguments.
    The units above `amp_thr`, `ratio_thr` and below `isi_thr` with the 'good' quality are returned.
    See `AllenSDK`_ for the default filter values.
    Args:
        isi_violations: The isi violation levels of each unit.
        amplitude_cutoff: The amplitude cutoff of each unit.
        prsence_ratio: The presence_ratio of each unit.
        quality: The recording quality of each unit.
        amp_thr: The amplitude threshold to filter.
        ratio_thr: The presence ratio threshold to filter.
        isi_thr: The isi level threshold to filter.

    .. _AllenSDK: https://github.com/AllenInstitute/AllenSDK/blob/a73b9dbf65d3b03d5c244fad0b271b141afebce1/allensdk/brain_observatory/ecephys/__init__.py
    """

    quality = np.array([q for q in quality])

    return filtered_unit_idx


def _spikes_by_units(spike_times: npt.NDArray[np.float64],
                     spike_time_index: npt.NDArray[np.int64]):
    """Make array of spike times of each unit.
    Args:
        spike_times: Spiking times of multiple units.
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


    """Read the area where each unit was recording.
        area: The list of areas registered for each peak channel.
        electrode_id: The list of electrode ids.
    """

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
    Args:
        units: The list of spiking time array of the recording units
    """

    spike_matrix = np.zeros((len(bin_edges) - 1, len(units)))
    for i, unit_spikes in enumerate(units):
        hist, bin_edges_hist = np.histogram(unit_spikes, bin_edges)
        spike_matrix[:, i] = hist

    return spike_matrix


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
                area_list = d[
                start_time = d[
                end_time = d[
                timeseries = d[
                timeseries_index = d[
                        ...]
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
    for session_key in sessions.keys():
        sessions[session_key] = sessions[session_key][:sampling_rate * 10 * 30]
    for i in range(len(session_frames)):
        session_frames[i] = np.tile(np.repeat(np.arange(900), 4), 10)

    return sessions, session_frames


    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path",
                        default=get_datapath("allen_movie1_neuropixel/VISp/"),
                        type=str)
    parser.add_argument("--sampling-rate", default=120, type=float)
    args = parser.parse_args()
    sessions_dic, session_frames = read_neuropixel(
        path=args.data_path,
        cortex=args.cortex,
        sampling_rate=args.sampling_rate)
    pseudo_mice = np.concatenate([v for v in sessions_dic.values()], axis=1)
    pseudo_mice_frames = session_frames[0]

    jl.dump(
        {
        },
        os.path.join(
            args.save_path,
