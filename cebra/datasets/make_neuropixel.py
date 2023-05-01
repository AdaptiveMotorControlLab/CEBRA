
def _spikes_by_units(spike_times: npt.NDArray[np.float64],
                     spike_time_index: npt.NDArray[np.int64]):
def _get_movie1(start: float, stop: float, unit_spikes):
def _spike_counts(bin_edges: npt.NDArray[np.float64], units: list):
                                                 amplitude_cutoff,
                                                 presence_ratio, quality)
                    spikes_in_movie1 = _get_movie1(start_time[0],
                                                   end_time[8999], unit)
    sessions_dic, session_frames = read_neuropixel(
        cortex=args.cortex, sampling_rate=args.sampling_rate)
    jl.dump(
        {
        },
        os.path.join(
            args.save_path,
