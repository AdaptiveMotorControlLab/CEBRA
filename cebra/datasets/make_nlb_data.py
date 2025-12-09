"""
Example on how to run in folder cebra/datasets:
```
python make_nlb_data.py --dataset_name mc_maze_large
```
"""

import argparse
import os
import pathlib

import joblib as jl
import numpy as np
import scipy.signal as signal
from nlb_tools.make_tensors import make_eval_input_tensors
from nlb_tools.make_tensors import make_eval_target_tensors
from nlb_tools.make_tensors import make_train_input_tensors
from nlb_tools.nwb_interface import NWBDataset

dataset_dict = {
    "mc_maze": "000128/sub-Jenkins/",
    "mc_maze_small": "000140/sub-Jenkins/",
    "mc_maze_medium": "000139/sub-Jenkins/",
    "mc_maze_large": "000138/sub-Jenkins/",
    "mc_rtt": "000129/sub-Indy",
    "area2_bump": "000127/sub-Han/",
    "dmfc_rsg": "000130/sub-Haydn",
}

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default="/data/celia/nlb")
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--savepath",
                    type=str,
                    default="/mnt/data_storage/celia/nlb/data")
parser.add_argument("--bin_width", type=int, default=5)
parser.add_argument("--smoothing_width", type=int, default=50)
parser.add_argument("--split", type=str, default="train")

args = parser.parse_args()
args.datapath = pathlib.Path(args.datapath)
args.savepath = pathlib.Path(args.savepath)

if args.dataset_name not in dataset_dict.keys():
    raise NotImplementedError(f"{args.dataset_name}")

if args.split == "full-train":
    prefix = ""
    phase = "test"

elif args.split != "test":
    prefix = "*train"
    phase = "val"

elif args.split == "test":
    prefix = "*test"
    phase = "test"

dataset = NWBDataset(args.datapath / dataset_dict[args.dataset_name], prefix)
dataset.resample(args.bin_width)

suffix = "" if (args.bin_width == 5) else f"_{int(round(args.bin_width))}"

if args.smoothing_width != 0:
    smoothing_tag = f"_smth{args.smoothing_width}"
else:
    smoothing_tag = ""


def smoothing(kern_sd, spikes, bin_size_ms):
    window = signal.gaussian(int(6 * kern_sd / bin_size_ms),
                             int(kern_sd / bin_size_ms),
                             sym=True)
    window /= np.sum(window)

    def filt(x):
        return np.convolve(x, window, "same")

    if kern_sd != 0:
        convolved = np.apply_along_axis(filt, 1, spikes)
        print(np.any(np.isnan(convolved)))
        return convolved
    else:
        return spikes


eval_dict = make_eval_input_tensors(dataset,
                                    dataset_name=args.dataset_name,
                                    trial_split=phase,
                                    save_file=False)
eval_spikes_heldin = eval_dict["eval_spikes_heldin"]

if args.split != "test":
    train_trial_split = "train" if (phase == "val") else ["train", "val"]
    train_dict = make_train_input_tensors(
        dataset,
        dataset_name=args.dataset_name,
        trial_split=train_trial_split,
        save_file=False,
        include_behavior=True,
        include_forward_pred=False,
    )

    train_spikes_heldin = train_dict["train_spikes_heldin"]
    train_spikes_heldout = train_dict["train_spikes_heldout"]

    # Generate target data
    target_dict = make_eval_target_tensors(
        dataset,
        dataset_name=args.dataset_name,
        train_trial_split="train",
        eval_trial_split="val",
        include_psth=False,
        save_file=False,
    )

    train_behavior = target_dict[f"{args.dataset_name}{suffix}"][
        "train_behavior"]

    if args.split == "train":
        eval_spikes_heldout = eval_dict["eval_spikes_heldout"]
        eval_behavior = target_dict[f"{args.dataset_name}{suffix}"][
            "eval_behavior"]
        data_dict = {
            "train_behavior":
                train_behavior,
            "eval_behavior":
                eval_behavior,
            "train_neural_heldin":
                smoothing(args.smoothing_width, train_spikes_heldin,
                          args.bin_width),
            "eval_neural_heldin":
                smoothing(args.smoothing_width, eval_spikes_heldin,
                          args.bin_width),
            "train_neural_heldout":
                train_spikes_heldout,
            "eval_neural_heldout":
                eval_spikes_heldout,
        }
    elif args.split == "full-train":
        data_dict = {
            "train_behavior":
                train_behavior,
            "train_neural_heldin":
                smoothing(args.smoothing_width, train_spikes_heldin,
                          args.bin_width),
            "train_neural_heldout":
                train_spikes_heldout,
            "eval_neural_heldin":
                smoothing(args.smoothing_width, eval_spikes_heldin,
                          args.bin_width),
        }

elif args.split == "test":
    data_dict = {
        "eval_neural_heldin":
            smoothing(args.smoothing_width, eval_spikes_heldin, args.bin_width)
    }

jl.dump(
    data_dict,
    args.savepath /
    f"{args.dataset_name}_{args.split}{suffix}{smoothing_tag}.jl",
)

if args.split != "test":
    jl.dump(
        target_dict,
        args.savepath / f"target_dict_{args.dataset_name}{suffix}.jl",
    )
