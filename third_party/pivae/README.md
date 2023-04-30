# piVAE and conv-piVAE experiments

The repository has a code for the experiments using [piVAE](https://github.com/zhd96/pi-vae/tree/d5db0794fa8dc3f4ae9793e049a5e9a762a9f729) and conv-piVAE.
We adapted `pi_vae.py` and `conv_pi_vae.py`

## Quickstart

Train a model for rat hippocampus using conv-piVAE.

``` bash
# For conv-piVAE implementation
$ export PYTHONPATH=../../
$ mkdir conv_pivae_rat
$ python task.py --data-path rat-hippocampus-achilles-3fold-trial-split-0 --time-window 10 --epochs 500 --save-flag --save-best-only /
--logdir conv_pivae_rat --lr 0.00025 --batch-size 200 --latent-dim 2 --label-dim 3 --task HippocampusCV --mode continuous

# To launch the decoding using monte-carlo sampling,
$ python mc_decode.py --path conv_pivae_rat --savepath conv_pivae_rat --sampling-num 100

```

The trained model, log and the decoding result will be saved in `conv_pivae_rat`.
