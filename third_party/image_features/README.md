# Utilities for computing image features to use as behavior labels.

Currently uses [DINO](https://github.com/facebookresearch/dino) for obtaining features.

## Quickstart (SLURM)

Start by generating a parameter file:

``` bash
$ python3 sweep.py dino > missing.sweep
```

Make sure to have a `pytorch.simg` singularity image in this folder. Details on building will follow.

Now, run all jobs in `missing.sweep` as a job array:

``` bash
sbatch -a 1-$(wc -l missing.sweep) dino.sh
```

Logs will be placed into `logs/`, computed features and the corresponding log files for all evaluated models
will be placed in `features/`.

Some commands interesting/useful for debugging:

```bash
# Only launch the first job in missing.sweep:
$ sbatch -a 1 dino.sh
# Check the latest log files:
$ watch "ls logs/*.* | sort | tail -n2 | xargs tail -n30" 
```

## Third party code

- [DINO](https://github.com/facebookresearch/dino) is Apache 2.0 licensed, and we obtained the code from commit `de9ee3df6cf39fac952ab558447af1fa1365362a`. Modifications in `eval_knn.py` and `movie_dataset.py`.
