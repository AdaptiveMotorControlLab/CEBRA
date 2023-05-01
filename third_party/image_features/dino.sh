#!/bin/bash

#SBATCH --job-name=allen-video-features
#SBATCH --error=logs/allen-video-%j-%A-%a.err
#SBATCH --output=logs/allen-video-%j-%A-%a.out 

#SBATCH --nodes=1                 
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

DATADIR=allen_movies/movies

params=( $(cat missing.sweep | sed ${SLURM_ARRAY_TASK_ID:-1}'q;d') )
video=${params[0]}
model=${params[1]}
patchsize=${params[2]}

logdir=features/allen_movies/$model/$patchsize/${video}
logfile=$logdir/$(date +%y%m%d-%H%M%S-%N)_knn.log

mkdir -p $(pwd)/$logdir
echo Data:  $video
echo Model: $model $patchsize
echo Log:   $logdir

# Start the actual work:
mkdir -p $logdir

ARGS="--load_features /app/$logdir"
if [[ ! -f $logdir/testfeat.pth ]]; then
  echo need to compute features.
  echo storing to $logdir
  ARGS="--dump_features /app/$logdir"
fi

singularity exec \
--nv \
-B $(pwd):/app -W /app \
pytorch.simg python -m torch.distributed.launch --master_port "$(( 16000 + $RANDOM ))"\
    --nproc_per_node=1 /app/dino/eval_knn.py \
    --arch $model --patch_size $patchsize  \
    --data_path /app/${DATADIR}/$video \
    --batch_size 32 \
    $ARGS  |& tee ${logfile} || touch ${logfile}.fail

# chmod a-w -R $logdir
touch ${logfile}.done
echo DONE.
