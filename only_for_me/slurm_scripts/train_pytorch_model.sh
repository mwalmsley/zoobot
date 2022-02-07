#!/bin/bash
#SBATCH --job-name=pytorch                     # Job name
#SBATCH --output=pytorch_%A.log 
#SBATCH --mem=32gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

THIS_DIR=/share/nas2/walml/repos/gz-decals-classifiers

EXPERIMENT_DIR=$THIS_DIR/results/pytorch_debug

$PYTHON /share/nas2/walml/repos/zoobot/zoobot/pytorch/examples/train_model.py \
    --experiment-dir $EXPERIMENT_DIR \
    --shard-img-size 300 \
    --resize-size 224 \
    --catalog ${THIS_DIR}/data/decals/shards/all_campaigns_ortho_v2/dr5/labelled_catalog.csv \
    --epochs 200 \
    --batch-size 512 \
    --distributed  \
    --wandb

# $PYTHON $ZOOBOT_DIR/train_model.py \
#     --experiment-dir $EXPERIMENT_DIR \
#     --shard-img-size 300 \
#     --resize-size 224 \
#     --train-dir $TRAIN_DIR \
#     --eval-dir $EVAL_DIR \
#     --epochs 200 \
#     --batch-size 512 \
#     --distributed  \
#     --wandb