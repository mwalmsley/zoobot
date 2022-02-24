#!/bin/bash
#SBATCH --job-name=spiral_yn                     # Job name
#SBATCH --output=replicate-decals_%A.log 
#SBATCH --mem=32gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

ZOOBOT_DIR=/share/nas/walml/repos/zoobot_test  # be careful if zoobot vs zoobot_test
PYTHON=/share/nas/walml/miniconda3/envs/zoobot/bin/python

TFRECORD_DIR=/share/nas/walml/repos/gz-decals-classifiers/data/decals/shards/all_2p5_unfiltered_n2
TRAIN_DIR=$TFRECORD_DIR/train_shards
EVAL_DIR=$TFRECORD_DIR/eval_shards

THIS_DIR=/share/nas/walml/repos/gz-decals-classifiers

EXPERIMENT_DIR=$THIS_DIR/results/replicated_train_only_spiral_yn_only  # excludes 10k eval galaxies

$PYTHON /share/nas/walml/repos/gz-decals-classifiers/train_single_task_model.py \
    --experiment-dir $EXPERIMENT_DIR \
    --shard-img-size 300 \
    --resize-size 224 \
    --train-dir $TRAIN_DIR \
    --eval-dir $EVAL_DIR \
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
