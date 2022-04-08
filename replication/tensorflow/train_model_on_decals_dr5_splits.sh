#!/bin/bash
#SBATCH --job-name=dr5-rep-tf                        # Job name
#SBATCH --output=dr5-rep-tf_%A.log 
#SBATCH --mem=0                                     # Job memory request
#SBATCH --cpus-per-task=24
#SBATCH --ntasks 1
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node
#SBATCH --exclude compute-0-7,compute-0-5

# Train Zoobot (tensorflow version) on DR5 shards from scratch on Slurm cluster.
# You will need to adjust various paths
# Run with sbatch replication/replicate.sh

pwd; hostname; date

nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

# make these with shards_sbatch
# W+22a originally trained with validation data from an automatic subset of the train data,
# so there's no explictly-known validation set
TFRECORD_DIR=/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_2p5_unfiltered_n2
TRAIN_DIR=$TFRECORD_DIR/train_shards
TEST_DIR=$TFRECORD_DIR/eval_shards

THIS_DIR=/share/nas2/walml/repos/gz-decals-classifiers

EXPERIMENT_DIR=$THIS_DIR/results/tensorflow/dr5/efficientnet_dr5_tensorflow

$PYTHON $ZOOBOT_DIR/zoobot/tensorflow/examples/train_model.py \
    --experiment-dir $EXPERIMENT_DIR \
    --shard-img-size 300 \
    --resize-size 224 \
    --train-dir $TRAIN_DIR \
    --test-dir $EVAL_DIR \
    --epochs 200 \
    --batch-size 512 \
    --gpus 2  \
    --wandb


# you might make (see shards.sh) and then train on some very small shards to debug everything first

# TRAIN_DIR=$ZOOBOT_DIR/data/decals/shards/decals_debug/train_shards
# EVAL_DIR=$ZOOBOT_DIR/data/decals/shards/decals_debug/eval_shards

# EXPERIMENT_DIR=$THIS_DIR/results/decals_debug

# python $ZOOBOT_DIR/train_model.py \
#     --experiment-dir $EXPERIMENT_DIR \
#     --shard-img-size 32 \
#     --resize-size 224 \
#     --train-dir $TRAIN_DIR \
#     --eval-dir $EVAL_DIR \
#     --epochs 2 \
#     --batch-size 8
