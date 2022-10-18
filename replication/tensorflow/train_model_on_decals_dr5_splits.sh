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

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

nvidia-smi

BASE_DIR=/share/nas2/walml  
# BASE_DIR=/nvme1/scratch/walml  # local (copy-paste into terminal)

ZOOBOT_DIR=$BASE_DIR/repos/zoobot
PYTHON=$BASE_DIR/miniconda3/envs/zoobot/bin/python
RESULTS_DIR=$BASE_DIR/repos/gz-decals-classifiers/results

# make these with shards_sbatch
# W+22a originally trained with validation data from an automatic subset of the train data,
# so there's no explictly-known validation set
# instead, I am going to use the very-slightly-different catalogs in pytorch-galaxy-datasets
# (I also do this for the pytorch version, where it's easier)

# to download, run decals_dr5_setup - see replication/pytorch/train_model_on_decals_dr5_splits.py
DATA_DIR=$BASE_DIR/repos/_data/decals_dr5

EXPERIMENT_DIR=$RESULTS_DIR/tensorflow/dr5/efficientnet_dr5_tensorflow_greyscale_catalog

$PYTHON $ZOOBOT_DIR/replication/tensorflow/train_model_on_decals_dr5_splits.py \
    --experiment-dir $EXPERIMENT_DIR \
    --resize-size 224 \
    --data-dir $DATA_DIR \
    --epochs 200 \
    --batch-size 512 \
    --gpus 2 \
    --wandb
