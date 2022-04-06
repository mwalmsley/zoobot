#!/bin/bash
#SBATCH --job-name=pytorch                     # Job name
#SBATCH --output=pytorch_%A.log 
#SBATCH --mem=0                                     # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24
#SBATCH --exclude=compute-0-7

pwd; hostname; date

nvidia-smi

export WANDB_CACHE_DIR=/share/nas2/walml/wandb_cache
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64  # maybe not needed

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

THIS_DIR=/share/nas2/walml/repos/gz-decals-classifiers

EXPERIMENT_DIR=$THIS_DIR/results/pytorch/resnet_debug_det_2gpu_b256_another_node

# $PYTHON /share/nas2/walml/repos/zoobot/zoobot/pytorch/estimators/cuda_check.py \

# --catalog does nothing currently because I provide the splits explicitly
$PYTHON /share/nas2/walml/repos/zoobot/zoobot/pytorch/examples/train_model.py \
    --experiment-dir $EXPERIMENT_DIR \
    --shard-img-size 300 \
    --resize-size 224 \
    --color \
    --catalog ${THIS_DIR}/data/decals/shards/all_campaigns_ortho_v2/dr5/labelled_catalog.csv \
    --epochs 200 \
    --batch-size 256 \
    --gpus 2  \
    --nodes 1 \
    --wandb
