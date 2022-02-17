#!/bin/bash
#SBATCH --job-name=pyslur                     # Job name
#SBATCH --output=clus_%A_%a.log 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task 12
#SBATCH --mem=20gb
#SBATCH --constraint=A100
#SBATCH --time=23:00:00    
#SBATCH --exclude=compute-0-[5-7]
#SBATCH --array=[1-4]

# https://github.com/PyTorchLightning/pytorch-lightning/issues/4612

pwd; hostname; date

nvidia-smi

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

THIS_DIR=/share/nas2/walml/repos/gz-decals-classifiers

EXPERIMENT_DIR=$THIS_DIR/results/early_stopping_slurm_multinode

echo $SLURM_ARRAY_TASK_ID

# export MASTER_ADDR=127.0.0.1
export MASTER_ADDR=galahad:192.168.1.1
export MASTER_PORT=52386
export WORLD_SIZE=4

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    export NODE_RANK=0
    export LOCAL_RANK=0
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    export NODE_RANK=0
    export LOCAL_RANK=1
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    export NODE_RANK=1
    export LOCAL_RANK=0
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then
    export NODE_RANK=1
    export LOCAL_RANK=1
else
  echo $SLURM_ARRAY_TASK_ID not understood
fi

echo $NODE_RANK $LOCAL_RANK $MASTER_ADDR $MASTER_PORT


# echo 'Run python script'

$PYTHON /share/nas2/walml/repos/zoobot/zoobot/pytorch/examples/train_model.py \
    --experiment-dir $EXPERIMENT_DIR \
    --shard-img-size 300 \
    --resize-size 224 \
    --catalog ${THIS_DIR}/data/decals/shards/all_campaigns_ortho_v2/dr5/labelled_catalog.csv \
    --epochs 200 \
    --batch-size 256 \
    --nodes 2 \
    --wandb

# with DDP, batch size is per gpu
