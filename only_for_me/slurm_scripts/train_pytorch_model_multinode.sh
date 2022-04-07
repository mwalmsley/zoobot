#!/bin/bash
#SBATCH --job-name=pyslur                     # Job name
#SBATCH --output=pyslur_%A.log 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 8
#SBATCH --mem=0
#SBATCH --constraint=A100
#SBATCH --time=23:00:00    
#SBATCH --exclude=compute-0-[5-7]


   #SBATCH --exclude compute-0-[0-25,27-40,42-49,100-106]

  

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

echo 'Run python script'

$PYTHON /share/nas2/walml/repos/zoobot/zoobot/pytorch/examples/train_model.py \
    --experiment-dir $EXPERIMENT_DIR \
    --shard-img-size 300 \
    --resize-size 224 \
    --catalog ${THIS_DIR}/data/decals/shards/all_campaigns_ortho_v2/dr5/labelled_catalog.csv \
    --epochs 200 \
    --batch-size 256 \
    --distributed \
    --wandb

# with DDP, batch size is per gpu
