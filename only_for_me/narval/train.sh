#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=0:20:0  
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

### SBATCH --gres=gpu:a100:1

nvidia-smi

PYTHON=/home/walml/envs/zoobot39_dev/bin/python

mkdir $SLURM_TMPDIR/cache

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
# export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
# echo "r$SLURM_NODEID master: $MASTER_ADDR"
# echo "r$SLURM_NODEID Launching python script"

REPO_DIR=/project/def-bovy/walml/zoobot/
srun $PYTHON $REPO_DIR/only_for_me/narval/train.py --save-dir $REPO_DIR/only_for_me/narval/debug_models --batch-size 4 --color --debug
# srun python $SLURM_TMPDIR/zoobot/only_for_me/narval/finetune.py
