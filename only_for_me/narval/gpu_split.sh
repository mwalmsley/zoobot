#!/bin/bash
#SBATCH --time=0:10:0  
#SBATCH --nodes=2             # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...). This is PER TASK.
# srun is slurm's way to start many jobs from the same sbatch script
# the sbatch script runs *once* and then the srun command is called ntasks-per-node times on each node
# Lightning knows via env variables that it is running on slurm and identifies which DDP instance it should spin up
# webdatasets then reads from lighting with LOCAL_RANK worker we're on and loads the appropriate data
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:v100:4  # This needs to match Trainer(devices=...). This is PER TASK. Total GPU = nodes*devices

# https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_1.html#setup-the-training-script
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html



nvidia-smi

PYTHON=/home/walml/envs/zoobot39_dev/bin/python

# mkdir $SLURM_TMPDIR/cache
# mkdir /tmp/cache

export NCCL_BLOCKING_WAIT=1  # "Set this environment variable if you wish to use the NCCL backend for inter-GPU communication."
# instructed by Compute Canada, not lightning

echo 'Running script'
REPO_DIR=/project/def-bovy/walml/zoobot
srun $PYTHON $REPO_DIR/only_for_me/narval/gpu_split.py --gpus 4 --nodes 2
