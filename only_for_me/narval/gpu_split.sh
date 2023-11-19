#!/bin/bash
#SBATCH --time=0:10:0  
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...). Presumably is PER TASK.
# this .sh always runs once, but the srun command (for parallel slurm tasks) gets called twice (ntasks). Lightning then hooks onto that by detecting it is in a slurm environment.
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:v100:2  # This needs to match Trainer(devices=...). Presumably is PER TASK. Total GPU = nodes*devices

# https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_1.html#setup-the-training-script
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

#### SBATCH --ntasks-per-node=1

nvidia-smi

PYTHON=/home/walml/envs/zoobot39_dev/bin/python
# source ~/envs/zoobot39_dev/bin/activate

# mkdir $SLURM_TMPDIR/cache
# mkdir /tmp/cache

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
# instructed by Compute Canada, not lightning

# export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
# echo "r$SLURM_NODEID master: $MASTER_ADDR"
# echo "r$SLURM_NODEID Launching python script"

echo 'Running script'
REPO_DIR=/project/def-bovy/walml/zoobot
srun $PYTHON $REPO_DIR/only_for_me/narval/gpu_split.py --gpus 2

