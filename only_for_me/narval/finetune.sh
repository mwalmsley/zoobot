#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:15:0      
# SBATCH --gres=gpu:a100:1

#### SBATCH --mail-user=<youremail@gmail.com>
#### SBATCH --mail-type=ALL

module load StdEnv/2020  # CUDA etc

PYTHON=/home/walml/envs/zoobot39_dev/bin/python

mkdir $SLURM_TMPDIR/walml
mkdir $SLURM_TMPDIR/walml/finetune
mkdir $SLURM_TMPDIR/walml/finetune/data
mkdir $SLURM_TMPDIR/walml/finetune/checkpoints

cp -r /project/def-bovy/walml/data/roots/galaxy_mnist $SLURM_TMPDIR/walml/finetune/data/

ls $SLURM_TMPDIR/walml/finetune/data/galaxy_mnist

$PYTHON /project/def-bovy/walml/zoobot/only_for_me/narval/finetune.py

ls $SLURM_TMPDIR/walml/finetune/checkpoints
