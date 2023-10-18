#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:1:0      

#### SBATCH --gres=gpu:a100:1
#### SBATCH --mail-user=<youremail@gmail.com>
#### SBATCH --mail-type=ALL

PYTHON=/home/envs/zoobot39_dev/bin/python

$PYTHON /project/def-bovy/walml/repos/zoobot/only_for_me/narval/finetune.py
