#!/bin/bash
#SBATCH --job-name=hsctree
#SBATCH --output=%x_%A.log 
#SBATCH --mem=0                                     # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24
pwd; hostname; date

nvidia-smi

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

$PYTHON $ZOOBOT_DIR/zoobot/pytorch/examples/finetuning/finetune_counts_full_tree.py