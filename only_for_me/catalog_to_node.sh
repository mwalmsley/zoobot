#!/bin/bash
#SBATCH --job-name=cmp_dr12                    # Job name
#SBATCH --output=cmp_dr12_%A.log 
#SBATCH --mem=0
#SBATCH --cpus-per-task=24                              # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00
#SBATCH --constraint=A100    
#SBATCH --exclusive   # only one task per node
#SBATCH --nodelist compute-0-3

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

df -h

$PYTHON /share/nas2/walml/repos/zoobot/only_for_me/catalog_to_node.py
