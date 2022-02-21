#!/bin/bash
#SBATCH --job-name=jpeg_dr12                    # Job name
#SBATCH --output=jpeg_dr12_%A.log 
#SBATCH --mem=80gb    # high mem node 
#SBATCH --cpus-per-task=16                               # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00    
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

$PYTHON /share/nas2/walml/repos/zoobot/only_for_me/png_to_jpg.py
