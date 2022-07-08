#!/bin/bash
#SBATCH --job-name=fn_adv                        # Job name
#SBATCH --mem=0                                     # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --output=finetune_advanced-%j.log                      # Standard output and error log
#SBATCH --exclusive
#SBATCH --constraint=A100
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24
pwd; hostname; date

nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

echo $LD_LIBRARY_PATH

REPO_LOC=/share/nas2/walml/repos/zoobot

# batch size of 256 works on a single A100 with effnet b0
# 512 with two A100's
/share/nas2/walml/miniconda3/envs/zoobot/bin/python /share/nas2/walml/repos/zoobot/zoobot/tensorflow/examples/finetune_advanced.py \
    --batch-size 512 \
    --epochs 50