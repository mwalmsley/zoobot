#!/bin/bash
# Job name
#SBATCH --output=%x_%A.log 
#SBATCH --mem=0                                     # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24
pwd; hostname; date

nvidia-smi

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python
DATA_DIR=/share/nas2/walml/repos/_data/decals_dr5

RESULTS_DIR=/share/nas2/walml/repos/gz-decals-classifiers/results
EXPERIMENT_DIR=$RESULTS_DIR/benchmarks/pytorch/dr5

ARCHITECTURE='efficientnet'
BATCH_SIZE=128

echo $PYTHON $ZOOBOT_DIR/benchmarks/pytorch/train_model_on_decals_dr5_splits.py \
    --save-dir $EXPERIMENT_DIR/$SLURM_JOB_NAME \
    --data-dir $DATA_DIR \
    --architecture $ARCHITECTURE \
    --resize-size 224 \
    --batch-size $BATCH_SIZE \
    --gpus $GPUS \
    $COLOR_STRING \
    $MIXED_PRECISION_STRING

$PYTHON $ZOOBOT_DIR/benchmarks/pytorch/train_model_on_decals_dr5_splits.py \
    --save-dir $EXPERIMENT_DIR/$SLURM_JOB_NAME \
    --data-dir $DATA_DIR \
    --architecture $ARCHITECTURE \
    --resize-size 224 \
    --batch-size $BATCH_SIZE \
    --gpus $GPUS \
    $COLOR_STRING \
    $MIXED_PRECISION_STRINGs