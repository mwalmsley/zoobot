#!/bin/bash
# Job name
#SBATCH --output=%x_%A.log                                 # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 

# # multi-node mode (new, specific to my cluster, may hang)
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-node=2
# #SBATCH --mem=25gb
# #SBATCH --cpus-per-task=12
# GPUS=2
# NODES=2   

# single node mode (more reliable)
#SBATCH --mem=0 
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24
NODES=1

pwd; hostname; date

nvidia-smi

export NCCL_DEBUG=INFO
export PYTORCH_KERNEL_CACHE_PATH=/share/nas2/walml/.cache/torch/kernels

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

if  [ "$DATASET" = "gz_decals_dr5" ]; 
then
    DATA_DIR=/share/nas2/walml/repos/_data/gz_decals
    RESULTS_DIR=/share/nas2/walml/repos/gz-decals-classifiers/results
    EXPERIMENT_DIR=$RESULTS_DIR/benchmarks/pytorch/dr5
fi

if [ "$DATASET" = "gz_evo" ]; 
then
    DATA_DIR=/share/nas2/walml/repos/_data
    RESULTS_DIR=/share/nas2/walml/repos/gz-decals-classifiers/results
    EXPERIMENT_DIR=$RESULTS_DIR/benchmarks/pytorch/evo
fi


echo $PYTHON $ZOOBOT_DIR/benchmarks/pytorch/train_model_on_benchmark_dataset.py \
    --save-dir $EXPERIMENT_DIR/$SLURM_JOB_NAME \
    --data-dir $DATA_DIR \
    --dataset $DATASET \
    --architecture $ARCHITECTURE \
    --resize-after-crop $RESIZE_AFTER_CROP \
    --batch-size $BATCH_SIZE \
    --gpus $GPUS \
    --nodes $NODES \
    --wandb \
    --seed $SEED \
    $COLOR_STRING \
    $MIXED_PRECISION_STRING \
    $DEBUG_STRING

srun $PYTHON $ZOOBOT_DIR/benchmarks/pytorch/train_model_on_benchmark_dataset.py \
    --save-dir $EXPERIMENT_DIR/$SLURM_JOB_NAME \
    --data-dir $DATA_DIR \
    --dataset $DATASET \
    --architecture $ARCHITECTURE \
    --resize-after-crop $RESIZE_AFTER_CROP \
    --batch-size $BATCH_SIZE \
    --gpus $GPUS \
    --nodes $NODES \
    --wandb \
    --seed $SEED \
    $COLOR_STRING \
    $MIXED_PRECISION_STRING \
    $DEBUG_STRING