#!/bin/bash
#SBATCH --time=23:30:0  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:v100:4

nvidia-smi

PYTHON=/home/walml/envs/zoobot39_dev/bin/python
# source ~/envs/zoobot39_dev/bin/activate

mkdir $SLURM_TMPDIR/cache
# mkdir /tmp/cache

export NCCL_BLOCKING_WAIT=1

REPO_DIR=/project/def-bovy/walml/zoobot
# srun $PYTHON $REPO_DIR/only_for_me/narval/train.py \
#     --save-dir $REPO_DIR/only_for_me/narval/desi_300px_f128_1gpu \
#     --batch-size 256 \
#     --num-features 128 \
#     --gpus 1 \
#     --num-workers 10 \
#     --color --wandb --mixed-precision --compile-encoder

srun $PYTHON $REPO_DIR/only_for_me/narval/train.py \
    --save-dir $REPO_DIR/only_for_me/narval/desi_300px_maxvittiny_rw_224_4gpu \
    --batch-size 64 \
    --gpus 4 \
    --num-workers 10 \
    --architecture maxvit_tiny_rw_224 \
    --color --wandb --mixed-precision --compile-encoder

    # maxvit_small_tf_224 \
