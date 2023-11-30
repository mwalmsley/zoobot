#!/bin/bash
#SBATCH --time=03:30:0  
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
#     --color --wandb --mixed-precision 

srun $PYTHON $REPO_DIR/only_for_me/narval/train.py \
    --save-dir $REPO_DIR/only_for_me/narval/desi_300px_efficientnet_b0_4gpu_terr \
    --batch-size 256 \
    --architecture efficientnet_b0 \
    --terrestrial \
    --gpus 4 \
    --num-workers 10 \
    --color --wandb --mixed-precision --compile-encoder


# batch sizes
# v100
# efficientnet_b0 256
# maxvit_tiny_rw_224 64
# tf_efficientnetv2_b0 256 - 50.55%, might squeeze x2
# tf_efficientnetv2_s 64? TODO
# pit_xs_224 512
# pit_s_224 256
# maxvit_small_224 32
# vit_small_patch16_224 32? 17%, too small. TODO 128 (but pure vit is probably not great)
# vit_tiny_patch16_224 64?
# maxvit_rmlp_small_rw_224 64 (97% allocated and very good efficiency)
# https://huggingface.co/timm/convnextv2_nano.fcmae TODO with MAE
# convnext_nano
# convnext_tiny - 128
# efficientnet_b2 - 32% at 64, can do 128
# convnext_small 64 - 49.25%, MAYBE 128 
# efficientnet_b4 - 48% at 64, could maybe do 128
# efficientnet_b5 - 64. remember it expects bigger images tho, may not work great
# maxvit_rmlp_base_rw_224 - 32 (95%). Now scaling at 16 gpus

# srun $PYTHON $REPO_DIR/only_for_me/narval/train.py \
#     --save-dir $REPO_DIR/only_for_me/narval/desi_300px_maxvit_rmlp_base_rw_224_4gpu_w005 \
#     --batch-size 32 \
#     --gpus 4 \
#     --nodes 1 \
#     --num-workers 5 \
#     --weight-decay 0.05 \
#     --architecture maxvit_rmlp_base_rw_224 \
#     --color --wandb --mixed-precision
    
    #  --compile-encoder

    # maxvit_small_tf_224 \
