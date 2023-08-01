#!/bin/bash
# run with ./benchmarks/pytorch/run_benchmarks.sh
# inspired by https://www.osc.edu/resources/getting_started/howto/howto_submit_multiple_jobs_using_parameters

TRAIN_JOB=/share/nas2/walml/repos/zoobot/benchmarks/pytorch/run_dataset_benchmark.sh
SEED=$RANDOM

# debug mode
# sbatch --job-name=dr5_py_debug_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,GPUS=1,SEED=$SEED,DEBUG_STRING='--debug' $TRAIN_JOB
# minimal hardware - 1 gpu, no mixed precision
# (not specifying a string will default to not doing it)
# sbatch --job-name=dr5_py_min_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,GPUS=1,SEED=$SEED $TRAIN_JOB


# GZ Evo i.e. all galaxies
# effnet, greyscale and color
# sbatch --job-name=evo_py_gr_eff_224_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=evo_py_gr_eff_300_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=300,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
sbatch --job-name=evo_py_co_eff_224_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,COLOR_STRING=--color,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=evo_py_co_eff_300_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=128,RESIZE_AFTER_CROP=300,DATASET=gz_evo,COLOR_STRING=--color,GPUS=2,SEED=$SEED $TRAIN_JOB

# and resnet18
# sbatch --job-name=evo_py_gr_res18_224_$SEED --export=ARCHITECTURE=resnet18,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=evo_py_gr_res18_300_$SEED --export=ARCHITECTURE=resnet18,BATCH_SIZE=256,RESIZE_AFTER_CROP=300,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# and resnet50
# sbatch --job-name=evo_py_gr_res50_224_$SEED --export=ARCHITECTURE=resnet50,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=evo_py_gr_res50_300_$SEED --export=ARCHITECTURE=resnet50,BATCH_SIZE=256,RESIZE_AFTER_CROP=300,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# and with max-vit tiny because hey transformers are cool

# smaller batch size due to memory
sbatch --job-name=evo_py_gr_vittiny_224_$SEED --export=ARCHITECTURE=maxvit_tiny_224,BATCH_SIZE=128,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
sbatch --job-name=evo_py_co_vittiny_224_$SEED --export=ARCHITECTURE=maxvit_tiny_224,BATCH_SIZE=128,RESIZE_AFTER_CROP=224,DATASET=gz_evo,COLOR_STRING=--color,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB

# and max-vit small (works badly)
# sbatch --job-name=evo_py_gr_vitsmall_224_$SEED --export=ARCHITECTURE=maxvit_small_224,BATCH_SIZE=64,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# and convnext (works badly)
# sbatch --job-name=evo_py_gr_$SEED --export=ARCHITECTURE=convnext_nano,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# and vit
# sbatch --job-name=evo_py_gr_vittinyp16_224_$SEED --export=ARCHITECTURE=vit_tiny_patch16_224,BATCH_SIZE=128,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB


# and in color with no mixed precision, for specific project
# sbatch --job-name=evo_py_co_res50_224_fullprec_$SEED --export=ARCHITECTURE=resnet50,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,COLOR_STRING=--color,GPUS=2,SEED=$SEED $TRAIN_JOB

# GZ Evo variants with smaller image sizes (not recommended)
# sbatch --job-name=evo_py_gr_128px_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=128,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=evo_py_c_128px_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=128,DATASET=gz_evo,COLOR_STRING=--color,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=evo_py_gr_64px_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=64,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=evo_py_c_64px_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=64,DATASET=gz_evo,COLOR_STRING=--color,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB


# GZ DECaLS DR5
# sbatch --job-name=dr5_py_gr_eff_224_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=dr5_py_co_eff_224_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,COLOR_STRING=--color,SEED=$SEED $TRAIN_JOB
# and with maxvit, tiny and small
# sbatch --job-name=dr5_py_vit_$SEED --export=ARCHITECTURE=maxvit_tiny_224,BATCH_SIZE=128,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=dr5_py_vit_$SEED --export=ARCHITECTURE=maxvit_small_224,BATCH_SIZE=64,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB

echo 'PyTorch jobs submitted'
