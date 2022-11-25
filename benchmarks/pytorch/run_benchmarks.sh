#!/bin/bash
# run with ./benchmarks/pytorch/run_benchmarks.sh
# inspired by https://www.osc.edu/resources/getting_started/howto/howto_submit_multiple_jobs_using_parameters

TRAIN_JOB=/share/nas2/walml/repos/zoobot/benchmarks/pytorch/run_decals_dr5_benchmark.sh
SEED=$RANDOM

# debug mode
# sbatch --job-name=dr5_py_debug_$SEED --export=GPUS=1,SEED=$SEED,DEBUG_STRING='--debug' $TRAIN_JOB

# minimal hardware - 1 gpu, no mixed precision
# (not specifying a string will default to not doing it)
# sbatch --job-name=dr5_py_min_b512_lrdef_seeded_$SEED --export=MIXED_PRECISION_STRING=--mixed-precision,GPUS=1,SEED=$SEED $TRAIN_JOB

# otherwise full hardware (standard setup) - 2 gpus, mixed precision
sbatch --job-name=dr5_py_gr_b512_lrdef_seeded_2pgu_$SEED --export=MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
# sbatch --job-name=dr5_py_co_nobn_$SEED --export=MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,COLOR_STRING=--color $TRAIN_JOB

echo 'PyTorch jobs submitted'
