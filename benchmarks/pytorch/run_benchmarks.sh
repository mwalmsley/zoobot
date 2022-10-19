#!/bin/bash
# run with ./benchmarks/pytorch/run_benchmarks.sh
# inspired by https://www.osc.edu/resources/getting_started/howto/howto_submit_multiple_jobs_using_parameters

TRAIN_JOB=/share/nas2/walml/repos/zoobot/benchmarks/pytorch/run_decals_dr5_benchmark.sh

# minimal hardware - 1 gpu, no mixed precision
# (not specifying a string will default to not doing it)
sbatch --job-name=dr5_py_min --export=GPUS=1 $TRAIN_JOB

# otherwise full hardware (standard setup) - 2 gpus, mixed precision
sbatch --job-name=dr5_py_gr --export=MIXED_PRECISION_STRING=--mixed-precision,GPUS=2 $TRAIN_JOB
sbatch --job-name=dr5_py_co --export=MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,COLOR_STRING=--color $TRAIN_JOB

echo 'Jobs submitted'

# TODO add other architectures

# some other possible configurations, testing other architectures:

# ARCHITECTURE='resnet_detectron'
# BATCH_SIZE=256
# GPUS=2
# mixed precision causes rare nan errors - not recommended!
# TODO need to update to ignore stochastic_depth_prob arg

# ARCHITECTURE='resnet_torchvision'
# BATCH_SIZE=256
# GPUS=2
# # mixed precision causes rare nan errors - not recommended!
# # only supports color (so you must add --color)
# TODO need to update to ignore stochastic_depth_prob arg

# be sure to add _color if appropriate