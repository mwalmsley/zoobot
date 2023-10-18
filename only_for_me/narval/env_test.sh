#!/bin/bash
# SBATCH --mem=16G
# SBATCH --nodes=1
# SBATCH --time=0:5:0  
# SBATCH --ntasks-per-node=4

echo "$now"

module load StdEnv/2020
module load python/3.9.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r /project/def-bovy/zoobot/only_for_me/narval/requirements.txt
cp -r /project/def-bovy/walml/galaxy-datasets $SLURM_TMPDIR/
cp -r /project/def-bovy/walml/zoobot $SLURM_TMPDIR/
pip install --no-deps -e $SLURM_TMPDIR/galaxy-datasets
pip install --no-deps -e $SLURM_TMPDIR/zoobot

echo "$now"