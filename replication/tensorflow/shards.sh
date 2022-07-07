#!/bin/bash
#SBATCH --job-name=shards                       # Job name
#SBATCH --output=shards_%A.log 
#SBATCH --mem=32gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

# input
CATALOG_DIR=/share/nas2/walml/galaxy_zoo/decals/long_term_model_archive/prepared_catalogs/decals_dr_galahad

# output
SHARD_DIR=/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_2p5_unfiltered_n2

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

$PYTHON $ZOOBOT_DIR/create_shards.py \
    --labelled-catalog $CATALOG_DIR/labelled_catalog.csv \
    --unlabelled-catalog $CATALOG_DIR/unlabelled_catalog.csv \
    --eval-size 10000 \
    --shard-dir $SHARD_DIR \
    --img-size 300

# alternatively, you might like to make some very small shards first just to check everything works
# (you can train on these using replicate.sh as well, see the similar commented section on that script)
# (it will learn very badly but the point is to see if everything is cabled up correctly)

# $PYTHON $ZOOBOT_DIR/create_shards.py \
#     --labelled-catalog $CATALOG_DIR/labelled_catalog.csv \
#     --unlabelled-catalog $CATALOG_DIR/unlabelled_catalog.csv \
#     --shard-dir $SHARD_DIR \
#     --max-labelled 500 \
#     --max-unlabelled 300 \
#     --eval-size 100 \
#     --img-size 32
