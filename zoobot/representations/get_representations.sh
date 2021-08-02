#!/bin/bash
#SBATCH --job-name=color                          # Job name
#SBATCH --mem=32gb                                      # Job memory request
#SBATCH -w compute-0-0                                  # use this gpu node
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --output=reformat-features-%j.log                      # Standard output and error log
pwd; hostname; date

REPO_LOC=/share/nas/walml/repos/astronomaly

# make predictions via gpu job
# /share/nas/walml/miniconda3/envs/zoobot/bin/python /share/nas/walml/repos/zoobot/zoobot/representations/get_representations.py

# reformat/concat predictions
/share/nas/walml/miniconda3/envs/zoobot/bin/python /share/nas/walml/repos/zoobot/zoobot/predictions/reformat_predictions.py

# compress representations
/share/nas/walml/miniconda3/envs/zoobot/bin/python /share/nas/walml/repos/zoobot/zoobot/representations/compress_representations.py
