
Sources

https://docs.alliancecan.ca/wiki/Narval/en
https://docs.alliancecan.ca/wiki/AI_and_Machine_Learning
https://docs.alliancecan.ca/wiki/PyTorch
https://prashp.gitlab.io/post/compute-canada-tut/
https://docs.alliancecan.ca/wiki/Python

ssh walml@narval.alliancecan.ca
ssh-copy-id to avoid password in future

    module purge
    module avail

Just for venv:  
    module load python/3.9.6

    mkdir ~/envs
    virtualenv --no-download ~/envs/zoobot39_dev
    source ~/envs/zoobot39_dev/bin/activate

    avail_wheels "torch*"

Latest is currently 2.0.1 (no 2.1.0 yet)

    pip install --no-index torch==2.0.1 torchvision torchtext torchaudio
    pip install --no-index pytorch_lightning wandb

Storage under /home/user is not ideal, 50gb space. Use /project/def-bovy/walml (1TB space).
Can transfer data via rsync login node.

Move ssh key for easy login (run on LOCAL desktop)

    ssh-copy-id walml@narval.alliancecan.ca

Make new pub id key for github (back on cluster)

    ssh-keygen -t rsa -b 4096
    cat ~/.ssh/id_rsa.pub
and add to [Github](https://github.com/settings/keys) as normal

Set  up repos


    cd /project/def-bovy/walml

(I made a .bashrc alias, export PROJECT=/project/def-bovy/walml)

pip install --no-index -r zoobot/only_for_me/narval/requirements.txt

and my own cloned repos
pip install --no-deps -e galaxy-datasets
pip install --no-deps -e zoobot

Run training

sbatch only_for_me/narval/finetune.sh

Works with simple images on multi-GPU, single node


https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_2.html#
https://pytorch.org/docs/stable/elastic/run.html#environment-variables
https://github.com/webdataset/webdataset/issues/250
https://github.com/webdataset/webdataset-lightning/blob/main/train.py
https://lightning.ai/forums/t/multi-gpu-multi-node-training-with-webdataset/2300
https://webdataset.github.io/webdataset/multinode/
https://webdataset.github.io/webdataset/creating/