# Installing with GPU

This assumes you are on Linux (e.g. using a university cluster). Any Linux distro. should work.

## Zoobot/PyTorch with GPU

    conda create --name zoobot38_torch python==3.8
    conda activate zoobot38_torch

Python 3.8 is required for torchvision 0.13.1.

Conda can install the same versions required by setup.py, and will also install CUDA 11.3

    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

Now you can install Zoobot.

As per the readme:

    git clone git@github.com:mwalmsley/zoobot.git
    pip install -e "zoobot[pytorch]"  # pytorch already installed above

or where you definitely won't change zoobot itself (e.g. in production at Zooniverse):

    pip install zoobot[pytorch]  # pytorch already installed above

## Zoobot/TensorFlow with GPU

    conda create --name zoobot38_tf python==3.8
    conda activate zoobot38_tf

Install CUDA:

    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

Now you can install TensorFlow and Zoobot.

As per the readme:

    git clone git@github.com:mwalmsley/zoobot.git
    pip install -e "zoobot[tensorflow]"  # includes tensorflow

or where you definitely won't change zoobot itself (e.g. in production at Zooniverse):

    pip install zoobot[tensorflow]  # includes tensorflow
