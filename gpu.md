# Installing with GPU

## PyTorch with GPU

    conda create --name zoobot38_torch python==3.8
    conda activate zoobot38_torch

Python 3.8 is required for torchvision 0.13.1.

Conda can install the same versions required by setup.py, and will also install CUDA 11.3

    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

Now you can install zoobot as normal. Pip will skip the packages above which are already installed.

As per the readme:

    pip install zoobot[pytorch]
or:
    git clone
    pip install -e ".[pytorch]"

## TensorFlow with GPU

    conda create --name zoobot37_tf python==3.7

