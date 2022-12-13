# Zoobot

[![Documentation Status](https://readthedocs.org/projects/zoobot/badge/?version=latest)](https://zoobot.readthedocs.io/)
[![Build Status](https://app.travis-ci.com/mwalmsley/zoobot.svg?branch=main)](https://app.travis-ci.com/mwalmsley/zoobot)
[![DOI](https://zenodo.org/badge/343787617.svg)](https://zenodo.org/badge/latestdoi/343787617)
<a href="https://ascl.net/2203.027"><img src="https://img.shields.io/badge/ascl-2203.027-blue.svg?colorB=262255" alt="ascl:2203.027" /></a>

Zoobot classifies galaxy morphology with deep learning. At Galaxy Zoo, we use Zoobot to help our volunteers classify the galaxies in all our recent catalogues: GZ DECaLS, GZ DESI, GZ Rings and GZ Cosmic Dawn.

Zoobot is trained using millions of answers by Galaxy Zoo volunteers. This code will let you **retrain** Zoobot to accurately solve your own classification task.

For example, let's say you want to find ringed galaxies and you have a small labelled dataset of 500 ringed or not-ringed galaxies. You can retrain Zoobot to find rings like so:

```python

    import pandas as pd
    from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
    from zoobot.pytorch.training import finetune

    # csv with 'ring' column (0 or 1) and 'file_loc' column (path to image)
    labelled_df = pd.read_csv('/your/path/some_labelled_galaxies.csv')

    datamodule = GalaxyDataModule(
      label_cols=['ring'],
      catalog=labelled_df,
      batch_size=32
    )

    config = {
        'finetune': {
            'label_dim': 2,
            'label_mode': 'classification',
            'n_epochs': 100
        }
    }

    # load trained Zoobot model
    encoder = finetune.load_encoder(checkpoint_loc)  

    # retrain to find rings
    _, finetuned_model = finetune.run_finetuning(config, encoder, datamodule, save_dir, logger=None)
```

Then you can make predict if new galaxies have rings:

```python
    from zoobot.pytorch.predictions import predict_on_catalog

    # csv with 'file_loc' column (path to image). Zoobot will predict the labels.
    unlabelled_df = pd.read_csv('/your/path/some_unlabelled_galaxies.csv')

    predict_on_catalog.predict(
      unlabelled_df,
      finetuned_model,
      label_cols=['ring'],  # only used for 
      save_loc='/your/path/finetuned_predictions.csv'
    )
```

Zoobot includes many guides and working examples - see the [Getting Started](#getting-started) section.

You can retrain Zoobot in the cloud with a free GPU using this [Google Colab notebook](https://colab.research.google.com/drive/1miKj3HVmt7NP6t7xnxaz7V4fFquwucW2?usp=sharing). To install locally, keep reading.

## Installation

To use a GPU to make retraining fast, you will first need to separately install either PyTorch (recommended) or Tensorflow, and compatible CUDA drivers.
Next, or if you don't need a GPU, install Zoobot itself.

### (Optional) Install PyTorch or TensorFlow, with CUDA

*If you're not using a GPU, skip this step. Use the pytorch_cpu or tensorflow_cpu options in the section below.*

Install PyTorch 1.12.1 or Tensorflow 2.10.0 and compatible CUDA drivers. I highly recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to do this. Conda will handle both creating a new virtual environment (`conda create`) and installing CUDA (`cudatoolkit`, `cudnn`)

CUDA 11.3 for PyTorch:

    conda create --name zoobot38_torch python==3.8
    conda activate zoobot38_torch
    conda install cudatoolkit=10.2

CUDA 11.2 and CUDNN 8.1 for TensorFlow 2.10.0:

    conda create --name zoobot38_tf python==3.8
    conda activate zoobot38_tf
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/  # add this environment variable

### (Required) Install Zoobot Itself

Download the code using git:

    git clone git@github.com:mwalmsley/zoobot.git

And then pick one of the three commands below to install Zoobot and either PyTorch or TensorFlow

    # Zoobot with PyTorch and a GPU. Requires CUDA 11.3.
    pip install -e "zoobot[pytorch_cu113]" --extra-index-url https://download.pytorch.org/whl/cu113

    # OR Zoobot with PyTorch and no GPU
    pip install -e "zoobot[pytorch_cpu]" --extra-index-url https://download.pytorch.org/whl/cpu

    # OR Zoobot with TensorFlow. Works with and without a GPU, but if you have a GPU, you need CUDA 11.2.
    pip install -e "zoobot[tensorflow]


And then install the downloaded Zoobot code using pip [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)...


To use a GPU, you must have already installed PyTorch or TensorFlow with a compatible CUDA version. Follow the steps above.

Zoobot is also available directly from pip (`pip install zoobot[option]`). Only use this if you are sure you won't be making changes to Zoobot itself. 

<!-- ### Direct Use

But if you won't be making any changes to Zoobot itself (e.g. you just want to apply it, or you're in a production environment), you can simply install directly from pip:

    pip install zoobot[pytorch_cpu]  # pytorch dependencies
    pip install zoobot[tensorflow_cpu]  # tensorflow dependencies -->

## Getting Started

I suggest starting with the worked examples 

To get started, see the [documentation](https://zoobot.readthedocs.io/). For pretrained model weights, precalculated representations, catalogues, and so forth, see the [data notes](https://zoobot.readthedocs.io/data_notes.html) in particular.

I also include some working examples for you to copy and adapt.

PyTorch:
- [pytorch/examples/finetuning/finetune_binary_classification.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/finetuning/finetune_binary_classification.py)
- [pytorch/examples/train_model_on_catalog.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/train_model_on_catalog.py) (only necessary to train from scratch)


TensorFlow:

- [tensorflow/examples/train_model_on_catalog.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/train_model_on_catalog.py) (only necessary to train from scratch)
- [tensorflow/examples/make_predictions.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/make_predictions.py)
- [tensorflow/examples/finetune_minimal.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/finetune_minimal.py)
- [tensorflow/examples/finetune_advanced.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/finetune_advanced.py)


I also include some examples which record how the models in W+22a (the GZ DECaLS data release) were trained:
- [replication/tensorflow/train_model_on_decals_dr5_splits.py](https://github.com/mwalmsley/zoobot/blob/main/replication/tensorflow/train_model_on_decals_dr5_splits.py)
- [replication/pytorch/train_model_on_decals_dr5_splits.py](https://github.com/mwalmsley/zoobot/blob/main/replication/pytorch/train_model_on_decals_dr5_splits.py)

There's also the [gz_decals_data_release_analysis_demo.ipynb](https://github.com/mwalmsley/zoobot/blob/main/gz_decals_data_release_analysis_demo.ipynb), which describes Zoobot's statistical predictions. When trained from scratch, it predicts the parameters for distributions, not simple class labels!

### Latest features (v1.0.0)

v1.0.0 is recognises that most of the complexity in this repo is training Zoobot from scratch, but most non-GZ users will probably simply want to load the pretrained Zoobot and finetune it on their data.

- Adds new finetuning interface (`finetune.run_finetuning()`), examples.
- Refocuses docs on finetuning rather than training from scratch.
- Rework installation process to separate CUDA from Zoobot (simpler, easier)
- Better wandb logging throughout, to monitor training
- Remove need to make TFRecords. Now TF directly uses images.
- Refactor out augmentations and datasets to `galaxy-datasets` repo. TF and Torch now use identical augmentations (via albumentations).
- Many small quality-of-life improvements

Contributions are welcome and will be credited in any future work.

### Benchmarks and Replication - Training from Scratch

The /benchmarks folder contains slurm and Python scripts to train Zoobot from scratch on GZD-5 labels. We use these scripts to make sure new code versions work well, and that TensorFlow and PyTorch achieve similar performance.

The resulting Zoobot model is very similar to those used for the GZ DECaLS catalogue and shared with the early versions of this repo. The GZ DESI Zoobot model is trained on additional data (GZD-1, GZD-2).

### Citing

If you use this repo for your research, please cite [the paper](https://arxiv.org/abs/2102.08414) and the [code](https://doi.org/10.5281/zenodo.6483175) (via Zenodo).
