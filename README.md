# Zoobot

[![Downloads](https://pepy.tech/badge/zoobot)](https://pepy.tech/project/zoobot)
[![Documentation Status](https://readthedocs.org/projects/zoobot/badge/?version=latest)](https://zoobot.readthedocs.io/)
![build](https://github.com/mwalmsley/zoobot/actions/workflows/run_CI.yml/badge.svg)
![publish](https://github.com/mwalmsley/zoobot/actions/workflows/python-publish.yml/badge.svg)
[![PyPI](https://badge.fury.io/py/zoobot.svg)](https://badge.fury.io/py/zoobot)
[![DOI](https://zenodo.org/badge/343787617.svg)](https://zenodo.org/badge/latestdoi/343787617)
[![status](https://joss.theoj.org/papers/447561ee2de4709eddb704e18bee846f/status.svg)](https://joss.theoj.org/papers/447561ee2de4709eddb704e18bee846f)
<a href="https://ascl.net/2203.027"><img src="https://img.shields.io/badge/ascl-2203.027-blue.svg?colorB=262255" alt="ascl:2203.027" /></a>

---
### :tada: Zoobot 2.0 is now available. Bigger and better models with streamlined finetuning. [Blog](https://walmsley.dev/posts/zoobot-scaling-laws), [paper](https://arxiv.org/abs/2404.02973) :tada:
---

Zoobot classifies galaxy morphology with deep learning.
<!-- At Galaxy Zoo, we use Zoobot to help our volunteers classify the galaxies in all our recent catalogues: GZ DECaLS, GZ DESI, GZ Rings and GZ Cosmic Dawn. -->

Zoobot is trained using millions of answers by Galaxy Zoo volunteers. This code will let you **retrain** Zoobot to accurately solve your own prediction task.

- [Install](#installation)
- [Quickstart](#quickstart)
- [Worked Examples](#worked-examples)
- [Pretrained Weights](https://zoobot.readthedocs.io/en/latest/pretrained_models.html)
- [Datasets](https://www.github.com/mwalmsley/galaxy-datasets)
- [Documentation](https://zoobot.readthedocs.io/) (for understanding/reference)
- [Mailing List](https://groups.google.com/g/zoobot) (for updates)

## Installation

<a name="installation"></a>

You can retrain Zoobot in the cloud with a free GPU using this [Google Colab notebook](https://colab.research.google.com/drive/1A_-M3Sz5maQmyfW2A7rEu-g_Zi0RMGz5?usp=sharing). To install locally, keep reading.

Download the code using git:

    git clone git@github.com:mwalmsley/zoobot.git

And then pick one of the three commands below to install Zoobot and PyTorch:

    # Zoobot with PyTorch and a GPU. Requires CUDA 12.1 (or CUDA 11.8, if you use `_cu118` instead)
    pip install -e "zoobot[pytorch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121

    # OR Zoobot with PyTorch and no GPU
    pip install -e "zoobot[pytorch-cpu]" --extra-index-url https://download.pytorch.org/whl/cpu

    # OR Zoobot with PyTorch on Mac with M1 chip
    pip install -e "zoobot[pytorch-m1]"

This installs the downloaded Zoobot code using pip [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) so you can easily change the code locally. Zoobot is also available directly from pip (`pip install zoobot[option]`). Only use this if you are sure you won't be making changes to Zoobot itself. For Google Colab, use `pip install zoobot[pytorch_colab]`

To use a GPU, you must *already* have CUDA installed and matching the versions above.
I share my install steps [here](#install_cuda). GPUs are optional - Zoobot will run retrain fine on CPU, just slower.

## Quickstart

<a name="quickstart"></a>

The [Colab notebook](https://colab.research.google.com/drive/1A_-M3Sz5maQmyfW2A7rEu-g_Zi0RMGz5?usp=sharing) is the quickest way to get started. Alternatively, the minimal example below illustrates how Zoobot works.

Let's say you want to find ringed galaxies and you have a small labelled dataset of 500 ringed or not-ringed galaxies. You can retrain Zoobot to find rings like so:

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

# load trained Zoobot model
model = finetune.FinetuneableZoobotClassifier(checkpoint_loc, num_classes=2)  

# retrain to find rings
trainer = finetune.get_trainer(save_dir)
trainer.fit(model, datamodule)
```

Then you can make predict if new galaxies have rings:

```python
from zoobot.pytorch.predictions import predict_on_catalog

# csv with 'file_loc' column (path to image). Zoobot will predict the labels.
unlabelled_df = pd.read_csv('/your/path/some_unlabelled_galaxies.csv')

predict_on_catalog.predict(
    unlabelled_df,
    model,
    label_cols=['ring'],  # only used for 
    save_loc='/your/path/finetuned_predictions.csv'
)
```

Zoobot includes many guides and working examples - see the [Getting Started](#getting-started) section below.

## Getting Started

<a name="getting_started"></a>

I suggest starting with the [Colab notebook](https://colab.research.google.com/drive/1A_-M3Sz5maQmyfW2A7rEu-g_Zi0RMGz5?usp=sharing) or the worked examples below, which you can copy and adapt.

For context and explanation, see the [documentation](https://zoobot.readthedocs.io/).

Pretrained models are listed [here](https://zoobot.readthedocs.io/en/latest/pretrained_models.html) and available on [HuggingFace](https://huggingface.co/collections/mwalmsley/zoobot-encoders-65fa14ae92911b173712b874)

### Worked Examples

<a name="worked_examples"></a>

- [pytorch/examples/finetuning/finetune_binary_classification.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/finetuning/finetune_binary_classification.py)
- [pytorch/examples/finetuning/finetune_counts_full_tree.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/finetuning/finetune_counts_full_tree.py)
- [pytorch/examples/representations/get_representations.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/representations/get_representations.py)
- [pytorch/examples/train_model_on_catalog.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/train_model_on_catalog.py) (only necessary to train from scratch)

There is more explanation and an API reference on the [docs](https://zoobot.readthedocs.io/).


### (Optional) Install PyTorch with CUDA

<a name="install_cuda"></a>

*If you're not using a GPU, skip this step. Use the pytorch-cpu option in the section below.*

Install PyTorch 2.1.0 and compatible CUDA drivers. I highly recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to do this. Conda will handle both creating a new virtual environment (`conda create`) and installing CUDA (`cudatoolkit`, `cudnn`)

CUDA 12.1 for PyTorch 2.1.0:

    conda create --name zoobot39_torch python==3.9
    conda activate zoobot39_torch
    conda install -c conda-forge cudatoolkit=12.1

### Recent release features (v2.0.0)

- New pretrained architectures: ConvNeXT, EfficientNetV2, MaxViT, and more. Each in several sizes.
- Reworked finetuning procedure. All these architectures are finetuneable through a common method.
- Reworked finetuning options. Batch norm finetuning removed. Cosine schedule option added.
- Reworked finetuning saving/loading. Auto-downloads encoder from HuggingFace.
- Now supports regression finetuning (as well as multi-class and binary). See `pytorch/examples/finetuning`
- Updated `timm` to 0.9.10, allowing latest model architectures. Previously downloaded checkpoints may not load correctly!
- (internal until published) GZ Evo v2 now includes Cosmic Dawn (HSC H2O). Significant performance improvement on HSC finetuning. Also now includes GZ UKIDSS (dragged from our archives).
- Updated `pytorch` to `2.1.0`
- Added support for webdatasets (only recommended for large-scale distributed training)
- Improved per-question logging when training from scratch
- Added option to compile encoder for max speed (not recommended for finetuning, only for pretraining).
- Deprecates TensorFlow. The CS research community focuses on PyTorch and new frameworks like JAX.

Contributions are very welcome and will be credited in any future work. Please get in touch! See [CONTRIBUTING.md](https://github.com/mwalmsley/zoobot/blob/main/benchmarks) for more.

### Benchmarks and Replication - Training from Scratch

The [benchmarks](https://github.com/mwalmsley/zoobot/blob/main/benchmarks) folder contains slurm and Python scripts to train Zoobot 1.0 from scratch. 

Training Zoobot using the GZ DECaLS dataset option will create models very similar to those used for the GZ DECaLS catalogue and shared with the early versions of this repo. The GZ DESI Zoobot model is trained on additional data (GZD-1, GZD-2), as the GZ Evo Zoobot model (GZD-1/2/5, Hubble, Candels, GZ2).

*Pretraining is becoming increasingly complex and is now partially refactored out to a separate repository. We are gradually migrating this `zoobot` repository to focus on finetuning.*

### Citing

If you use this software, or otherwise wish to cite Zoobot as a software package, please use the [JOSS paper](https://doi.org/10.21105/joss.05312):

    @article{Walmsley2023, doi = {10.21105/joss.05312}, url = {https://doi.org/10.21105/joss.05312}, year = {2023}, publisher = {The Open Journal}, volume = {8}, number = {85}, pages = {5312}, author = {Mike Walmsley and Campbell Allen and Ben Aussel and Micah Bowles and Kasia Gregorowicz and Inigo Val Slijepcevic and Chris J. Lintott and Anna M. m. Scaife and Maja Jabłońska and Kosio Karchev and Denise Lanzieri and Devina Mohan and David O’Ryan and Bharath Saiguhan and Crisel Suárez and Nicolás Guerra-Varas and Renuka Velu}, title = {Zoobot: Adaptable Deep Learning Models for Galaxy Morphology}, journal = {Journal of Open Source Software} } 

You might be interested in reading papers using Zoobot:

- [Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314,000 galaxies](https://arxiv.org/abs/2102.08414) (2022)
- [A Comparison of Deep Learning Architectures for Optical Galaxy Morphology Classification](https://arxiv.org/abs/2111.04353) (2022)
- [Practical Galaxy Morphology Tools from Deep Supervised Representation Learning](https://arxiv.org/abs/2110.12735) (2022)
- [Towards Foundation Models for Galaxy Morphology](https://arxiv.org/abs/2206.11927) (2022)
- [Harnessing the Hubble Space Telescope Archives: A Catalogue of 21,926 Interacting Galaxies](https://arxiv.org/abs/2303.00366) (2023)
- [Galaxy Zoo DESI: Detailed morphology measurements for 8.7M galaxies in the DESI Legacy Imaging Surveys](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stad2919/7283169?login=false) (2023)
- [Galaxy mergers in Subaru HSC-SSP: A deep representation learning approach for identification, and the role of environment on merger incidence](https://doi.org/10.1051/0004-6361/202346743) (2023)
- [Astronomaly at Scale: Searching for Anomalies Amongst 4 Million Galaxies](https://arxiv.org/abs/2309.08660) (2023, submitted)
- [Transfer learning for galaxy feature detection: Finding Giant Star-forming Clumps in low redshift galaxies using Faster R-CNN](https://arxiv.org/abs/2312.03503) (2023)
- [Euclid preparation. Measuring detailed galaxy morphologies for Euclid with Machine Learning](https://arxiv.org/abs/2402.10187) (2024, submitted)

Many other works use Zoobot indirectly via the [Galaxy Zoo DECaLS](https://arxiv.org/abs/2102.08414) catalog (and now via the new [Galaxy Zoo DESI](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stad2919/7283169?login=false) catalog).
