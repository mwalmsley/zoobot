# Zoobot

[![Documentation Status](https://readthedocs.org/projects/zoobot/badge/?version=latest)](https://zoobot.readthedocs.io/)
[![Build Status](https://app.travis-ci.com/mwalmsley/zoobot.svg?branch=main)](https://app.travis-ci.com/mwalmsley/zoobot)
[![DOI](https://zenodo.org/badge/343787617.svg)](https://zenodo.org/badge/latestdoi/343787617)
<a href="https://ascl.net/2203.027"><img src="https://img.shields.io/badge/ascl-2203.027-blue.svg?colorB=262255" alt="ascl:2203.027" /></a>

Zoobot classifies galaxy morphology with deep learning. This code will let you:

- **Reproduce** and improve the Galaxy Zoo DECaLS automated classifications
- **Finetune** the classifier for new tasks

For example, you can train a new classifier like so:

```python
model = define_model.get_model(
    output_dim=len(schema.label_cols),  # schema defines the questions and answers
    input_size=initial_size, 
    crop_size=int(initial_size * 0.75),
    resize_size=resize_size
)

model.compile(
    loss=losses.get_multiquestion_loss(schema.question_index_groups),
    optimizer=tf.keras.optimizers.Adam()
)

training_config.train_estimator(
    model, 
    train_config,  # parameters for how to train e.g. epochs, patience
    train_dataset,
    test_dataset
)
```

You can finetune Zoobot with a free GPU using this [Google Colab notebook](https://colab.research.google.com/drive/1miKj3HVmt7NP6t7xnxaz7V4fFquwucW2?usp=sharing). To install locally, keep reading.

Download the code using git:

    # I recommend using a virtual environment, see below
    git clone git@github.com:mwalmsley/zoobot.git

And then install Zoobot using pip, specifying either the pytorch dependencies, the tensorflow dependencies, or both:

    pip install -e zoobot[pytorch]  # pytorch dependencies
    pip install -e zoobot[tensorflow]  # tensorflow dependencies
    pip install -e zoobot[pytorch,tensorflow]  # both

I recommend installing in a virtual environment like anaconda.  For example, `conda create --name zoobot python=3.7`, then `conda activate zoobot`.
Do not install directly with anaconda itself (e.g. `conda install tensorflow`) as Anaconda may install older versions.
Use pip instead, as above. Python 3.7 or greater is required.

The `main` branch is for stable-ish releases. The `dev` branch includes the shiniest features but may change at any time.

To get started, see the [documentation](https://zoobot.readthedocs.io/). For pretrained model weights, precalculated representations, catalogues, and so forth, see the [data notes](https://zoobot.readthedocs.io/data_notes.html) in particular.

I also include some working examples for you to copy and adapt:

- [tensorflow/examples/decals_dr5_to_shards.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/decals_dr5_to_shards.py) (only necessary to train from scratch)
- [tensorflow/examples/train_model_on_shards.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/train_model_on_shards.py) (only necessary to train from scratch)
- [tensorflow/examples/make_predictions.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/make_predictions.py)
- [tensorflow/examples/finetune_minimal.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/finetune_minimal.py)
- [tensorflow/examples/finetune_advanced.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/finetune_advanced.py)
- [pytorch/examples/train_model_on_catalog.py](https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/train_model_on_catalog.py) (only necessary to train from scratch)

I also include some examples which record how the models in W+22a (the GZ DECaLS data release) were trained:
- [replication/tensorflow/train_model_on_decals_dr5_splits.py](https://github.com/mwalmsley/zoobot/blob/main/replication/tensorflow/train_model_on_decals_dr5_splits.py)
- [replication/pytorch/train_model_on_decals_dr5_splits.py](https://github.com/mwalmsley/zoobot/blob/main/replication/pytorch/train_model_on_decals_dr5_splits.py)

There's also the [gz_decals_data_release_analysis_demo.ipynb](https://github.com/mwalmsley/zoobot/blob/main/gz_decals_data_release_analysis_demo.ipynb), which describes Zoobot's statistical predictions. When trained from scratch, it predicts the parameters for distributions, not simple class labels!

### Latest features

- PyTorch version! Integrates with PyTorch Lightning and WandB. Multi-GPU support. Trains on jpeg images, rather than TFRecords, and does not yet have a finetuning example script.
- Train on colour (3-band) images: Add --color (American-friendly) to `train_model.py`
- Select which EfficientNet variant to train using the `get_effnet` arg in `define_model.py` - or replace with a func. returning your own architecture!
- New `predict_on_dataset.py` and `save_predictons.py` modules with useful functions for making predictions on large sets of images. Predictions are now saved to .hdf5 by default, which is much more convenient than csv for multi-forward-pass predictions. If using .hdf5, `reformat_predictions.py` is no longer needed.
- New `visualize_predictions.py`, `evaluate_model.py` and `compare_models.py` scripts for measuring model performance.
- Multi-GPU distributed training
- Support for Weights and Biases (wandb)
- Worked examples for custom representations
- [Colab notebook](https://colab.research.google.com/drive/1miKj3HVmt7NP6t7xnxaz7V4fFquwucW2?usp=sharing) for GZ predictions and fine-tuning
- Schemas (questions and answers for the decision trees) extended to include DECaLS DR1/2 and DR8, in various combinations. See `zoobot.shared.label_metadata.py`.
- Test time augmentations are now off by default but can be enabled with `--test-time-augs` on `train_model.py`
- `create_shards.py` has been refactored. Use the new example script `decals_dr5_to_shards.py` to replicate Zoobot on DECaLS, and `create_shards.py` for general creation of TFRecords from catalogs. `decals_dr5_to_shards.py` now includes train/val/test splits, which it should have had in the first place.
- `zoobot/data_utils/image_datasets.py` will optionally check if the image paths provided really exist (slightly slower, but sometimes useful). `tfrecord_datasets` and `image_datasets` now serve equivalent purposes.

Contributions are welcome and will be credited in any future work.

### Replication

For replication of the GZ DECaLS classifier see /replicate. This contains slurm scripts to:
- Create training TFRecords equivalent to those used to train the published classifier
- Train the classifier itself (by calling `zoobot/tensorflow/examples/train_model.py`)

### Citing

If you use this repo for your research, please cite [the paper](https://arxiv.org/abs/2102.08414) and the [code](https://doi.org/10.5281/zenodo.6483175) (via Zenodo).
