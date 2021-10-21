# Zoobot

[![Documentation Status](https://readthedocs.org/projects/zoobot/badge/?version=latest)](https://zoobot.readthedocs.io/en/latest/?badge=latest)

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

Install using git and pip:

    # I recommend using a virtual environment, see below
    git clone git@github.com:mwalmsley/zoobot.git
    pip install -r zoobot/requirements.txt
    pip install -e zoobot

I recommend installing in a virtual environment like anaconda.  For example, `conda create --name zoobot python=3.7`, then `conda activate zoobot`.
Do not install directly with anaconda itself (e.g. `conda install tensorflow`). Anaconda currently installs tensorflow 2.0.0, which is too old for the latest features used here.
Use pip instead, as above.

The `main` branch is for stable-ish releases. The `dev` branch includes the shiniest features but may change at any time.

To get started, see the [documentation](https://zoobot.readthedocs.io/).

I also include some working examples for you to copy and adapt:

- [train_model.py](https://github.com/mwalmsley/zoobot/blob/main/train_model.py)
- [make_predictions.py](https://github.com/mwalmsley/zoobot/blob/main/make_predictions.py)
- [finetune_minimal.py](https://github.com/mwalmsley/zoobot/blob/main/finetune_minimal.py)
- [finetune_advanced.py](https://github.com/mwalmsley/zoobot/blob/main/finetune_advanced.py)

Latest cool features on dev branch (June 2021):

- Multi-GPU distributed training
- Support for Weights and Biases (wandb)
- Worked examples for custom representations

Contributions are welcome and will be credited in any future work.

If you use this repo for your research, please cite [the paper](https://arxiv.org/abs/2102.08414).
