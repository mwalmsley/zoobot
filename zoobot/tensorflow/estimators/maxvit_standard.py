import logging

# Normal packages
import numpy as np

# ML related.
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_datasets as tfds

# MaxVit Packages
import maxvit.models.hparams as hparams
import maxvit.models.maxvit as layers

# building our transformer!
def MaxViTModel(
    maxvit_model, 
    image_size, 
    batch_size,
    epochs
    ):

    config = hparams.lookup(maxvit_model)

    config.train.image_size = image_size
    config.train.epochs = epochs
    config.train.batch_size = batch_size

    config.eval.image_size = image_size
    config.eval.batch_size = batch_size

    model = layers.MaxViT(config.model)

    return model

def get_maxvit_model(
    input_shape=None,
    maxvit_model = None,
    use_imagenet_weights=False,
    batch_size,
    epochs
):
    return MaxViTModel(
        maxvit_model,
        input_shape,
        batch_size,
        epochs
    )

def MaxViTTinyModel(
):
    maxvit_model = 'MaxViTTiny'
    return maxvit_model
    