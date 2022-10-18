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
def MaxViTModel(model_name):
    config = hparams.lookup(model_name)

    model = layers.MaxViT(config.model)

    return model


def MaxViTTinyModel(
):

    return MaxViTModel(
        model_name = 'MaxViTTiny'
    )