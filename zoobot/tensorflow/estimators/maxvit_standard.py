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
def MaxViTModel(maxvit_model):

    config = hparams.lookup(maxvit_model)

    model = layers.MaxViT(config.model)
    
    return model

def get_maxvit_model(
    input_shape=None,
    maxvit_model = None,
    use_imagenet_weights=False
):
    return MaxViTModel(
        maxvit_model
    )

def MaxViTTinyModel(
):
    maxvit_model = 'MaxViTTiny'
    return maxvit_model
    