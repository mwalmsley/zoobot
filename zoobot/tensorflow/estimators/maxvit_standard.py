# imports

import logging

# Normal packages
import numpy as np

# ML related
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_datasets as tfds

# MaxVit Packages
import maxvit.models.hparams as hparams
import maxvit.models.maxvit as layers

# building our transformer!
def MaxViTModel(maxvit_model, input_shape,):

    config = hparams.lookup(maxvit_model)
    config.model.num_classes = 1280  # matching both models
    config.train.image_size = input_shape
    config.eval.image_size = input_shape

    model = layers.MaxViT(config.model)
    return model


def get_maxvit_model(input_shape,
                     get_maxvit,
                     use_image_weights=False):
    return MaxViTModel(maxvit_model=get_maxvit#'MaxViTTiny'
    , input_shape=input_shape)
