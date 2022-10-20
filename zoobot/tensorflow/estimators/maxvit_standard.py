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
def MaxViTModel(maxvit_model,input_shape):

    if maxvit_model == 'MaxViTTinyiest':
        maxvit_name = 'MaxViTTiny'
        config = hparams.lookup(maxvit_name)

        config.train.image_size = input_shape
        config.eval.image_size = input_shape

        config.model.num_block = [1,1,2,1]
        config.model.stem_hsize = [8,8]
        config.model.hidden_size = [32, 64, 128, 256]
        config.model.num_classes = 1280  # matching both models
    elif maxvit_model == 'MaxViTTiny':
        maxvit_name = 'MaxViTTiny'
        config = hparams.lookup(maxvit_name)

        config.train.image_size = input_shape
        config.eval.image_size = input_shape
        config.model.num_classes = 1280  # matching both models
    else:
        maxvit_name = maxvit_model
        config = hparams.lookup(maxvit_name)
        config.train.image_size = input_shape
        config.eval.image_size = input_shape
        config.model.num_classes = 1280  # matching both models


    model = layers.MaxViT(config.model)

    return model


def get_maxvit_model(input_shape,
                     get_maxvit = 'MaxViTTiny',
                     use_image_weights=False):
    return MaxViTModel(maxvit_model=get_maxvit#'MaxViTTiny'
    , input_shape=input_shape)
