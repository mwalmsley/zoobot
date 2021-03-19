import logging

import numpy as np
import tensorflow as tf

from zoobot.estimators import efficientnet_standard, efficientnet_custom, custom_layers
from zoobot.training import losses


class CustomSequential(tf.keras.Sequential):

    def __init__(self):
        """Will this override?
        """
        super().__init__()
        self.step = 0

    def call(self, x, training):
        "How about this?"
        tf.summary.image('model_input', x, step=self.step)
        # tf.summary.image('model_input', x, step=0)
        return super().call(x, training)


def add_preprocessing_layers(model, crop_size, resize_size):
    if crop_size < resize_size:
        logging.warning('Crop size {} < final size {}, losing resolution'.format(
            crop_size, resize_size))

    resize = True
    if np.abs(crop_size - resize_size) < 10:
        logging.warning(
            'Crop size and final size are similar: skipping resizing and cropping directly to resize_size (ignoring crop_size)')
        resize = False
        crop_size = resize_size

    model.add(custom_layers.PermaRandomRotation(np.pi, fill_mode='reflect'))
    model.add(custom_layers.PermaRandomFlip())
    model.add(custom_layers.PermaRandomCrop(crop_size, crop_size))
    if resize:
        logging.info('Using resizing, to {}'.format(resize_size))
        model.add(tf.keras.layers.experimental.preprocessing.Resizing(
            resize_size, resize_size, interpolation='bilinear'
        ))


def get_model(output_dim, input_size, crop_size, resize_size, weights_loc=None, include_top=True, expect_partial=False):

    # dropout_rate = 0.3
    # drop_connect_rate = 0.2  # gets scaled by num blocks, 0.6ish = 1

    logging.info('Input size {}, crop size {}, final size {}'.format(
        input_size, crop_size, resize_size))

    # model = CustomSequential()  # to log the input image for debugging
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(input_size, input_size, 1)))

    add_preprocessing_layers(model, crop_size=crop_size,
                             resize_size=resize_size)  # inplace

    shape_after_preprocessing_layers = (resize_size, resize_size, 1)
    # now headless
    effnet = efficientnet_custom.define_headless_efficientnet(
        input_shape=shape_after_preprocessing_layers,
        get_effnet=efficientnet_standard.EfficientNetB0
        # further kwargs will be passed to get_effnet
        # dropout_rate=dropout_rate,
        # drop_connect_rate=drop_connect_rate
    )
    model.add(effnet)
    # model.add(tf.keras.layers.Dense(16))
    # model.add(tf.keras.layers.Dense(2))
#
    if include_top:
        assert output_dim is not None
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        efficientnet_custom.custom_top_dirichlet(model, output_dim)  # inplace
    # efficientnet.custom_top_dirichlet_reparam(model, output_dim, schema)

    # will be updated by callback
    model.step = tf.Variable(
        0, dtype=tf.int64, name='model_step', trainable=False)

    if weights_loc:
        load_weights(model, weights_loc, expect_partial=expect_partial)
    return model


# inplace
def load_weights(model, weights_loc, expect_partial=False):
    # https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    logging.info('Loading weights from {}'.format(weights_loc))
    load_status = model.load_weights(weights_loc)
    load_status.assert_nontrivial_match()
    load_status.assert_existing_objects_matched()
    if expect_partial:  # some checkpointed values won't match (the optimiser state during predictions, hopefully)
        load_status.expect_partial()


def load_model(checkpoint_dir, include_top, input_size, crop_size, resize_size, output_dim=34, expect_partial=False):
    # utility wrapper for the common task of loading a pretrained model from scratch
    # resize_size and output_dim must match w/e the pretrained model used
    # input_size and crop_size can vary, but be careful
    model = get_model(
        output_dim=output_dim,
        input_size=input_size,
        crop_size=crop_size,
        resize_size=resize_size,
        include_top=include_top
    )
    load_weights(model, checkpoint_dir, expect_partial=expect_partial)
    return model
