import logging

import numpy as np
import tensorflow as tf

from zoobot.estimators import  efficientnet_standard, efficientnet_custom, custom_layers
from zoobot.training import losses


class CustomSequential(tf.keras.Sequential):

    def call(self, x, training):
        tf.summary.image('model_input', x, step=self.step)
        # tf.summary.image('model_input', x, step=0)
        return super().call(x, training)


def add_preprocessing_layers(model, crop_size, final_size):
    if crop_size < final_size:
        logging.warning('Crop size {} < final size {}, losing resolution'.format(crop_size, final_size))
    
    resize = True
    if np.abs(crop_size - final_size) < 10:
        logging.warning('Crop size and final size are similar: skipping resizing and cropping directly to final_size (ignoring crop_size)')
        resize = False
        crop_size = final_size

    model.add(custom_layers.PermaRandomRotation(np.pi, fill_mode='reflect'))
    model.add(custom_layers.PermaRandomFlip())
    model.add(custom_layers.PermaRandomCrop(
        crop_size, crop_size  # from 256, bad to the resize up again but need more zoom...
    ))
    if resize:
        logging.info('Using resizing, to {}'.format(final_size))
        model.add(tf.keras.layers.experimental.preprocessing.Resizing(
            final_size, final_size, interpolation='bilinear'
        ))


def get_model(schema, initial_size, crop_size, final_size, weights_loc=None):

    # dropout_rate = 0.3
    # drop_connect_rate = 0.2  # gets scaled by num blocks, 0.6ish = 1

    logging.info('Initial size {}, crop size {}, final size {}'.format(initial_size, crop_size, final_size))

    # model = CustomSequential()  # to log the input image for debugging
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(initial_size, initial_size, 1)))

    add_preprocessing_layers(model, crop_size=crop_size, final_size=final_size)  # inplace

    output_dim = len(schema.label_cols)

    input_shape = (final_size, final_size, 1)
    # now headless
    effnet = efficientnet_custom.EfficientNet_custom_top(
        schema=schema,
        input_shape=input_shape,
        get_effnet=efficientnet_standard.EfficientNetB0
        # further kwargs will be passed to get_effnet
        # dropout_rate=dropout_rate,
        # drop_connect_rate=drop_connect_rate
    )
    model.add(effnet)
    # model.add(tf.keras.layers.Dense(16))
    # model.add(tf.keras.layers.Dense(2))
# 
    efficientnet_custom.custom_top_dirichlet(model, output_dim, schema)  # inplace
    # efficientnet.custom_top_dirichlet_reparam(model, output_dim, schema)

    # will be updated by callback
    model.step = tf.Variable(0, dtype=tf.int64, name='model_step', trainable=False)


    multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
    loss = lambda x, y: multiquestion_loss(x, y)

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam()
        # metrics=abs_metrics + q_loss_metrics + a_loss_metrics
    )

    # print(model)
    # exit()
    model.summary()
    # model.layers[-1].summary()

    if weights_loc:
        logging.info('Loading weights from {}'.format(weights_loc))
        load_status = model.load_weights(weights_loc)  # inplace
        # may silently fail without these
        load_status.assert_nontrivial_match()
        load_status.assert_existing_objects_matched()

    return model