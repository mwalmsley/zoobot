import logging

import numpy as np
import tensorflow as tf

from zoobot.tensorflow.estimators import efficientnet_standard, efficientnet_custom, custom_layers


# class CustomSequential(tf.keras.Sequential):

#     def __init__(self):
#         super().__init__()
#         self.step = 0

#     def call(self, x, training):
#         """
#         Override tf.keras.Sequential to optionally save image data to tensorboard.
#         Slow but useful for debugging.
#         Not used by default (see get_model). I suggest only uncommenting when you want to debug.
#         """
#         tf.summary.image('model_input', x, step=self.step)
#         tf.summary.histogram('model_input', x, step=self.step)
#         return super().call(x, training)


def get_augmentation_layers(crop_size, always_augment=False):
    """
    
    The following augmentations are applied, in order:
        - Random rotation (aliased)
        - Random flip (horizontal and/or vertical)
        - Random crop (not centered) down to ``(crop_size, crop_size)``

    Designed for use with tf Functional API

    TODO I would prefer to refactor this so augmentations are separate from the model, as with pytorch.
    But it's not a high priority change.

    Recent changes:
    - resize_size option removed. Do in preprocessing step instead.
    - Switching to albumentations (for consistent API with PyTorch)

    Args:
        crop_size (int): desired length of image after random crop (assumed square)
        always_augment (bool, optional): If True, augmentations also happen at test time. Defaults to False.

    Returns:
        (tf.keras.Sequential): applying augmentations with e.g. x_aug = model(x)
    """

    model = tf.keras.Sequential(name='augmentations')

    if always_augment:
        rotation_layer = custom_layers.PermaRandomRotation
        flip_layer = custom_layers.PermaRandomFlip
        crop_layer = custom_layers.PermaRandomCrop
    else:
        rotation_layer = tf.keras.layers.experimental.preprocessing.RandomRotation
        flip_layer = tf.keras.layers.experimental.preprocessing.RandomFlip
        crop_layer = tf.keras.layers.experimental.preprocessing.RandomCrop

    # np.pi fails with tf 2.5
    model.add(rotation_layer(0.5, fill_mode='reflect'))  # rotation range +/- 0.25 * 2pi i.e. +/- 90*.
    model.add(flip_layer())
    model.add(crop_layer(crop_size, crop_size))

    return model


def get_model(
    output_dim,
    input_size,
    crop_size,
    resize_size,
    weights_loc=None,
    include_top=True,
    expect_partial=False,
    channels=1,
    use_imagenet_weights=False,
    always_augment=True,
    dropout_rate=0.2,
    get_effnet=efficientnet_standard.EfficientNetB0
    ):
    """
    Create a trainable efficientnet model.
    First layers are galaxy-appropriate augmentation layers - see :meth:`zoobot.estimators.define_model.add_augmentation_layers`.
    Expects single channel image e.g. (300, 300, 1), likely with leading batch dimension.

    Optionally (by default) include the head (output layers) used for GZ DECaLS.
    Specifically, global average pooling followed by a dense layer suitable for predicting dirichlet parameters.
    See ``efficientnet_custom.custom_top_dirichlet``

    Args:
        output_dim (int): Dimension of head dense layer. No effect when include_top=False.
        input_size (int): Length of initial image e.g. 300 (assumed square)
        crop_size (int): Length to randomly crop image. See :meth:`zoobot.estimators.define_model.add_augmentation_layers`.
        weights_loc (str, optional): If str, load weights from efficientnet checkpoint at this location. Defaults to None.
        include_top (bool, optional): If True, include head used for GZ DECaLS: global pooling and dense layer. Defaults to True.
        expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.
        channels (int, default 1): Number of channels i.e. C in NHWC-dimension inputs. 

    Returns:
        tf.keras.Model: trainable efficientnet model including augmentations and optional head
    """

    logging.info('Input size {}, crop size {}'.format(
        input_size, crop_size))

    # model = CustomSequential()  # to log the input image for debugging
    # model = tf.keras.Sequential()

    input_shape = (input_size, input_size, channels)

    inputs = tf.keras.Input(shape=input_shape, name='preprocessed_image_batch')


    # model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    tf.summary.image(
        name='images_before_augmentation',
        data=inputs,
        max_outputs=3,
        description='Images passed to Zoobot'
    )

    # Sequential block of augmentations
    x = get_augmentation_layers(
        crop_size=crop_size,
        always_augment=always_augment)(inputs)

    tf.summary.image(
        name='images_after_augmentation',
        data=x,
        max_outputs=3,
        description='Images after applying tf.keras augmentations within Zoobot'
    )

    # Functional-created Model of EfficientNet
    shape_after_preprocessing_layers = (resize_size, resize_size, channels)
    logging.info('Model expects input of {}, adjusted to {} after preprocessing'.format(input_shape, shape_after_preprocessing_layers))

    # now headless
    effnet = efficientnet_custom.define_headless_efficientnet(
        input_shape=shape_after_preprocessing_layers,
        get_effnet=get_effnet,
        # further kwargs will be passed to get_effnet
        use_imagenet_weights=use_imagenet_weights,
    )
    x = effnet(x)  # hopefully supports functional
    tf.summary.histogram(name='embedding', data=x)


    # Functional head
    if include_top:
        assert output_dim is not None
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = custom_layers.PermaDropout(dropout_rate, name='top_dropout')(x)
        x = efficientnet_custom.custom_top_dirichlet(output_dim)(x)  # inplace
        tf.summary.histogram(name='dirichlet_outputs', data=x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="zoobot")

    # # will be updated by callback
    # model.step = tf.Variable(
    #     0, dtype=tf.int64, name='model_step', trainable=False)

    if weights_loc:
        # raise NotImplementedError
        load_weights(model, weights_loc, expect_partial=expect_partial)

    return model


# inplace
def load_weights(model, checkpoint_loc, expect_partial=False):
    """
    Load weights checkpoint to ``model``.
    Acts inplace i.e. model is modified by reference.

    Args:
        model (tf.keras.Model): Model into which to load checkpoint
        checkpoint_loc (str): path to checkpoint e.g. /path/checkpoints/checkpoint (where checkpoints includes checkpoint.index etc)
        expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.
    """
    # https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    logging.info('Loading weights from {}'.format(checkpoint_loc))
    load_status = model.load_weights(checkpoint_loc)
    load_status.assert_nontrivial_match()
    if expect_partial:  # some checkpointed values not in the current program won't match (the optimiser state during predictions, hopefully)
        load_status.expect_partial()
    # everything in the current program should match
    # do after load_status.expect_partial to silence optimizer warnings
    load_status.assert_existing_objects_matched()


def load_model(checkpoint_loc, include_top, input_size, crop_size, output_dim=34, expect_partial=False, channels=1, always_augment=True, dropout_rate=0.2):
    """    
    Utility wrapper for the common task of defining the GZ DECaLS model and then loading a pretrained checkpoint.
    crop_size must match the pretrained model used.
    output_dim must match if ``include_top=True``
    ``input_size`` and ``crop_size`` can vary as image will be resized anyway, but be careful deviating from training procedure.

    Args:
        checkpoint_loc (str): path to checkpoint e.g. /path/checkpoints/checkpoint (where checkpoints includes checkpoint.index etc)
        include_top (bool, optional): If True, include head used for GZ DECaLS: global pooling and dense layer.
        input_size (int): Length of initial image e.g. 300 (assumed square)
        crop_size (int): Length to randomly crop image. See ``get_augmentation_layers``.
        output_dim (int, optional): Dimension of head dense layer. No effect when include_top=False. Defaults to 34.
        expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.

    Returns:
        tf.keras.Model: GZ DECaLS-like model with weights loaded from ``checkpoint_loc``, optionally including GZ DECaLS-like head.
    """

    model = get_model(
        output_dim=output_dim,
        input_size=input_size,
        crop_size=crop_size,
        include_top=include_top,
        channels=channels,
        always_augment=always_augment,
        dropout_rate=dropout_rate
    )
    load_weights(model, checkpoint_loc, expect_partial=expect_partial)
    return model
