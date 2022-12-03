import logging

import tensorflow as tf

from zoobot.tensorflow.estimators import efficientnet_standard, efficientnet_custom, custom_layers

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
# for Functional, must be wrapped in Lambda layer
class LogHistogram(tf.keras.layers.Layer):
    def __init__(self, name, description=None):
        super(LogHistogram, self).__init__()
        self.log_name = name  # 'name' is taken
        self.description = description
    def call(self, data, training):
        if training:
            tf.summary.histogram(name=self.log_name, data=data, description=self.description)
        return data
class LogScalar(tf.keras.layers.Layer):
    def __init__(self, name, description=None):
        super(LogScalar, self).__init__()
        self.log_name = name  # 'name' is taken
        self.description = description
    def call(self, data, training):
        if training:
            tf.summary.scalar(name=self.log_name, data=data, description=self.description)
        return data
class LogImage(tf.keras.layers.Layer):
    def __init__(self, name, description=None):
        super(LogImage, self).__init__()
        self.log_name = name  # 'name' is taken
        self.description = description
    def call(self, data, training):
        if training:
            tf.summary.image(name=self.log_name, data=data, description=self.description)
        return data



def get_model(
    output_dim,
    input_size,
    weights_loc=None,
    include_top=True,
    expect_partial=False,
    channels=1,
    use_imagenet_weights=False,
    dropout_rate=0.2,
    test_time_dropout=True,
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
        weights_loc (str, optional): If str, load weights from efficientnet checkpoint at this location. Defaults to None.
        include_top (bool, optional): If True, include head used for GZ DECaLS: global pooling and dense layer. Defaults to True.
        expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.
        channels (int, default 1): Number of channels i.e. C in NHWC-dimension inputs. 

    Returns:
        tf.keras.Model: trainable efficientnet model including augmentations and optional head
    """
    logging.info('Input size {} (should match resize_after_crop)'.format(
        input_size))

    # model = CustomSequential()  # to log the input image for debugging
    # model = tf.keras.Sequential()

    input_shape = (input_size, input_size, channels)

    inputs = tf.keras.Input(shape=input_shape, name='preprocessed_image_batch')

    x = LogImage(
        name='images_as_input',
        description='Images passed to Zoobot'
    )(inputs)

    # now headless
    effnet = efficientnet_custom.define_headless_efficientnet(
        input_shape=input_shape,
        get_effnet=get_effnet,
        # further kwargs will be passed to get_effnet
        use_imagenet_weights=use_imagenet_weights,
    )
    x = effnet(x)
    x = LogHistogram(name='embedding')(x)

    # Functional head
    if include_top:
        assert output_dim is not None
        if test_time_dropout:
            logging.info('Using test-time dropout')
            dropout_layer = custom_layers.PermaDropout
        else:
            logging.info('Not using test-time dropout')
            dropout_layer = tf.keras.layers.Dropout
        x = dropout_layer(dropout_rate, name='top_dropout')(x)
        x = efficientnet_custom.custom_top_dirichlet(output_dim)(x)
        x = LogHistogram(name='dirichlet_outputs')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="zoobot")
 
    if weights_loc:
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


def load_model(checkpoint_loc, include_top, input_size, output_dim=34, expect_partial=False, channels=1, dropout_rate=0.2):
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
        include_top=include_top,
        channels=channels,
        dropout_rate=dropout_rate
    )
    load_weights(model, checkpoint_loc, expect_partial=expect_partial)
    return model
