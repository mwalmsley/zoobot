import tensorflow as tf
from tensorflow.keras import layers

from zoobot.estimators import define_model


def freeze_model(model):
    # Freeze the pretrained weights
    model.trainable = False


def unfreeze_model(model, n_layers_to_unfreeze=None):
    # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    model.trainable = True
    if isinstance(n_layers_to_unfreeze, int):
        # pylint: disable=invalid-unary-operand-type
        layers_to_unfreeze = model.layers[-n_layers_to_unfreeze:]
        # pylint: enable=invalid-unary-operand-type
    else:
        layers_to_unfreeze = model.layers

    for layer in layers_to_unfreeze:
        # leaving BatchNorm layers frozen
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    # model will be trainable next time it is compiled
