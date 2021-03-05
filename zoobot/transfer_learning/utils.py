import tensorflow as tf
from tensorflow.keras import layers

from zoobot.estimators import define_model



# TODO refactor elsewhere?



def get_headless_frozen_model(base_model: tf.keras.Sequential, new_head: tf.keras.Sequential):  # use penultimate layer by default
    base_model.summary()

    freeze_model(base_model)
    headless_output = base_model.layers[-2].output
    new_output = layers.Dense(2, activation="softmax", name="softmax_output")(headless_output)

    return tf.keras.models.Model(inputs=base_model.inputs, outputs=new_output)
    

    # base_model.summary()
    # base_model.pop()
    # base_model.summary()

    # base_model.add(new_head)
    # return base_model  # now modified to be frozen except for the new head


def freeze_model(model):
    # Freeze the pretrained weights
    model.trainable = False


def classification_head(num_classes):
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=1280),  # dim after GlobalAveragePooling, before dense layer
        layers.Dense(num_classes, activation="softmax", name="softmax_output")
    ])




def unfreeze_model(model, n_layers_to_unfreeze=None):

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

