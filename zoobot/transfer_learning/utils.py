import tensorflow as tf
from tensorflow.keras import layers

from zoobot.estimators import define_model


def freeze_model(model):
    # Freeze the pretrained weights
    # inplace
    model.trainable = False


def unfreeze_model(model, unfreeze_names=['block7', 'top']):
    # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    # required for any layer to be trainable.
    # however, setting to True sets *every* layer trainable (why, tf, why...)
    # so need to then set each layer individually trainable or not trainable below
    model.trainable = True  # everything trainable, recursively.

    for layer in model.layers:
        # recursive
        # if isinstance(layer, tf.keras.Sequential) or isinstance(layer, tf.python.keras.engine.functional.Functional):  # layer is itself a model (effnet is functional due to residual connections)
        if isinstance(layer, tf.keras.Model):  # includes subclasses Sequential and Functional
            unfreeze_model(layer, unfreeze_names=unfreeze_names)  # recursive

        elif any([layer.name.startswith(name) for name in unfreeze_names]):
            if isinstance(layer, layers.BatchNormalization):
                print('freezing batch norm layer {}'.format(layer.name))
                layer.trainable = False
            else:
                print('unfreezing', layer.name)
                layer.trainable = True
                # print('Freezing batch norm layer')
                # https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization?version=stable#note_that_2
                # this will also switch layer to inference mode from tf2, no need to separately pass training=False
        else:
            layer.trainable = False  # not a recursive call, and not with a name to unfreeze

    # model will be trainable next time it is compiled

def check_batchnorm_frozen(model):
    for layer in model.layers:
        print(layer)
        if isinstance(layer, tf.keras.Model):
            check_batchnorm_frozen(layer)
        elif isinstance(layer, layers.BatchNormalization):
            assert not layer.trainable
            print('checks out')
