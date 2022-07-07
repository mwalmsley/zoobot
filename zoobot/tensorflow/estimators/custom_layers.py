import tensorflow as tf
from tensorflow.keras import layers


# this would be more elegant, but sadly using a stack with multiple sequential models 
# (as opposed to some layers, then a sequential model) 
# seems to silently break loading weights

# class CustomPreprocessing(tf.keras.Sequential):
#     def call(self, x, training):
#         # I add the step manually to the top-level model, but this inner model won't have that same var - could add if needed
#         x = super().call(x, training=True)  # always use training=True
#         tf.summary.image('after_preprocessing_layers', x, step=0)
#         return x


class PermaDropout(layers.Dropout):
    def call(self, x, training=None):
        return super().call(x, training=True)  # ME, force dropout on at test time

# class PermaRandomTranslation(layers.experimental.preprocessing.RandomTranslation):
#     def call(self, x, training=None):
#         return super().call(x, training=True)

class PermaRandomRotation(tf.keras.layers.RandomRotation):
    def call(self, x, training=None):
        return super().call(x, training=True)

class PermaRandomFlip(tf.keras.layers.RandomFlip):
    def call(self, x, training=None):
        return super().call(x, training=True)

class PermaRandomCrop(tf.keras.layers.RandomCrop):
    def call(self, x, training=None):
        return super().call(x, training=True)
