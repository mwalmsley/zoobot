

from torch import Tensor, nn, functional


# this would be more elegant, but sadly using a stack with multiple sequential models 
# (as opposed to some layers, then a sequential model) 
# seems to silently break loading weights

# class CustomPreprocessing(tf.keras.Sequential):
#     def call(self, x, training):
#         # I add the step manually to the top-level model, but this inner model won't have that same var - could add if needed
#         x = super().call(x, training=True)  # always use training=True
#         tf.summary.image('after_preprocessing_layers', x, step=0)
#         return x


class PermaDropout(nn.modules.dropout._DropoutNd):
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout
    def forward(self, input: Tensor) -> Tensor:
            return nn.functional.dropout(input, self.p, True, self.inplace)  # simply replaced self.training with True

# # class PermaRandomTranslation(layers.experimental.preprocessing.RandomTranslation):
# #     def call(self, x, training=None):
# #         return super().call(x, training=True)

# class PermaRandomRotation(tf.keras.layers.experimental.preprocessing.RandomRotation):
#     def call(self, x, training=None):
#         return super().call(x, training=True)

# class PermaRandomFlip(tf.keras.layers.experimental.preprocessing.RandomFlip):
#     def call(self, x, training=None):
#         return super().call(x, training=True)

# class PermaRandomCrop(tf.keras.layers.experimental.preprocessing.RandomCrop):
#     def call(self, x, training=None):
#         return super().call(x, training=True)
