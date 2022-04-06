import logging
from sys import modules

import numpy as np
import torch
from torch import nn

from zoobot.pytorch.estimators import efficientnet_standard, efficientnet_custom, custom_layers

import pytorch_lightning as pl


class ZoobotModel(pl.LightningModule):
    def __init__(self, schema, loss, channels, get_architecture=efficientnet_standard.efficientnet_b0, representation_dim=1280):
        super().__init__()

        self.schema = schema

        self.model = get_model(self.output_dims, channels=channels, get_architecture=get_architecture, representation_dim=representation_dim)
        self.loss = loss

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        concentrations = self(x)
        # true, pred convention as with sklearn
        # self.loss returns shape of (galaxy, question), sum to ()
        loss = torch.sum(self.loss(labels, concentrations, self.schema.question_index_groups))/len(labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # identical to training_step except for log
        x, labels = batch
        concentrations = self(x)
        loss = torch.sum(self.loss(labels, concentrations, self.schema.question_index_groups))/len(labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # similarly
        x, labels = batch
        concentrations = self(x)
        loss = torch.sum(self.loss(labels, concentrations, self.schema.question_index_groups))/len(labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):  # was lr=0.02, from copying tutorial - oops
        # torch and tf defaults are the same (now), but be explicit anyway just for clarity
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))  


    # def on_train_batch_end(self, outputs, batch, batch_idx, unused=None):
    #     torch.cuda.empty_cache()
    #     return super().on_train_batch_end(outputs, batch, batch_idx, unused)

    @property
    def output_dims(self):
        return len(self.schema.answers)


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


# def add_augmentation_layers(model, crop_size, resize_size, always_augment=False):
#     """
#     Add image augmentation layers to end of ``model``.
    
#     The following augmentations are applied, in order:
#         - Random rotation (aliased)
#         - Random flip (horizontal and/or vertical)
#         - Random crop (not centered) down to ``(crop_size, crop_size)``
#         - Resize down to ``(resize_size, resize_size)``

#     If crop_size is within 10 of resize_size, resizing is skipped and instead the image is cropped directly to `resize_size`.
#     This is both faster and avoids information loss from aliasing.
#     I strongly suggest this approach if possible.

#     Args:
#         model (tf.keras.Model): Model to add augmentation layers. Layers are added at *end*, so likely an empty model e.g. tf.keras.Sequential()
#         crop_size (int): desired length of image after random crop (assumed square)
#         resize_size (int): desired length of image after resizing (assumed square)
#         always_augment (bool, optional): If True, augmentations also happen at test time. Defaults to False.
#     """
#     if crop_size < resize_size:
#         logging.warning('Crop size {} < final size {}, losing resolution'.format(
#             crop_size, resize_size))

#     resize = True
#     if np.abs(crop_size - resize_size) < 10:
#         logging.warning(
#             'Crop size and final size are similar: skipping resizing and cropping directly to resize_size (ignoring crop_size)')
#         resize = False
#         crop_size = resize_size

#     if always_augment:
#         rotation_layer = custom_layers.PermaRandomRotation
#         flip_layer = custom_layers.PermaRandomFlip
#         crop_layer = custom_layers.PermaRandomCrop
#     else:
#         rotation_layer = tf.keras.layers.experimental.preprocessing.RandomRotation
#         flip_layer = tf.keras.layers.experimental.preprocessing.RandomFlip
#         crop_layer = tf.keras.layers.experimental.preprocessing.RandomCrop


#     # np.pi fails with tf 2.5
#     model.add(rotation_layer(0.5, fill_mode='reflect'))  # rotation range +/- 0.25 * 2pi i.e. +/- 90*.
#     model.add(flip_layer())
#     model.add(crop_layer(crop_size, crop_size))
#     if resize:
#         logging.info('Using resizing, to {}'.format(resize_size))
#         model.add(tf.keras.layers.experimental.preprocessing.Resizing(
#             resize_size, resize_size, interpolation='bilinear'
#         ))


def get_model(
    output_dim,
    weights_loc=None,
    include_top=True,
    expect_partial=False,
    channels=1,
    use_imagenet_weights=False,
    always_augment=True,
    dropout_rate=0.2,
    get_architecture=efficientnet_standard.efficientnet_b0,
    representation_dim=1280  # or 2048 for resnet
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
        resize_size (int): Length to resize image. See :meth:`zoobot.estimators.define_model.add_augmentation_layers`.
        weights_loc (str, optional): If str, load weights from efficientnet checkpoint at this location. Defaults to None.
        include_top (bool, optional): If True, include head used for GZ DECaLS: global pooling and dense layer. Defaults to True.
        expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.
        channels (int, default 1): Number of channels i.e. C in NHWC-dimension inputs. 

    Returns:
        tf.keras.Model: trainable efficientnet model including augmentations and optional head
    """

    modules_to_use = []

    effnet = get_architecture(
        input_channels=channels,
        use_imagenet_weights=use_imagenet_weights,
        include_top=False,  # no final three layers: pooling, dropout and dense
    )
    modules_to_use.append(effnet)

    if include_top:
        assert output_dim is not None
        # modules_to_use.append(tf.keras.layers.GlobalAveragePooling2D())  # included already in standard effnet in pytorch version - "AdaptiveAvgPool2d"
        modules_to_use.append(custom_layers.PermaDropout(dropout_rate))
        modules_to_use.append(efficientnet_custom.custom_top_dirichlet(representation_dim, output_dim))  # unlike tf version, not inplace

    if weights_loc:
        raise NotImplementedError
    #     load_weights(model, weights_loc, expect_partial=expect_partial)

    model = nn.Sequential(*modules_to_use)

    return model

# # inplace
# def load_weights(model, checkpoint_loc, expect_partial=False):
#     """
#     Load weights checkpoint to ``model``.
#     Acts inplace i.e. model is modified by reference.

#     Args:
#         model (tf.keras.Model): Model into which to load checkpoint
#         checkpoint_loc (str): path to checkpoint e.g. /path/checkpoints/checkpoint (where checkpoints includes checkpoint.index etc)
#         expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.
#     """
#     # https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
#     logging.info('Loading weights from {}'.format(checkpoint_loc))
#     load_status = model.load_weights(checkpoint_loc)
#     load_status.assert_nontrivial_match()
#     load_status.assert_existing_objects_matched()
#     if expect_partial:  # some checkpointed values won't match (the optimiser state during predictions, hopefully)
#         load_status.expect_partial()


# def load_model(checkpoint_loc, include_top, input_size, crop_size, resize_size, output_dim=34, expect_partial=False, channels=1, always_augment=True, dropout_rate=0.2):
#     """    
#     Utility wrapper for the common task of defining the GZ DECaLS model and then loading a pretrained checkpoint.
#     resize_size must match the pretrained model used.
#     output_dim must match if ``include_top=True``
#     ``input_size`` and ``crop_size`` can vary as image will be resized anyway, but be careful deviating from training procedure.

#     Args:
#         checkpoint_loc (str): path to checkpoint e.g. /path/checkpoints/checkpoint (where checkpoints includes checkpoint.index etc)
#         include_top (bool, optional): If True, include head used for GZ DECaLS: global pooling and dense layer.
#         input_size (int): Length of initial image e.g. 300 (assumed square)
#         crop_size (int): Length to randomly crop image. See ``add_augmentation_layers``.
#         resize_size (int): Length to resize image. See ``add_augmentation_layers``.
#         output_dim (int, optional): Dimension of head dense layer. No effect when include_top=False. Defaults to 34.
#         expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.

#     Returns:
#         tf.keras.Model: GZ DECaLS-like model with weights loaded from ``checkpoint_loc``, optionally including GZ DECaLS-like head.
#     """

#     model = get_model(
#         output_dim=output_dim,
#         input_size=input_size,
#         crop_size=crop_size,
#         resize_size=resize_size,
#         include_top=include_top,
#         channels=channels,
#         always_augment=always_augment,
#         dropout_rate=dropout_rate
#     )
#     load_weights(model, checkpoint_loc, expect_partial=expect_partial)
#     return model
