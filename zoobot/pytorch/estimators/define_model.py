import logging

import torch
from torch import nn

from zoobot.pytorch.estimators import efficientnet_standard, efficientnet_custom, custom_layers

import pytorch_lightning as pl



class GenericLightningModule(pl.LightningModule):
    """
    All Zoobot models use the lightningmodule API and so share this structure
    The funcs below this class define e.g. the model itself, the loss function, etc.

    Use like
        lightning_model = GenericLightningModule(plain_pytorch_model, loss_func)
    """

    def __init__(
        self,
        model,
        loss_func,
        ):
        super().__init__()

        self.model = model

        self.loss_func = loss_func  # accept (labels, preds), return losses of shape (batch)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        predictions = self(x)  # by default, these are Dirichlet concentrations

        # true, pred convention as with sklearn
        # self.loss_func returns shape of (galaxy, question), sum to ()
        loss = torch.sum(self.loss_func(predictions, labels))/len(labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # identical to training_step except for log
        x, labels = batch
        predictions = self(x)
        loss = torch.sum(self.loss_func(predictions, labels))/len(labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # similarly
        x, labels = batch
        predictions = self(x)
        loss = torch.sum(self.loss_func(predictions, labels))/len(labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # torch and tf defaults are the same (now), but be explicit anyway just for clarity
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))  


def get_plain_pytorch_zoobot_model(
    output_dim,
    weights_loc=None,
    include_top=True,
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

