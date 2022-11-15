# Based on Inigo's BYOL FT step
# https://github.com/inigoval/finetune/blob/main/finetune.py
import logging
from functools import partial

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn.functional as F
import torchmetrics as tm
# from einops import rearrange

from zoobot.pytorch.estimators import efficientnet_custom
from zoobot.pytorch.training import losses


class FineTune(pl.LightningModule):

    def __init__(
        self,
        encoder: pl.LightningModule,
        encoder_dim: int,
        label_dim: int,
        n_epochs=100,  # TODO early stopping
        n_layers=0,  # how many layers deep to FT
        batch_size=1024,
        lr_decay=0.75,
        prog_bar=True,
        seed=42,
        label_mode='classification',  # or 'counts'
        schema=None  # required for 'counts'
    ):
        super().__init__()

        self.encoder = encoder
        self.n_layers = n_layers
        self.freeze = True if n_layers == 0 else False

        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.n_epochs = n_epochs

        # by default
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None

        if label_mode == 'classification':
            logging.info('Using classification head and cross-entropy loss')
            self.head = torch.nn.Linear(
                in_features=encoder_dim, out_features=label_dim)
            self.label_smoothing = 0.1 if self.freeze else 0
            self.loss = cross_entropy_loss
            self.train_acc = tm.Accuracy(average="micro", threshold=0)
            self.val_acc = tm.Accuracy(average="micro", threshold=0)
            self.test_acc = tm.Accuracy(average="micro", threshold=0)
        elif label_mode in ['counts', 'count']:
            logging.info('Using dirichlet head and dirichlet (count) loss')
            self.head = efficientnet_custom.custom_top_dirichlet(
                input_dim=encoder_dim, output_dim=label_dim)
            self.loss = partial(
                dirichlet_loss, question_index_groups=schema.question_index_groups)
        else:
            raise ValueError(
                f'Label mode "{label_mode}" not recognised - should be "classification" or "counts"')

        self.seed = seed
        self.prog_bar = prog_bar

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        # x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    # def on_fit_start(self):

    #     # Log size of data-sets
    #     # TODO review if this works for mine
    #     # logging_params = {key: len(value) for key, value in self.trainer.datamodule.data.items()}
    #     # self.logger.log_hyperparams(logging_params)
    #     if self.logger is not None:
    #         # hopefully this exists?
    #         self.logger.log({'train_dataset_size': len(
    #             self.trainer.train_datamodule())})

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        y_pred = self.forward(x)
        # self.train_acc(y_pred, y)
        loss = self.loss(y, y_pred)
        # self.log("finetuning/train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=self.prog_bar)
        return {'loss': loss, 'preds': y_pred, 'targets': y}

    def on_train_batch_end(self, outputs, *args) -> None:
        self.log("finetuning/train_loss",
                 outputs['loss'], on_step=False, on_epoch=True, prog_bar=self.prog_bar)
        if self.train_acc is not None:
            self.train_acc(outputs['preds'], outputs['targets'])
            self.log("finetuning/train_acc", self.train_acc,
                     on_step=False, on_epoch=True, prog_bar=self.prog_bar)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_pred = self.forward(x)
        # self.val_acc(y_pred, y)
        loss = self.loss(y, y_pred)
        return {'loss': loss, 'preds': y_pred, 'targets': y}

    def on_validation_batch_end(self, outputs, *args) -> None:
        self.log(f"finetuning/val_loss",
                 outputs['loss'], on_step=False, on_epoch=True, prog_bar=self.prog_bar)
        if self.val_acc is not None:
            self.val_acc(outputs['preds'], outputs['targets'])
            self.log(f"finetuning/val_acc", self.val_acc,
                     on_step=False, on_epoch=True, prog_bar=self.prog_bar)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
      # now identical to val_step
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y, y_pred)
        return {'loss': loss, 'preds': y_pred, 'targets': y}

    def on_test_batch_end(self, outputs, *args) -> None:
        self.log('test/test_loss', outputs['loss'])
        if self.test_acc is not None:
            self.test_acc(outputs['preds'], outputs['targets'])
            self.log(f"finetuning/test_acc", self.test_acc,
                     on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.freeze:
            params = self.head.parameters()
            # adam not adamW
            return torch.optim.Adam(params, lr=1e-4)
        else:
            # lr = 0.001 * self.batch_size / 256
            lr = 1e-4
            params = [{"params": self.head.parameters(), "lr": lr}]

            # this bit is specific to Zoobot EffNet

            # zoobot model is single Sequential()
            effnet_with_pool = list(self.encoder.children())[0]

            layers = [layer for layer in effnet_with_pool.children(
            ) if isinstance(layer, torch.nn.Sequential)]
            layers.reverse()  # inplace

            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

            # Append parameters of layers for finetuning along with decayed learning rate
            for i, layer in enumerate(layers[: self.n_layers]):
                params.append({"params": layer.parameters(),
                              "lr": lr * (self.lr_decay**i)})

            # Initialize AdamW optimizer with cosine decay learning rate
            opt = torch.optim.AdamW(
                params, weight_decay=0.05, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, self.n_epochs)
            return [opt], [scheduler]

# https://github.com/inigoval/byol/blob/1da1bba7dc5cabe2b47956f9d7c6277decd16cc7/byol_main/networks/models.py#L29
class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.linear(x)
        return logits.softmax(dim=-1)
        # loss should be F.cross_entropy


def cross_entropy_loss(y, y_pred, label_smoothing=0.):
    # note the flipped arg order (sklearn convention in my func)
    return F.cross_entropy(y_pred, y, label_smoothing=label_smoothing)


def dirichlet_loss(y, y_pred, question_index_groups):
    # aggregation equiv. to sum(axis=1).mean(), but fewer operations
    return losses.calculate_multiquestion_loss(y, y_pred, question_index_groups).mean()*len(question_index_groups)


def run_finetuning(config, encoder, datamodule, logger, save_dir):

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='finetuning/val_loss',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=save_dir,
        filename="{epoch}",
        save_weights_only=True,
        save_top_k=1
    )

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint],
        max_epochs=config["finetune"]["n_epochs"],
        **config["trainer"],
    )

    model = FineTune(encoder, batch_size=datamodule.batch_size,
                     **config["finetune"])

    trainer.fit(model, datamodule)

    # trainer.test(model, dataloaders=datamodule)

    return checkpoint, model

















    

# def investigate_structure():

#     from zoobot.pytorch.estimators import define_model


#     model = define_model.get_plain_pytorch_zoobot_model(output_dim=1280, include_top=False)

#     # print(model)
#     # with include_top=False, first and only child is EffNet
#     effnet_with_pool = list(model.children())[0]

#     # 0th is actually EffNet, 1st and 2nd are AvgPool and Identity
#     effnet = list(effnet_with_pool.children())[0]

#     for layer_n, layer in enumerate(effnet.children()):

#         # first bunch are Sequential module wrapping e.g. 3 MBConv blocks
#         print('\n', layer_n)
#         if isinstance(layer, torch.nn.Sequential):
#             print(layer)

#     # so the blocks to finetune are each Sequential (repeated MBConv) block
#     # and other blocks can be left alone
#     # (also be careful to leave batch-norm alone)


""""
Sequential(
  (0): MBConv(
    (block): Sequential(
      (0): ConvNormActivation(
        (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (1): ConvNormActivation(
        (0): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (2): SqueezeExcitation(
        (avgpool): AdaptiveAvgPool2d(output_size=1)
        (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        (activation): SiLU(inplace=True)
        (scale_activation): Sigmoid()
      )
      (3): ConvNormActivation(
        (0): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stochastic_depth): StochasticDepth(p=0.1, mode=row)
  )
  (1): MBConv(
    (block): Sequential(
      (0): ConvNormActivation(
        (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (1): ConvNormActivation(
        (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (2): SqueezeExcitation(
        (avgpool): AdaptiveAvgPool2d(output_size=1)
        (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        (activation): SiLU(inplace=True)
        (scale_activation): Sigmoid()
      )
      (3): ConvNormActivation(
        (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stochastic_depth): StochasticDepth(p=0.1125, mode=row)
  )
  (2): MBConv(
    (block): Sequential(
      (0): ConvNormActivation(
        (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (1): ConvNormActivation(
        (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (2): SqueezeExcitation(
        (avgpool): AdaptiveAvgPool2d(output_size=1)
        (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        (activation): SiLU(inplace=True)
        (scale_activation): Sigmoid()
      )
      (3): ConvNormActivation(
        (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stochastic_depth): StochasticDepth(p=0.125, mode=row)
  )
)
"""
