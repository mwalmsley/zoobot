# Based on Inigo's BYOL FT step
# https://github.com/inigoval/finetune/blob/main/finetune.py
import logging
import os
import warnings
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import torch
from torch import Tensor
import torch.nn.functional as F
import torchmetrics as tm

from zoobot.pytorch.estimators import efficientnet_custom
from zoobot.pytorch.training import losses

# https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/7
# I do this recursively and only for BatchNorm2d (not dropout, which I still want active)
def freeze_batchnorm_layers(model):
      for name, child in (model.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            logging.info('freezing {} {}'.format(child, name))
            child.eval()  # no grads, no param updates, no statistic updates
        else:
          freeze_batchnorm_layers(child)  # recurse



class FinetunedZoobotLightningModule(pl.LightningModule):

    def __init__(
        self,
        encoder: pl.LightningModule,
        encoder_dim: int,
        label_dim: int,
        n_epochs=100,  # TODO early stopping
        n_layers=0,  # how many layers deep to FT
        batch_size=1024,
        lr_decay=0.75,
        dropout_prob=0.5,
        freeze_batchnorm=True,
        prog_bar=True,
        seed=42,
        label_mode='classification',  # or 'counts'
        schema=None  # required for 'counts'
    ):
        super().__init__()

        # adds every __init__ arg to model.hparams
        # will also add to wandb if using logging=wandb, I think
        # necessary if you want to reload!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # this raises a warning that encoder is already a Module hence saved in checkpoint hence no need to save as hparam
            # true - except we need it to instantiate this class, so it's really handy to have saved as well
            # therefore ignore the warning
            self.save_hyperparameters()

        self.encoder = encoder
        self.n_layers = n_layers
        self.freeze = True if n_layers == 0 else False

        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.dropout_prob = dropout_prob
        self.n_epochs = n_epochs

        # by default
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None

        self.freeze_batchnorm = freeze_batchnorm

        if self.freeze_batchnorm:
            freeze_batchnorm_layers(self.encoder)  # inplace

        if label_mode == 'classification':
            logging.info('Using classification head and cross-entropy loss')
            self.head = LinearClassifier(input_dim=encoder_dim, output_dim=label_dim, dropout_prob=self.dropout_prob)
            self.label_smoothing = 0.1 if self.freeze else 0
            self.loss = partial(cross_entropy_loss, label_smoothing=self.label_smoothing)
            self.train_acc = tm.Accuracy(average="micro", threshold=0)
            self.val_acc = tm.Accuracy(average="micro", threshold=0)
            self.test_acc = tm.Accuracy(average="micro", threshold=0)
        elif label_mode in ['counts', 'count']:
            logging.info('Using dropout+dirichlet head and dirichlet (count) loss')
            self.head = torch.nn.Sequential(
              torch.nn.Dropout(p=self.dropout_prob),
              efficientnet_custom.custom_top_dirichlet(
                  input_dim=encoder_dim, output_dim=label_dim)
            )

            self.loss = partial(
                dirichlet_loss, question_index_groups=schema.question_index_groups)
        else:
            raise ValueError(
                f'Label mode "{label_mode}" not recognised - should be "classification" or "counts"')

        self.seed = seed
        self.prog_bar = prog_bar

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y, y_pred)
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
            # using adam not adamW for now - TODO config/hparam?
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
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return F.softmax(x, dim=1)
        # loss should be F.cross_entropy


def cross_entropy_loss(y, y_pred, label_smoothing=0.):
    # note the flipped arg order (sklearn convention in my func)
    return F.cross_entropy(y_pred, y, label_smoothing=label_smoothing)


def dirichlet_loss(y, y_pred, question_index_groups):
    # aggregation equiv. to sum(axis=1).mean(), but fewer operations
    return losses.calculate_multiquestion_loss(y, y_pred, question_index_groups).mean()*len(question_index_groups)


def run_finetuning(config, encoder, datamodule, save_dir, logger=None):

    checkpoint_callback = ModelCheckpoint(
        monitor='finetuning/val_loss',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=os.path.join(save_dir, 'checkpoints'),
        filename=config["checkpoint"]["file_template"],
        save_weights_only=True,
        save_top_k=config["checkpoint"]["save_top_k"]
    )

    early_stopping_callback = EarlyStopping(
      monitor='finetuning/val_loss',
      mode='min',
      patience=config["early_stopping"]["patience"]
    )

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config["finetune"]["n_epochs"],
        **config["trainer"],
    )

    model = FinetunedZoobotLightningModule(encoder, batch_size=datamodule.batch_size,
                     **config["finetune"])

    trainer.fit(model, datamodule)

    # when ready (don't peek often, you'll overfit)
    # trainer.test(model, dataloaders=datamodule)

    return checkpoint_callback.best_model_path, model














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
      (0): Conv2dNormActivation(
        (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (1): Conv2dNormActivation(
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
      (3): Conv2dNormActivation(
        (0): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stochastic_depth): StochasticDepth(p=0.1, mode=row)
  )
  (1): MBConv(
    (block): Sequential(
      (0): Conv2dNormActivation(
        (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (1): Conv2dNormActivation(
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
      (3): Conv2dNormActivation(
        (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stochastic_depth): StochasticDepth(p=0.1125, mode=row)
  )
  (2): MBConv(
    (block): Sequential(
      (0): Conv2dNormActivation(
        (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU(inplace=True)
      )
      (1): Conv2dNormActivation(
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
      (3): Conv2dNormActivation(
        (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (stochastic_depth): StochasticDepth(p=0.125, mode=row)
  )
)
"""
