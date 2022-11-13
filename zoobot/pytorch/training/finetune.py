# Based on Inigo's BYOL FT step
# https://github.com/inigoval/finetune/blob/main/finetune.py

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn.functional as F
import torchmetrics as tm
from einops import rearrange




class FineTune(pl.LightningModule):
    """
    Parent class for LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(
        self,
        encoder: pl.LightningModule,
        dim: int,
        n_classes: int,
        n_epochs=100,  # TODO early stopping
        n_layers=0,  # how many layers deep to FT
        batch_size=1024,
        lr_decay=0.75,
        prog_bar=True,
        seed=42
    ):
        super().__init__()

        self.encoder = encoder
        self.n_layers = n_layers
        self.freeze = True if n_layers == 0 else False

        self.head = torch.nn.Linear(in_features=dim, out_features=n_classes)
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.n_epochs = n_epochs

        self.seed = seed
        self.prog_bar = prog_bar

        self.train_acc = tm.Accuracy(average="micro", threshold=0)
        self.val_acc = tm.Accuracy(average="micro", threshold=0)
        self.test_acc = tm.Accuracy(average="micro", threshold=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        # x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def on_fit_start(self):

        # Log size of data-sets
        # TODO review if this works for mine
        # logging_params = {key: len(value) for key, value in self.trainer.datamodule.data.items()}
        # self.logger.log_hyperparams(logging_params)
        if self.logger is not None:
          self.logger.log({'train_dataset_size': len(datamodule.train_datamodule())})

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        self.train_acc(y_pred, y)
        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.freeze else 0)
        self.log("finetuning/train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=self.prog_bar)
        self.log("finetuning/train_loss", loss, on_step=False, on_epoch=True, prog_bar=self.prog_bar)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_pred = self.forward(x)
        self.val_acc(y_pred, y)
        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.freeze else 0)
        self.log(f"finetuning/val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=self.prog_bar)
        self.log(f"finetuning/val_loss", loss, on_step=False, on_epoch=True, prog_bar=self.prog_bar)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.test_acc(preds, y)
        self.log(f"finetuning/test_acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.freeze:
            params = self.head.parameters()
            # TODO why use SGD here but AdamW when unfreezing other layers?
              # Scale base lr=0.1
            # lr = 0.1 * self.batch_size / 256
            # return torch.optim.SGD(params, momentum=0.9, lr=lr)
            return torch.optim.Adam(params, lr=1e-4)
        else:
            lr = 0.001 * self.batch_size / 256
            params = [{"params": self.head.parameters(), "lr": lr}]

            # this bit is specific to Zoobot EffNet

            

            effnet_with_pool = list(self.encoder.children())[0]  # zoobot model is single Sequential()
            print(effnet_with_pool)

            # effnet = list(effnet_with_pool.children())[0]  # first child of that is EffNet, 2nd and 3rd are pool and identity
            # print(effnet)
            layers = [layer for layer in effnet_with_pool.children() if isinstance(layer, torch.nn.Sequential)]
            # layers.reverse()  # inplace

            # print(layers)

            # layers = self.encoder.finetuning_layers[::-1]
            
            
            
            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

            # Append parameters of layers for finetuning along with decayed learning rate
            for i, layer in enumerate(layers[: self.n_layers]):
                params.append({"params": layer.parameters(), "lr": lr * (self.lr_decay**i)})

            # Initialize AdamW optimizer with cosine decay learning rate
            opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.n_epochs)
            return [opt], [scheduler]

# https://github.com/inigoval/byol/blob/1da1bba7dc5cabe2b47956f9d7c6277decd16cc7/byol_main/networks/models.py#L29
# could just use the layer directly, but maybe this is more extensible
# class LogisticRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)


def run_finetuning(config, encoder, datamodule, logger):

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        # dirpath=config["files"] / config["run_id"] / "finetuning",
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        filename="{epoch}",  # filename may not work here TODO
        save_weights_only=True,
        # save_top_k=3,
    )

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint],
        max_epochs=config["finetune"]["n_epochs"],
        **config["trainer"],
    )

    model = FineTune(encoder, batch_size=datamodule.batch_size, **config["finetune"])

    trainer.fit(model, datamodule)

    trainer.test(model, dataloaders=datamodule)

    return checkpoint, model

if __name__ == '__main__':

    import pandas as pd
    import logging
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv('data/example_ring_catalog_basic.csv')
    # # paths = list(df['local_png_loc'])
    # # labels = list(df['ring'].astype(int))
    # # logging.info('Labels: \n{}'.format(pd.value_counts(labels))) 
    # df['file_loc'] = df['local_png_loc'].str.replace('.png', '.jpg')
    # del df['local_png_loc']
    # df.to_csv('/home/walml/repos/zoobot/data/temp.csv', index=False)

    from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
    from zoobot.pytorch.estimators import define_model

    datamodule = GalaxyDataModule(
      label_cols=['ring'],
      catalog=df,
      batch_size=32
    )

    # datamodule.setup()
    # for images, labels in datamodule.train_dataloader():
    #   print(images.shape)
    #   print(labels.shape)
    #   exit()

    config = {
        'trainer': {
          'devices': 1,
          'accelerator': 'gpu'
        },
        'finetune': {
            'dim': 1280,  # TODO rename
            'n_epochs': 100,
            'n_layers': 2,
            'n_classes': 2
        }
    }

    ckpt_loc = '/home/walml/repos/gz-decals-classifiers/results/benchmarks/pytorch/dr5/dr5_py_gr_2270/checkpoints/epoch=360-step=231762.ckpt'
    model = define_model.ZoobotLightningModule.load_from_checkpoint(ckpt_loc)  # or .best_model_path, eventually

    """
    Model:  ZoobotLightningModule(
    (train_accuracy): Accuracy()
    (val_accuracy): Accuracy()
    (model): Sequential(
      (0): EfficientNet(
    """
    # TODO self properties needed
    # 0 and 1 are self.Accuracy
    # print('Model: ', list(model.modules())[0])
    # zoobot = list(model.modules())[0]
    # print('Model: ', list(zoobot.modules())[0])

    # for name, _ in model.named_modules():
    #   print(name)

    encoder = model.get_submodule('model.0')  # includes avgpool and head
    # print(encoder)


    # encoder = define_model.get_plain_pytorch_zoobot_model(output_dim=1280, include_top=False)
    # TODO remove top?

    run_finetuning(config, encoder, datamodule, logger=None)


def investigate_structure():

    from zoobot.pytorch.estimators import define_model


    model = define_model.get_plain_pytorch_zoobot_model(output_dim=1280, include_top=False)

    # print(model)
    # with include_top=False, first and only child is EffNet
    effnet_with_pool = list(model.children())[0]

    # 0th is actually EffNet, 1st and 2nd are AvgPool and Identity
    effnet = list(effnet_with_pool.children())[0]

    for layer_n, layer in enumerate(effnet.children()):

        # first bunch are Sequential module wrapping e.g. 3 MBConv blocks
        print('\n', layer_n)
        if isinstance(layer, torch.nn.Sequential):
            print(layer)

    # so the blocks to finetune are each Sequential (repeated MBConv) block
    # and other blocks can be left alone
    # (also be careful to leave batch-norm alone)



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