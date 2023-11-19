# import datetime
import logging
import os
import time
import glob

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

import argparse

from zoobot.pytorch.datasets import webdataset_utils
from zoobot.shared import schemas
from zoobot.pytorch import datasets

from torch import nn

class ToyLightningModule(pl.LightningModule):

   def __init__(self):
      super(ToyLightningModule, self).__init__()

      self.conv1 = nn.Conv2d(3, 6, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, 5)
      # pool again
      # shape (B, F, H, W)

   def forward(self, x):
      x = self.pool(nn.functional.relu(self.conv1(x)))
      x = self.pool(nn.functional.relu(self.conv2(x)))
      time.sleep(1)
      return torch.mean(x, dim=(1, 2, 3))  # shape (B)


   def training_step(self, batch, batch_idx):
      images, labels = batch
      y_hat = self(images)  # mean after some convs
      y = labels[:, 0].float() / 20.  # first random number, divided by a big number to now be below 0
      loss = F.cross_entropy(y_hat, y)
      return loss  # meaningless but mathematically works

   def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():

   logging.basicConfig(level=logging.INFO)
   logging.warning('Script start')

   parser = argparse.ArgumentParser()
   parser.add_argument('--save-dir', dest='save_dir', type=str)
   parser.add_argument('--batch-size', dest='batch_size', default=16, type=int)
   parser.add_argument('--gpus', dest='gpus', default=1, type=int)
   parser.add_argument('--nodes', dest='nodes', default=1, type=int)
   parser.add_argument('--mixed-precision', dest='mixed_precision',
   default=False, action='store_true')
   parser.add_argument('--debug', dest='debug',
   default=False, action='store_true')
   parser.add_argument('--wandb', dest='wandb',
   default=False, action='store_true')
   parser.add_argument('--seed', dest='random_state', default=1, type=int)
   args = parser.parse_args()

   # if os.path.isdir('/home/walml/repos/zoobot'):
   save_dir = '/home/walml/repos/temp'

   # else:
      # save_dir = os.environ['SLURM_TMPDIR']

   schema = schemas.decals_all_campaigns_ortho_schema

   # shards = webdataset_utils.make_mock_wds(save_dir, schema.label_cols, n_shards=10, shard_size=32)
   # exit()
   # webdataset_utils.load_wds_directly(shards[0], max_to_load=None)
   # webdataset_utils.load_wds_with_webdatamodule(shards, label_cols=schema.label_cols, max_to_load=None)
   shards = list(glob.glob('/home/walml/repos/temp/mock_shard_*_32.tar'))
   # exit()

   train_shards = shards[:8]
   val_shards = shards[8:]  # not actually used

   datamodule = datasets.webdatamodule.WebDataModule(
      train_urls=train_shards,
      val_urls=val_shards,
      batch_size=args.batch_size,
      num_workers=1,
      label_cols=schema.label_cols,
      cache_dir=None
      # TODO pass through the rest
   )
   # use_distributed_sampler=False

   trainer = pl.Trainer(
      # log_every_n_steps=16,  # at batch 512 (A100 MP max), DR5 has ~161 train steps
      accelerator='gpu',
      devices=args.gpus,  # per node
      num_nodes=args.nodes,
      #   strategy='auto',
      precision='16-mixed',
      logger=False,
      #   callbacks=callbacks,
      max_epochs=1,
      default_root_dir=save_dir,
      #   plugins=plugins,
      # use_distributed_sampler=use_distributed_sampler
   )

   # logging.info((trainer.strategy, trainer.world_size,
   # trainer.local_rank, trainer.global_rank, trainer.node_rank))

   lightning_model = ToyLightningModule()

   trainer.fit(lightning_model, datamodule)  # uses batch size of datamodule

   # batch size 16
   # shard size 16, 10 shards with 8 being assigned as training shards so 8*32 train images, 8*2=16 train batches


if __name__=='__main__':
   main()
