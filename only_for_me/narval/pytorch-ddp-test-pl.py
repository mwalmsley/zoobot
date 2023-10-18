# import datetime

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

# import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, pytorch-lightning parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')


def main():
    print("Starting...")

    args = parser.parse_args()

    class Net(pl.LightningModule):

       def __init__(self):
          super(Net, self).__init__()

          self.conv1 = nn.Conv2d(3, 6, 5)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = nn.Conv2d(6, 16, 5)
          self.fc1 = nn.Linear(16 * 5 * 5, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 10)

       def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = x.view(-1, 16 * 5 * 5)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x

       def training_step(self, batch, batch_idx):
          x, y = batch
          y_hat = self(x)
          loss = F.cross_entropy(y_hat, y)
          return loss

       def configure_optimizers(self):
          return torch.optim.Adam(self.parameters(), lr=args.lr)

    net = Net()

    """ Here we initialize a Trainer() explicitly with 1 node and 2 GPUs per node.
        To make this script more generic, you can use torch.cuda.device_count() to set the number of GPUs
        and you can use int(os.environ.get("SLURM_JOB_NUM_NODES")) to set the number of nodes. 
        We also set progress_bar_refresh_rate=0 to avoid writing a progress bar to the logs, 
        which can cause issues due to updating logs too frequently."""

    trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=1, strategy='ddp', max_epochs = args.max_epochs, enable_progress_bar=False) 

    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = CIFAR10(root='/project/def-bovy/walml/data/roots/cifar10', train=True, download=False, transform=transform_train)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)

    trainer.fit(net,train_loader)


if __name__=='__main__':
   main()