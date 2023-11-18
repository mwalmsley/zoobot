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

def main():
   
   shards = [f'shard_{x}' for x in range(4)]
   

if __name__=='__main__':
   main()