
import os
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.io import read_image  # may want to replace with self.read_image for e.g. FITS, Liza
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl

# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html

# for now, all this really does is split a dataframe and apply no transforms
class DECALSDR8DataModule(pl.LightningDataModule):
    def __init__(self, catalog: pd.DataFrame, schema, greyscale=True, seed=42):
        super().__init__()
        self.catalog = catalog
        self.schema = schema
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # TODO make optional
            # transforms.RandomCrop(size=(224, 224)),
            transforms.RandomResizedCrop(
                size=(224, 224),  # after crop then resize
                scale=(0.7, 0.8),  # crop factor
                ratio=(0.9, 1.1),  # crop aspect ratio
                interpolation=transforms.InterpolationMode.BILINEAR),  # new aspect ratio
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90., interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
            ])  # TODO more

    def prepare_data(self):
        pass   # could include some basic checks

    def setup(self, stage: Optional[str] = None):

        train_catalog, hidden_catalog = train_test_split(
            self.catalog, train_size=0.7, random_state=self.seed
        )
        val_catalog, test_catalog = train_test_split(
            hidden_catalog, train_size=1./3., random_state=self.seed
        )
        del hidden_catalog

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = DECALSDR8Dataset(
                catalog=train_catalog, schema=self.schema, transform=self.transform
            )
            self.val_dataset = DECALSDR8Dataset(
                catalog=val_catalog, schema=self.schema, transform=self.transform
            ) 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = DECALSDR8Dataset(
                catalog=test_catalog, schema=self.schema, transform=self.transform
            )


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=256, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=256)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=256)

    # @property
    # def dims(self):



# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class DECALSDR8Dataset(Dataset):
    def __init__(self, catalog: pd.DataFrame, schema, transform=None, target_transform=None):
        # catalog should be split already
        # should have correct image locations under file_loc
        self.catalog = catalog
        self.schema = schema
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        galaxy = self.catalog.iloc[idx]
        img_path = galaxy['file_loc']
        image = read_image(img_path)
        label = get_galaxy_label(galaxy, self.schema)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# import logging

def get_galaxy_label(galaxy, schema):
    return galaxy[schema.label_cols].values.astype(int)
    # logging.info(votes)
    # return torch.from_numpy(votes)
