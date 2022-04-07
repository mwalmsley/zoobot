
import os
from typing import Optional
import logging
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.io import read_image  # may want to replace with self.read_image for e.g. FITS, Liza
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import simplejpeg
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
class DECALSDR8DataModule(pl.LightningDataModule):
    def __init__(
        self,
        schema,
        # provide full catalog for automatic split, or...
        catalog=None,
        # provide train/val/test catalogs for your own previous split
        train_catalog=None,
        val_catalog=None,
        test_catalog=None,
        greyscale=True,
        album=False,
        batch_size=256,
        resize_size=224,
        crop_scale_bounds=(0.7, 0.8),
        crop_ratio_bounds=(0.9, 1.1),
        use_memory=False,
        num_workers=4,
        prefetch_factor=4,
        seed=42
        ):
        super().__init__()

        if catalog is not None:  # catalog provided, should not also provide explicit split catalogs
            assert train_catalog is None
            assert val_catalog is None
            assert test_catalog is None
        else:  # catalog not provided, must provide explicit split catalogs
            assert train_catalog is not None
            assert val_catalog is not None
            assert test_catalog is not None

        self.schema = schema

        self.catalog = catalog
        self.train_catalog = train_catalog
        self.val_catalog = val_catalog
        self.test_catalog = test_catalog

        self.batch_size = batch_size

        self.resize_size = resize_size
        self.crop_scale_bounds = crop_scale_bounds
        self.crop_ratio_bounds = crop_ratio_bounds

        self.use_memory = use_memory

        self.num_workers = num_workers
        self.seed = seed

        self.greyscale = greyscale
        self.album = album

        if self.album:
            logging.info('Using albumentations for augmentations')
            self.transform_with_album()
        else:
            logging.info('Using torchvision for augmentations')
            self.transform_with_torchvision()

        self.prefetch_factor = prefetch_factor
        self.dataloader_timeout = 120  # seconds

        logging.info('Num workers: {}'.format(self.num_workers))
        logging.info('Prefetch factor: {}'.format(self.prefetch_factor))


    def transform_with_torchvision(self):

        # assume input is 0-255 uint8 tensor

        transforms_to_apply = [transforms.ConvertImageDtype(torch.float)]  # automatically normalises from 0-255 int to 0-1 float
    
        if self.greyscale:
            transforms_to_apply += [GrayscaleUnweighted()]    # transforms.Grayscale() adds perceptual weighting to rgb channels

        transforms_to_apply += [
            transforms.RandomResizedCrop(
                size=self.resize_size,  # assumed square
                scale=self.crop_scale_bounds,  # crop factor
                ratio=self.crop_ratio_bounds,  # crop aspect ratio
                interpolation=transforms.InterpolationMode.BILINEAR),  # new aspect ratio
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180., interpolation=transforms.InterpolationMode.BILINEAR)
        ]

        self.transform = transforms.Compose(transforms_to_apply)


    def transform_with_album(self):

        if self.greyscale:
            transforms_to_apply = [A.Lambda(name='ToGray', image=ToGray(reduce_channels=True), always_apply=True)]
        else:
            transforms_to_apply = []

            transforms_to_apply += [
                A.ToFloat(),
                A.Rotate(limit=180, interpolation=1, always_apply=True, border_mode=0, value=0), # anything outside of the original image is set to 0.
                A.RandomResizedCrop(
                    height=self.resize_size, # after crop resize
                    width=self.resize_size,
                    scale=self.crop_scale_bounds, # crop factor
                    ratio=self.crop_ratio_bounds, # crop aspect ratio
                    interpolation=1, # This is "INTER_LINEAR" == BILINEAR interpolation. See: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
                    always_apply=True
                ), # new aspect ratio
                A.VerticalFlip(p=0.5),
                ToTensorV2()
            ]

        self.transform = A.Compose(transforms_to_apply)  # TODO more


    def prepare_data(self):
        pass   # could include some basic checks

    def setup(self, stage: Optional[str] = None):

        if self.catalog is not None:
            self.train_catalog, hidden_catalog = train_test_split(
                self.catalog, train_size=0.7, random_state=self.seed
            )
            self.val_catalog, self.test_catalog = train_test_split(
                hidden_catalog, train_size=1./3., random_state=self.seed
            )
            del hidden_catalog
        else:
            assert self.train_catalog is not None
            assert self.val_catalog is not None
            assert self.test_catalog is not None

        # isn't python clever - first class classes
        if self.use_memory:
            # dataset_class = DECALSDR8DatasetMemory
            raise NotImplementedError
        else:
            dataset_class = DECALSDR8Dataset

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = dataset_class(
                catalog=self.train_catalog, schema=self.schema, album=self.album, transform=self.transform
            )
            self.val_dataset = dataset_class(
                catalog=self.val_catalog, schema=self.schema, album=self.album, transform=self.transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = dataset_class(
                catalog=self.test_catalog, schema=self.schema, transform=self.transform
            )


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers>0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers>0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers>0, prefetch_factor=self.prefetch_factor, timeout=self.dataloader_timeout)



# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class DECALSDR8Dataset(Dataset):
    def __init__(self, catalog: pd.DataFrame, schema, album=False, transform=None, target_transform=None):
        # catalog should be split already
        # should have correct image locations under file_loc
        self.catalog = catalog
        self.album = album
        self.schema = schema
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        galaxy = self.catalog.iloc[idx]
        # option A
        # img_path = galaxy['file_loc']
        # image = read_image(img_path) # PIL under the hood: Returns CHW Tensor.
        # option B - tiny bit faster when CPU-limited
        with open(galaxy['file_loc'], 'rb') as f:
            image = torch.from_numpy(decode_jpeg(f.read()).transpose(2,0,1))
        label = get_galaxy_label(galaxy, self.schema)

        # logging.info((image.shape, torch.max(image), image.dtype, label))  # always 0-255 uint8

        if self.transform:
            # TODO eww an extra if
            if self.album:
                # album wants HWC np.array
                image = np.asarray(image).transpose(1,2,0)  # send it back to np from tensor...
                # Returns torch.tensor CHW for torch using ToTensorV2() as last transform
                # e.g.: https://albumentations.ai/docs/examples/pytorch_classification/
                image = self.transform(image=image)['image']
            else:   
                image = self.transform(image)  # already a CHW tensor, which torchvision wants

        if self.target_transform:
            label = self.target_transform(label)

        # logging.info((image.shape, torch.max(image), image.dtype, label))  #  should be 0-1 float
        return image, label


def load_encoded_jpeg(loc):
    with open(loc, "rb") as f:
        return f.read()  # bytes, not yet decoded

def decode_jpeg(encoded_bytes):
    return simplejpeg.decode_jpeg(encoded_bytes, fastdct=True, fastupsample=True)


def get_galaxy_label(galaxy, schema):
    return galaxy[schema.label_cols].values.astype(int)

# torchvision


class GrayscaleUnweighted(torch.nn.Module):

    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def forward(self, img):
        """
        PyTorch (and tensorflow) does greyscale conversion as a *weighted* mean by default (as colours have different perceptual brightnesses).
        Here, do a simple mean.
        Args:
            img (Tensor): Image to be converted to grayscale.

        Returns:
            Tensor: Grayscaled image.
        """
        # https://pytorch.org/docs/stable/generated/torch.mean.html
        return img.mean(dim=-3, keepdim=True)  # (..., C, H, W) convention

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)



# albumentations versuib of GrayscaleUnweighted
class ToGray():

    def __init__(self, reduce_channels=False):
        if reduce_channels:
            self.mean = lambda arr: arr.mean(axis=2, keepdims=True)
        else:
            self.mean = lambda arr: arr.mean(axis=2, keepdims=True).repeat(3, axis=2)

    def __call__(self, image, **kwargs):
        return self.mean(image)


"""
I played with an in-memory version for speed, but it wasn't particularly faster - cpu is the limiting factor at Manchester
"""

# class DECALSDR8DatasetMemory(DECALSDR8Dataset):
#     # compressed data will be loaded into memory
#     # use cpu/simplejpeg to decode as needed, can't store decoded all in memory

#     def __init__(self, catalog: pd.DataFrame, schema, transform=None, target_transform=None):
#         super().__init__(catalog=catalog, schema=schema, transform=transform, target_transform=target_transform)

#         logging.info('Loading encoded jpegs into memory: {}'.format(len(self.catalog)))

#         self.catalog = self.catalog.sort_values('file_loc')  # for sequential -> faster hddreading. Shuffle later.
#         logging.warning('In-Memory loading will shuffle for faster reading - outputs will not align with earlier/later reads')

#         # assume I/O limited so use pool
#         pool = Pool(processes=int(os.cpu_count()/2))
#         self.encoded_galaxies = pool.map(load_encoded_jpeg, self.catalog['file_loc'])  # list not generator
#         logging.info('Loading complete: {}'.format(len(self.encoded_galaxies)))

#         shuffle_indices = list(range(len(self.catalog)))
#         random.shuffle(shuffle_indices)

#         self.catalog = self.catalog.iloc[shuffle_indices].reset_index()
#         self.encoded_galaxies = list({self.encoded_galaxies[idx] for idx in shuffle_indices})
#         logging.info('Shuffling complete')

#     def __getitem__(self, idx):
#         galaxy = self.catalog.iloc[idx]
#         label = get_galaxy_label(galaxy, self.schema)
#         image = decode_jpeg(self.encoded_galaxies[idx])
#         # Read image as torch array for consistency

#         # logging.info(image.shape)
#         if self.transform:
#             # image = np.asarray(image).transpose(2,0,1)  # not needed simplejpeg gives np array HWC
#             # logging.info(type(image))
#             image = self.transform(image=image)['image']  # assumed to output torch
#             # image = self.transform(image)
#         else:
#             image = torch.from_numpy(image)

#         if self.target_transform:
#             label = self.target_transform(label)

#         return image, label
