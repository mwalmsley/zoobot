import os
import types
import logging
import torch.utils.data
import numpy as np
import pytorch_lightning as pl
from itertools import islice

import webdataset as wds

from galaxy_datasets import transforms

# https://github.com/webdataset/webdataset-lightning/blob/main/train.py
class WebDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_urls=None,
            val_urls=None,
            test_urls=None,
            predict_urls=None,
            label_cols=None,
            # hardware
            batch_size=64,
            num_workers=4,
            prefetch_factor=4,
            cache_dir=None,
            color=False,
            crop_scale_bounds=(0.7, 0.8),
            crop_ratio_bounds=(0.9, 1.1),
            resize_after_crop=224
            ):
        super().__init__()

        self.train_urls = train_urls
        self.val_urls = val_urls
        self.test_urls = test_urls
        self.predict_urls = predict_urls

        if train_urls is not None:
            # assume the size of each shard is encoded in the filename as ..._{size}.tar
            self.train_size = interpret_dataset_size_from_urls(train_urls)
        if val_urls is not None:
            self.val_size = interpret_dataset_size_from_urls(val_urls)
        if test_urls is not None:
            self.test_size = interpret_dataset_size_from_urls(test_urls)
        if predict_urls is not None:
            self.predict_size = interpret_dataset_size_from_urls(predict_urls)

        self.label_cols = label_cols

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.cache_dir = cache_dir

        # could use mixin
        self.color = color
        self.resize_after_crop = resize_after_crop
        self.crop_scale_bounds = crop_scale_bounds
        self.crop_ratio_bounds = crop_ratio_bounds

        for url_name in ['train', 'val', 'test', 'predict']:
            urls = getattr(self, f'{url_name}_urls')
            if urls is not None:
                logging.info(f"{url_name} (before hardware splits) = {len(urls)} e.g. {urls[0]}", )

        logging.info(f"batch_size: {self.batch_size}, num_workers: {self.num_workers}")

    def make_image_transform(self, mode="train"):

        augmentation_transform = transforms.default_transforms(
            crop_scale_bounds=self.crop_scale_bounds,
            crop_ratio_bounds=self.crop_ratio_bounds,
            resize_after_crop=self.resize_after_crop,
            pytorch_greyscale=not self.color
        )  # A.Compose object

        # logging.warning('Minimal augmentations for speed test')
        # augmentation_transform = transforms.fast_transforms(
        #     resize_after_crop=self.resize_after_crop,
        #     pytorch_greyscale=not self.color
        # )  # A.Compose object


        def do_transform(img):
            return np.transpose(augmentation_transform(image=np.array(img))["image"], axes=[2, 0, 1]).astype(np.float32)
        return do_transform


    def make_loader(self, urls, mode="train"):
        logging.info('Making loader with mode {}'.format(mode))

        dataset_size = getattr(self, f'{mode}_size')
        if mode == "train":
            shuffle = min(dataset_size, 5000)
        else:
            assert mode in ['val', 'test', 'predict'], mode
            shuffle = 0

        transform_image = self.make_image_transform(mode=mode)

        transform_label = dict_to_label_cols_factory(self.label_cols)

        dataset =  wds.WebDataset(urls, cache_dir=self.cache_dir, shardshuffle=shuffle>0, nodesplitter=nodesplitter_func)
            # https://webdataset.github.io/webdataset/multinode/ 
            # WDS 'knows' which worker it is running on and selects a subset of urls accordingly
           
        if shuffle > 0:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.decode("rgb")
    
        if mode == 'predict':
            if self.label_cols != ['id_str']:
                logging.info('Will return images only')
                # dataset = dataset.extract_keys('image.jpg').map(transform_image)
                dataset = dataset.to_tuple('image.jpg').map_tuple(transform_image)  # (im,) tuple. But map applied to all elements
                # .map(get_first)
            else:
                logging.info('Will return id_str only')
                dataset = dataset.to_tuple('__key__')
        else:
            dataset = (
                dataset.to_tuple('image.jpg', 'labels.json')
                .map_tuple(transform_image, transform_label)
            )

        # torch collate stacks dicts nicely while webdataset only lists them
        # so use the torch collate instead
        dataset = dataset.batched(self.batch_size, torch.utils.data.default_collate, partial=False) 

        # temp hack instead
        if mode in ['train', 'val']:
            assert dataset_size % self.batch_size == 0, (dataset_size, self.batch_size, dataset_size % self.batch_size)
        # for test/predict, always single GPU anyway

        # if mode == "train":
            # ensure same number of batches in all clients
            # loader = loader.ddp_equalize(dataset_size // self.batch_size)
            # print("# loader length", len(loader))

        loader = webdataset_to_webloader(dataset, self.num_workers, self.prefetch_factor)

        return loader

    def train_dataloader(self):
        return self.make_loader(self.train_urls, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")
    
    def predict_dataloader(self):
        return self.make_loader(self.predict_urls, mode="predict")

def identity(x):
    return x

def nodesplitter_func(urls):
    urls_to_use = list(wds.split_by_node(urls))  # rely on WDS for the hard work
    rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
    logging.debug(
        f'''
        Splitting urls within webdatamodule with WORLD_SIZE: 
        {world_size}, RANK: {rank}, WORKER: {worker} of {num_workers}\n
        URLS: {len(urls_to_use)} (e.g. {urls_to_use[0]})\n\n)
        '''
        )
    return urls_to_use

def interpret_shard_size_from_url(url):
    assert isinstance(url, str), TypeError(url)
    return int(url.rstrip('.tar').split('_')[-1])

def interpret_dataset_size_from_urls(urls):
    return sum([interpret_shard_size_from_url(url) for url in urls])

def get_first(x):
    return x[0]

def custom_collate(x):
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    return torch.utils.data.default_collate(x)


def webdataset_to_webloader(dataset, num_workers, prefetch_factor):
    loader = wds.WebLoader(
            dataset,
            batch_size=None,  # already batched
            shuffle=False,  # already shuffled
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor
        )

    # loader.length = dataset_size // batch_size
    return loader


def dict_to_label_cols_factory(label_cols=None):
    if label_cols is not None:
        def label_transform(label_dict):
            return torch.from_numpy(np.array([label_dict.get(col, 0) for col in label_cols])).double()  # gets cast to int in zoobot loss
        return label_transform
    else:
        return identity  # do nothing
