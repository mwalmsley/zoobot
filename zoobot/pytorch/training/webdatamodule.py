import os
import types
import logging
import torch.utils.data
import numpy as np
import pytorch_lightning as pl
from itertools import islice

import webdataset as wds

from galaxy_datasets.transforms import default_transforms

# https://github.com/webdataset/webdataset-lightning/blob/main/train.py
class WebDataModule(pl.LightningDataModule):
    def __init__(self, train_urls, val_urls, train_size=None, val_size=None, label_cols=None, batch_size=64, num_workers=4, prefetch_factor=4, cache_dir=None):
        super().__init__()

        # if isinstance(train_urls, types.GeneratorType):
        #     train_urls = list(train_urls)
        # if isinstance(val_urls, types.GeneratorType):
        #     val_urls = list(val_urls)
        self.train_urls = train_urls
        self.val_urls = val_urls

        if train_size is None:
            # assume the size of each shard is encoded in the filename as ..._{size}.tar
            train_size = sum([int(url.rstrip('.tar').split('_')[-1]) for url in train_urls])
        if val_size is None:
            val_size = sum([int(url.rstrip('.tar').split('_')[-1]) for url in val_urls])

        self.train_size = train_size
        self.val_size = val_size

        self.label_cols = label_cols

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.cache_dir = cache_dir


        logging.info(f'Creating webdatamodule with WORLD_SIZE: {os.environ.get("WORLD_SIZE")}, RANK: {os.environ.get("RANK")}')

        print("train_urls = ", self.train_urls)
        print("val_urls = ", self.val_urls)
        print("train_size = ", self.train_size)
        print("val_size = ", self.val_size)
        print("batch_size", self.batch_size, "num_workers", self.num_workers)

    def make_image_transform(self, mode="train"):
        # if mode == "train":
        # elif mode == "val":

        augmentation_transform = default_transforms()  # A.Compose object
        def do_transform(img):
            return np.transpose(augmentation_transform(image=np.array(img))["image"], axes=[2, 0, 1]).astype(np.float32)
        return do_transform

    def make_label_transform(self):
        if self.label_cols is not None:
            def label_transform(label_dict):
                return torch.from_numpy(np.array([label_dict.get(col, 0) for col in self.label_cols]))
            return label_transform
        else:
            return identity  # do nothing
    

    def make_loader(self, urls, mode="train"):
        if mode == "train":
            dataset_size = self.train_size
            shuffle = 5000
        elif mode == "val":
            dataset_size = self.val_size
            shuffle = 0

        transform_image = self.make_image_transform(mode=mode)

        transform_label = self.make_label_transform()

        dataset = (
            # https://webdataset.github.io/webdataset/multinode/ 
            # WDS 'knows' which worker it is running on and selects a subset of urls accordingly
            wds.WebDataset(urls, cache_dir=self.cache_dir, shardshuffle=shuffle>0, nodesplitter=nodesplitter_func)
            .shuffle(shuffle)
            .decode("rgb")
            .to_tuple('image.jpg', 'labels.json')
            .map_tuple(transform_image, transform_label)
            # torch collate stacks dicts nicely while webdataset only lists them
            # so use the torch collate instead
            .batched(self.batch_size, torch.utils.data.default_collate, partial=False) 
            # .repeat(5)
        )

        # from itertools import islice
        # for batch in islice(dataset, 0, 3):
        #     images, labels = batch
        #     # print(len(sample))
        #     print(images.shape)
        #     print(len(labels))  # list of dicts
        #     # exit()

        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # already batched
            shuffle=False,  # already shuffled
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

        # print('sampling')
        # for sample in islice(loader, 0, 3):
        #     images, labels = sample
        #     print(images.shape)
        #     print(len(labels))  # list of dicts
            # exit()

        loader.length = dataset_size // self.batch_size

        # temp hack instead
        assert dataset_size % self.batch_size == 0  
        # if mode == "train":
            # ensure same number of batches in all clients
            # loader = loader.ddp_equalize(dataset_size // self.batch_size)
            # print("# loader length", len(loader))

        return loader

    def train_dataloader(self):
        return self.make_loader(self.train_urls, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")

    # @staticmethod
    # def add_loader_specific_args(parser):
    #     parser.add_argument("-b", "--batch-size", type=int, default=128)
    #     parser.add_argument("--workers", type=int, default=6)
    #     parser.add_argument("--bucket", default="./shards")
    #     parser.add_argument("--shards", default="imagenet-train-{000000..001281}.tar")
    #     parser.add_argument("--valshards", default="imagenet-val-{000000..000006}.tar")
    #     return parser

# def nodesplitter_func(urls): # SimpleShardList
#     # print(urls)
#     try:
#         node_id, node_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
#         urls_to_use = list(urls)[node_id::node_count]
#         logging.info(f'id: {node_id}, of count {node_count}. \nURLS: {len(urls_to_use)} of {len(urls)} ({urls_to_use})\n\n')
#         return urls_to_use
#     except RuntimeError:
#         # print('Distributed not initialised. Hopefully single node.')
#         return urls

def identity(x):
    return x

def nodesplitter_func(urls):
    # num_urls = len(list(urls.copy()))
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


# def split_by_worker(urls):
#     rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
#     if num_workers > 1:
#         logging.info(f'Slicing urls for rank {rank}, world_size {world_size}, worker {worker}')
#         for s in islice(urls, worker, None, num_workers):
#             yield s
#     else:
#         logging.warning('only one worker?!')
#         for s in urls:
#             yield s
