import logging
import os
import cv2
import json
from itertools import islice
import glob

import tqdm
import numpy as np
import pandas as pd
from PIL import Image  # necessary to avoid PIL.Image error assumption in web_datasets

from galaxy_datasets.shared import label_metadata
from galaxy_datasets import gz2
from galaxy_datasets.transforms import default_transforms
from galaxy_datasets.pytorch import galaxy_dataset

import webdataset as wds

import zoobot.pytorch.datasets.webdatamodule as webdatamodule


def make_mock_wds(save_dir, label_cols, n_shards, shard_size):
    shards = [os.path.join(save_dir, f'mock_shard_{shard_n}') for shard_n in range(n_shards)]
    for shard in shards:
        sink = wds.TarWriter(shard)
        for galaxy_n in range(shard_size):
            data = {
                "__key__": f'id_{galaxy_n}',
                "image.jpg": (np.random.rand(424, 424)*255.).astype(np.uint8),
                "labels.json": json.dumps(dict(zip(label_cols, [np.random.randint(low=0, high=10) for _ in range(len(label_cols))])))
            }
            sink.write(data)



def df_to_wds(df: pd.DataFrame, label_cols, save_loc, n_shards):
    df['id_str'] = df['id_str'].astype(str).str.replace('.', '_')

    shard_dfs = np.array_split(df, n_shards)
    logging.info('shards: ', len(shard_dfs))
    logging.info('shard size: ', len(shard_dfs[0]))
    for shard_n, shard_df in tqdm.tqdm(enumerate(shard_dfs), total=len(shard_dfs)):
        shard_save_loc = save_loc.replace('.tar', f'_{shard_n}_{len(shard_df)}.tar')
        logging.info(shard_save_loc)
        sink = wds.TarWriter(shard_save_loc)
        for _, galaxy in shard_df.iterrows():
            sink.write(galaxy_to_wds(galaxy, label_cols))
        sink.close()


def galaxy_to_wds(galaxy: pd.Series, label_cols):

    im = cv2.imread(galaxy['file_loc'])
    # cv2 loads BGR for 'history', fix
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    labels = json.dumps(galaxy[label_cols].to_dict())
    id_str = str(galaxy['id_str'])
    return {
        "__key__": id_str,
        "image.jpg": im,
        "labels.json": labels
    }

# just for debugging
def load_wds_directly(wds_loc):

    dataset = wds.WebDataset(wds_loc) \
    .decode("rgb")

    for sample in islice(dataset, 0, 3):
        logging.info(sample['__key__'])     
        logging.info(sample['image.jpg'].shape)  # .decode(jpg) converts to decoded to 0-1 RGB float, was 0-255
        logging.info(type(sample['labels.json']))  # automatically decoded


# just for debugging
def load_wds_with_augmentation(wds_loc):

    augmentation_transform = default_transforms()  # A.Compose object
    def do_transform(img):
        return np.transpose(augmentation_transform(image=np.array(img))["image"], axes=[2, 0, 1]).astype(np.float32)

    dataset = wds.WebDataset(wds_loc) \
        .decode("rgb") \
        .to_tuple('image.jpg', 'labels.json') \
        .map_tuple(do_transform, identity)
    
    for sample in islice(dataset, 0, 3):
        logging.info(sample[0].shape)
        logging.info(sample[1])

# just for debugging
def load_wds_with_webdatamodule(save_loc, label_cols):
    wdm = webdatamodule.WebDataModule(
        train_urls=glob.glob(save_loc.replace('.tar', '_*.tar')),
        val_urls=[],
        # train_size=len(train_catalog),
        # val_size=0,
        label_cols=label_cols,
        num_workers=1
    )
    wdm.setup('fit')

    for sample in islice(wdm.train_dataloader(), 0, 3):
        images, labels = sample
        logging.info(images.shape)
        # logging.info(len(labels))  # list of dicts
        logging.info(labels)


def identity(x):
    # no lambda to be pickleable
    return x

