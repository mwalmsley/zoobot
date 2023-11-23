import logging
from typing import List
import os
import cv2
import json
from itertools import islice
import glob


import albumentations as A

import tqdm
import numpy as np
import pandas as pd
from PIL import Image  # necessary to avoid PIL.Image error assumption in web_datasets

from galaxy_datasets import gz2
from galaxy_datasets.transforms import default_transforms

import webdataset as wds

import zoobot.pytorch.datasets.webdatamodule as webdatamodule


def make_mock_wds(save_dir: str, label_cols: List, n_shards: int, shard_size: int):
    counter = 0
    shards = [os.path.join(save_dir, f'mock_shard_{shard_n}_{shard_size}.tar') for shard_n in range(n_shards)]
    for shard in shards:
        sink = wds.TarWriter(shard)
        for galaxy_n in range(shard_size):
            data = {
                "__key__": f'id_{galaxy_n}',
                "image.jpg": (np.random.rand(424, 424)*255.).astype(np.uint8),
                "labels.json": json.dumps(dict(zip(label_cols, [np.random.randint(low=0, high=10) for _ in range(len(label_cols))])))
            }
            sink.write(data)
            counter += 1
    print(counter)
    return shards



def df_to_wds(df: pd.DataFrame, label_cols, save_loc: str, n_shards: int, sparse_label_df=None):

    assert '.tar' in save_loc
    df['id_str'] = df['id_str'].astype(str).str.replace('.', '_')
    if sparse_label_df is not None:
        logging.info(f'Using sparse label df: {len(sparse_label_df)}')
    shard_dfs = np.array_split(df, n_shards)
    logging.info(f'shards: {len(shard_dfs)}. Shard size: {len(shard_dfs[0])}')

    transforms_to_apply = [
        # below, for 224px fast training fast augs setup
        # A.Resize(
        #     height=350,  # now more aggressive, 65% crop effectively
        #     width=350,  # now more aggressive, 65% crop effectively
        #     interpolation=cv2.INTER_AREA  # slow and good interpolation
        # ),
        # A.CenterCrop(
        #     height=224,
        #     width=224,
        #     always_apply=True
        # ),
        # below, for standard training default augs
        # small boundary trim and then resize expecting further 224px crop
        # we want 0.7-0.8 effective crop
        # in augs that could be 0.x-1.0, and here a pre-crop to 0.8 i.e. 340px
        # but this would change the centering
        # let's stick to small boundary crop and 0.75-0.85 in augs
        A.CenterCrop(
            height=400,
            width=400,
            always_apply=True
        ),
        A.Resize(
            height=300,
            width=300,
            interpolation=cv2.INTER_AREA  # slow and good interpolation
        )
    ]
    transform = A.Compose(transforms_to_apply)
    # transform = None

    for shard_n, shard_df in tqdm.tqdm(enumerate(shard_dfs), total=len(shard_dfs)):
        if sparse_label_df is not None:
            shard_df = pd.merge(shard_df, sparse_label_df, how='left', validate='one_to_one', suffixes=('', '_badlabelmerge'))  # auto-merge
        shard_save_loc = save_loc.replace('.tar', f'_{shard_n}_{len(shard_df)}.tar')
        logging.info(shard_save_loc)
        sink = wds.TarWriter(shard_save_loc)
        for _, galaxy in shard_df.iterrows():
            sink.write(galaxy_to_wds(galaxy, label_cols, transform=transform))
        sink.close()


def galaxy_to_wds(galaxy: pd.Series, label_cols, transform=None):

    im = cv2.imread(galaxy['file_loc'])
    # cv2 loads BGR for 'history', fix
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    # if central_crop is not None:
    #     width, height, _ = im.shape
    #     # assert width == height, (width, height)
    #     mid = int(width/2)
    #     half_central_crop = int(central_crop/2)
    #     low_edge, high_edge = mid - half_central_crop, mid + half_central_crop
    #     im = im[low_edge:high_edge, low_edge:high_edge]
    #     assert im.shape == (central_crop, central_crop, 3)

    # apply albumentations
    if transform is not None:
        im = transform(image=im)['image']

    labels = json.dumps(galaxy[label_cols].to_dict())
    id_str = str(galaxy['id_str'])
    return {
        "__key__": id_str,
        "image.jpg": im,
        "labels.json": labels
    }

# just for debugging
def load_wds_directly(wds_loc, max_to_load=3):

    dataset = wds.WebDataset(wds_loc) \
    .decode("rgb")

    if max_to_load is not None:
        sample_iterator = islice(dataset, 0, max_to_load)
    else:
        sample_iterator = dataset
    for sample in sample_iterator:
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
def load_wds_with_webdatamodule(save_loc, label_cols, batch_size=16, max_to_load=3):
    wdm = webdatamodule.WebDataModule(
        train_urls=save_loc,
        val_urls=save_loc,  # not used
        # train_size=len(train_catalog),
        # val_size=0,
        label_cols=label_cols,
        num_workers=1,
        batch_size=batch_size
    )
    wdm.setup('fit')

    if max_to_load is not None:
        sample_iterator =islice(wdm.train_dataloader(), 0, max_to_load)
    else:
        sample_iterator = wdm.train_dataloader()
    for sample in sample_iterator:
        images, labels = sample
        logging.info(images.shape)
        # logging.info(len(labels))  # list of dicts
        logging.info(labels.shape)


def identity(x):
    # no lambda to be pickleable
    return x

if __name__ == '__main__':

    save_dir = '/home/walml/repos/temp'
    from galaxy_datasets.shared import label_metadata
    label_cols = label_metadata.decals_all_campaigns_ortho_label_cols

    make_mock_wds(save_dir, label_cols, n_shards=4, shard_size=512)
