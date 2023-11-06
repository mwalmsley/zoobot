import logging
import os
import shutil
import sys
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

def galaxy_to_wds(galaxy: pd.Series, label_cols):

    im = cv2.imread(galaxy['file_loc'])
    # cv2 loads BGR for 'history', fix
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    labels = json.dumps(galaxy[label_cols].to_dict())
    id_str = str(galaxy['id_str'])
    # print(id_str)
    return {
        "__key__": id_str,
        "image.jpg": im,
        "labels.json": labels
    }

def df_to_wds(df: pd.DataFrame, label_cols, save_loc, n_shards):
    df['id_str'] = df['id_str'].astype(str).str.replace('.', '_')

    shard_dfs = np.array_split(df, n_shards)
    print('shards: ', len(shard_dfs))
    print('shard size: ', len(shard_dfs[0]))
    for shard_n, shard_df in tqdm.tqdm(enumerate(shard_dfs), total=len(shard_dfs)):
        shard_save_loc = save_loc.replace('.tar', f'_{shard_n}_{len(shard_df)}.tar')
        print(shard_save_loc)
        sink = wds.TarWriter(shard_save_loc)
        for index, galaxy in shard_df.iterrows():
            sink.write(galaxy_to_wds(galaxy, label_cols))
        sink.close()

def check_wds(wds_loc):

    dataset = wds.WebDataset(wds_loc) \
    .decode("rgb")

    for sample in islice(dataset, 0, 3):
        print(sample['__key__'])     
        print(sample['image.jpg'].shape)  # .decode(jpg) converts to decoded to 0-1 RGB float, was 0-255
        print(type(sample['labels.json']))  # automatically decoded

def identity(x):
    # no lambda to be pickleable
    return x


def load_wds(wds_loc):

    augmentation_transform = default_transforms()  # A.Compose object
    def do_transform(img):
        return np.transpose(augmentation_transform(image=np.array(img))["image"], axes=[2, 0, 1]).astype(np.float32)

    dataset = wds.WebDataset(wds_loc) \
        .decode("rgb") \
        .to_tuple('image.jpg', 'labels.json') \
        .map_tuple(do_transform, identity)
    
    for sample in islice(dataset, 0, 3):
        print(sample[0].shape)
        print(sample[1])


def main():

    train_catalog, _ = gz2(root='/home/walml/repos/zoobot/only_for_me/narval/temp', download=True, train=True)
    # print(len(train_catalog))
    # exit()
    divisor = 4096
    batches = len(train_catalog) // divisor
    print(batches)
    train_catalog = train_catalog[:batches*divisor]
    print(len(train_catalog))
    label_cols = label_metadata.gz2_ortho_label_cols

    save_loc = "/home/walml/repos/zoobot/only_for_me/narval/gz2/gz2_train.tar"
    
    df_to_wds(train_catalog, label_cols, save_loc, n_shards=batches)

    # check_wds(save_loc)

    # load_wds(save_loc)

    import zoobot.pytorch.training.webdatamodule as webdatamodule

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
        print(images.shape)
        # print(len(labels))  # list of dicts
        print(labels)
        exit()



if __name__ == '__main__':

    main()

