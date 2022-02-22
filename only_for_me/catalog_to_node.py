import os
import logging
from multiprocessing import Pool
from random import random

from PIL import Image

import pandas as pd


def move_to_node(catalog: pd.DataFrame, new_base_folder='/state/partition1'):

    old_locs = catalog['file_loc']
    new_locs = old_locs.str.replace('/share/nas2', new_base_folder)  # will keep the walml/galaxy_zoo/decals/...

    pool = Pool(processes=os.cpu_count())
    for _ in pool.imap_unordered(move_image, zip(old_locs, new_locs), chunksize=1000):
        pass

    logging.info('Saving complete - exiting gracefully')


def move_image(loc_tuple):
    old_loc, new_loc = loc_tuple
    assert os.path.isfile(old_loc)
    if not os.path.isfile(new_loc):
        target_dir = os.path.dirname(new_loc)
        if not os.path.isdir(target_dir):
            logging.info('Making dir: {}'.format(target_dir))
            os.makedirs(target_dir, exist_ok=True)  # recursive, okay to exist if another thread JUST made it
        Image.open(old_loc).save(new_loc, quality=90)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/labelled_catalog.csv'
    # catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr8/labelled_catalog.csv'
    catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr12/labelled_catalog.csv'
    catalog = pd.read_csv(catalog_loc)
    catalog['file_loc'] = catalog['file_loc'].str.replace('/raid/scratch',  '/share/nas2')
    catalog['file_loc'] = catalog['file_loc'].str.replace('/dr8_downloader/',  '/dr8/')
    catalog['file_loc'] = catalog['file_loc'].str.replace('.jpeg', '.png')  # they are currently all pngs
    catalog = catalog.sample(100, random_state=42)
    # png_paths = list(catalog['file_loc'].sample(100, random_state=42))
    # png_paths = list(catalog['file_loc'])
    move_to_node(catalog)
