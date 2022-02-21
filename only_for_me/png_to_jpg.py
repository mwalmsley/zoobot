import os
import logging
from multiprocessing import Pool

from PIL import Image

import pandas as pd


def to_jpg(image_loc):
    assert os.path.isfile(image_loc)
    jpg_loc = image_loc.replace('/png/', '/jpeg/').replace('.png', '.jpeg')
    if not os.path.isfile(jpg_loc):
        target_dir = os.path.dirname(jpg_loc)
        if not os.path.isdir(target_dir):
            logging.info('Making dir: {}'.format(target_dir))
            os.makedirs(target_dir, exist_ok=True)  # recursive, okay to exist if another thread JUST made it
        Image.open(image_loc).save(jpg_loc, quality=90)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/labelled_catalog.csv'
    # catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr8/labelled_catalog.csv'
    catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr12/labelled_catalog.csv'
    catalog = pd.read_csv(catalog_loc)
    catalog['file_loc'] = catalog['file_loc'].str.replace('/raid/scratch',  '/share/nas2')
    catalog['file_loc'] = catalog['file_loc'].str.replace('/dr8_downloader/',  '/dr8/')
    catalog['file_loc'] = catalog['file_loc'].str.replace('.jpeg', '.png')  # they are currently all pngs
    # png_paths = list(catalog['file_loc'].sample(100, random_state=42))
    png_paths = list(catalog['file_loc'])

    pool = Pool(processes=os.cpu_count())
    for _ in pool.imap_unordered(to_jpg, png_paths, chunksize=1000):
        pass

    logging.info('Saving complete - exiting gracefully')
