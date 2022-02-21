import os
import logging
from multiprocessing import Pool

from PIL import Image

import pandas as pd


def to_jpg(image_loc):
    assert os.path.isfile(image_loc)
    jpg_loc = image_loc.replace('/png/', '/jpeg/').replace('.png', '.jpeg')
    print(jpg_loc)
    # if not os.path.exists(jpg_loc):
    #     if not os.path.isdir(jpg_loc):
    #         try:
    #             os.mkdirs(jpg_loc)  # recursive
    #         except IsADirectoryError:
    #             pass  # anther thread already made it
    #     Image.open(image_loc).save(jpg_loc)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/labelled_catalog.csv'
    catalog = pd.read_csv(catalog_loc)
    catalog['file_loc'] = catalog['file_loc'].str.replace('/raid/scratch',  '/share/nas2')
    catalog['file_loc'] = catalog['file_loc'].str.replace('/dr8_downloader/',  '/dr8/')
    catalog['file_loc'] = catalog['file_loc'].str.replace('.jpeg', '.png')  # they are currently all pngs

    pool = Pool(processes=os.cpu_count())

    subset = list(catalog['file_loc'].sample(100, random_state=42))
    pool.imap_unordered(to_jpg, subset, chunksize=1000)

    logging.info('Saving complete - exiting gracefully')
