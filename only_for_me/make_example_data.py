import logging
import shutil
import os

import pandas as pd


def get_new_path(current_path):
    return 'data/example_images/basic/{}'.format(os.path.basename(current_path))


if __name__ == '__main__':
    """
    Used to create the basic ring catalog dataset for finetune_minimal.py
    This won't be any use unless you're me. I'm adding to the repo for reproducibility/completeness/personal notes.
    """

    file_format = 'png'
    ring_catalog = pd.read_csv('data/ring_catalog_with_morph.csv') 
    # this should point to the location on my laptop from which to copy each galaxy. Will not be saved.
    ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].str.replace('/media/walml/beta1/decals/png_native/dr5', '/Volumes/beta/decals/png_native/dr5')

    # apply selection cuts
    feat = ring_catalog['smooth-or-featured_featured-or-disk_fraction'] > 0.6
    face = ring_catalog['disk-edge-on_no_fraction'] > 0.75
    not_spiral = ring_catalog['has-spiral-arms_no_fraction'] > 0.5
    ring_catalog = ring_catalog[feat & face & not_spiral].reset_index(drop=True)
    logging.info('Labels after selection cuts: \n{}'.format(pd.value_counts(ring_catalog['ring'])))

    rings = ring_catalog.query('ring == 1')
    not_rings = ring_catalog.query('ring == 0').sample(len(rings), replace=False, random_state=1)
    ring_catalog_balanced = pd.concat([rings, not_rings]).reset_index()

    # for _, row in ring_catalog_balanced.iterrows():
    #     shutil.copyfile(row['local_png_loc'], get_new_path(row['local_png_loc']))

    # print(ring_catalog_balanced['local_png_loc'][0])
    ring_catalog_balanced['local_png_loc'] = ring_catalog_balanced['local_png_loc'].apply(get_new_path)
    # print(ring_catalog_balanced['local_png_loc'][0])
    # exit()
    ring_catalog_balanced['copied_okay'] = ring_catalog_balanced['local_png_loc'].apply(os.path.isfile)
    if not all(ring_catalog_balanced['copied_okay']):
        raise FileExistsError(ring_catalog_balanced[~ring_catalog_balanced['local_png_loc']][0])

    relevant_cols = ['iauname', 'local_png_loc', 'ring', 'smooth-or-featured_featured-or-disk_fraction', 'disk-edge-on_no_fraction', 'has-spiral-arms_no_fraction']
    ring_catalog_balanced[relevant_cols].to_csv('data/example_ring_catalog.csv', index=False)
