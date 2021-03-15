import logging
import shutil
import os

import pandas as pd


def get_new_path(current_path):
    return 'data/example_images/{}'.format(os.path.basename(current_path))


if __name__ == '__main__':


    file_format = 'png'
    ring_catalog = pd.read_csv('data/ring_catalog_with_morph.csv')  # TODO change path
    ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].str.replace('/media/walml/beta1/decals/png_native/dr5', '/raid/scratch/walml/galaxy_zoo/decals/png')

    # apply selection cuts
    feat = ring_catalog['smooth-or-featured_featured-or-disk_fraction'] > 0.6
    face = ring_catalog['disk-edge-on_no_fraction'] > 0.75
    not_spiral = ring_catalog['has-spiral-arms_no_fraction'] > 0.5
    ring_catalog = ring_catalog[feat & face & not_spiral].reset_index(drop=True)
    logging.info('Labels after selection cuts: \n{}'.format(pd.value_counts(ring_catalog['ring'])))

    rings = ring_catalog.query('ring == 1')
    not_rings = ring_catalog.query('ring == 0').sample(len(rings), replace=False)
    ring_catalog_balanced = pd.concat([rings, not_rings]).reset_index()

    # for _, row in ring_catalog_balanced.iterrows():
    #     shutil.copyfile(row['local_png_loc'], get_new_path(row['local_png_loc']))

    ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].apply(get_new_path)

    ring_catalog_balanced.to_csv('data/example_ring_catalog.csv', index=False)
