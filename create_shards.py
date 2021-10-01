"""
Save catalog columns and images to tfrecord shards.
Allowed to assume:
- Each catalog entry has an image under `file_loc`. Must be .png file (easy to extend to others if you need, submit a PR)
- Each catalog entry has an identifier under `id_str`
"""
import argparse
import os
import shutil
import logging
import json
import time
import glob
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from zoobot import label_metadata
from zoobot.data_utils import catalog_to_tfrecord, checks
from zoobot.estimators import preprocess


class ShardConfig():
    """
    Assumes that you have:
    - a directory of fits files  (e.g. `fits_native`)
    - a catalog of files, with file locations under the column 'fits_loc' (relative to repo root)

    Checks that catalog paths match real fits files
    Creates unlabelled shards and single shard of labelled subjects
    Creates sqlite database describing what's in those shards

    JSON serializable for later loading
    """

    def __init__(
        self,
        shard_dir,
        size,
        shard_size=4096
        ):
        """
        Args:
            shard_dir (str): directory into which to save shards
            size (int, optional): Defaults to 128. Resolution to save fits to tfrecord (i.e. width in pixels)
            final_size (int, optional): Defaults to 64. Resolution to load from tfrecord into model
            shard_size (int, optional): Defaults to 4096. Galaxies per shard.
        """
        self.size = size
        self.shard_size = shard_size
        self.shard_dir = shard_dir

        self.channels = 3  # save 3-band image to tfrecord. Augmented later by model input func.

        # paths for fixed tfrecords for initial training and (permanent) evaluation
        self.train_dir = os.path.join(self.shard_dir, 'train_shards') 
        self.eval_dir = os.path.join(self.shard_dir, 'eval_shards')

        # paths for catalogs. Used to look up .fits locations during active learning.
        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        self.config_save_loc = os.path.join(self.shard_dir, 'shard_config.json')


    def train_tfrecord_locs(self):
        return [os.path.join(self.train_dir, loc) for loc in os.listdir(self.train_dir)
            if loc.endswith('.tfrecord')]


    def eval_tfrecord_locs(self):
        return [os.path.join(self.eval_dir, loc) for loc in os.listdir(self.eval_dir)
            if loc.endswith('.tfrecord')]


    def prepare_shards(self, labelled_catalog, unlabelled_catalog, train_test_fraction, labelled_columns_to_save: List):
        """
        Save the images in labelled_catalog and unlabelled_catalog to tfrecord shards.

        Assumes:
        - Each catalog entry has an image under `file_loc`. Must be .png file (easy to extend to others if you need, submit a PR)
        - Each catalog entry has an identifier under `id_str`

        Will split galaxies in ``labelled_catalog`` into train and test subsets according to ``train_test_fraction``,
        and then save to self.train_dir and self.eval_dir respectively.

        Galaxies in ``unlabelled_catalog`` will be saved to self.shard_dir.

        Use ``labelled_columns_to_save`` to include data alongside each image in the final tfrecords for the labelled catalog *only*.
        Must include 'id_str', and should likely also include the columns with labels
        
        Args:
            labelled_catalog (pd.DataFrame): labelled galaxies, including file_loc column
            unlabelled_catalog (pd.DataFrame): unlabelled galaxies, including file_loc column
            train_test_fraction (float): fraction of labelled catalog to use as training data
            labelled_columns_to_save list: Save catalog cols to tfrecord, under same name. 
        """

        # personal file manipulation, because my catalogs are old. Just make sure file_loc actually points to the files in the first place...
        labelled_catalog['file_loc'] = labelled_catalog['file_loc'].str.replace('/media/walml/beta/decals/png_native/dr5', '/share/nas/walml/galaxy_zoo/decals/dr5/png')
        unlabelled_catalog['file_loc'] = unlabelled_catalog['file_loc'].str.replace('/media/walml/beta/decals/png_native/dr5', '/share/nas/walml/galaxy_zoo/decals/dr5/png')

        assert 'id_str' in labelled_columns_to_save

        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)
        os.mkdir(self.train_dir)
        os.mkdir(self.eval_dir)

        # check that file paths resolve correctly
        logging.info('Example file locs: \n{}'.format(labelled_catalog['file_loc'][:3].values))
        checks.check_no_missing_files(labelled_catalog['file_loc'], max_to_check=2000)
        checks.check_no_missing_files(unlabelled_catalog['file_loc'], max_to_check=2000)

        # assume the catalog is true, don't modify halfway through
        logging.info('\nLabelled subjects: {}'.format(len(labelled_catalog)))
        logging.info('Unlabelled subjects: {}'.format(len(unlabelled_catalog)))
        logging.info(f'Train-test fraction: {train_test_fraction}')
        labelled_catalog.to_csv(self.labelled_catalog_loc)
        unlabelled_catalog.to_csv(self.unlabelled_catalog_loc)

        train_df = labelled_catalog
        eval_df = unlabelled_catalog
        logging.info('\nTraining subjects: {}'.format(len(train_df)))
        logging.info('Eval subjects: {}'.format(len(eval_df)))
        if len(train_df) < len(eval_df):
            print('More eval subjects than training subjects - is this intended?')
        train_df.to_csv(os.path.join(self.train_dir, 'train_df.csv'))
        eval_df.to_csv(os.path.join(self.eval_dir, 'eval_df.csv'))

        logging.info('Writing {} train and {} test galaxies to shards'.format(len(train_df), len(eval_df)))
        for (df, save_dir) in [(train_df, self.train_dir), (eval_df, self.eval_dir)]:
            write_catalog_to_tfrecord_shards(
                df,
                img_size=self.size,
                columns_to_save=labelled_columns_to_save,
                save_dir=save_dir,
                shard_size=self.shard_size
            )

        logging.info('Writing {} unlabelled galaxies to shards (optional)'.format(len(unlabelled_catalog)))
        columns_to_save = ['id_str']
        write_catalog_to_tfrecord_shards(
            unlabelled_catalog,
            self.size,
            columns_to_save,
            self.shard_dir,
            self.shard_size
        )

        assert self.ready()

        # serialized for later/logs
        self.write()


    def ready(self):
        assert os.path.isdir(self.shard_dir)
        assert os.path.isdir(self.train_dir)
        assert os.path.isdir(self.eval_dir)
        assert os.path.isfile(self.labelled_catalog_loc)
        assert os.path.isfile(self.unlabelled_catalog_loc)
        return True

    def to_dict(self):
        return {
            'size': self.size,
            'shard_size': self.shard_size,
            'shard_dir': self.shard_dir,
            'channels': self.channels,
            'train_dir': self.train_dir,
            'eval_dir': self.eval_dir,
            'labelled_catalog_loc': self.labelled_catalog_loc,
            'unlabelled_catalog_loc': self.unlabelled_catalog_loc,
            'config_save_loc': self.config_save_loc
        }

    def write(self):
        with open(self.config_save_loc, 'w+') as f:
            json.dump(self.to_dict(), f)


def load_shard_config(shard_config_loc: str):
    # shards to use
    shard_config = load_shard_config_naive(shard_config_loc)
    # update shard paths in case shard dir was moved since creation
    new_shard_dir = os.path.dirname(shard_config_loc)
    shard_config.shard_dir = new_shard_dir
    attrs = [
        'train_dir',
        'eval_dir',
        'labelled_catalog_loc',
        'unlabelled_catalog_loc',
        'config_save_loc'
    ]
    for attr in attrs:
        old_loc = getattr(shard_config, attr)
        new_loc = os.path.join(new_shard_dir, os.path.split(old_loc)[-1])
        logging.info('Was {}, now {}'.format(attr, new_loc))
        setattr(shard_config, attr, new_loc)
    return shard_config


def load_shard_config_naive(shard_config_loc):
    with open(shard_config_loc, 'r') as f:
        shard_config_dict = json.load(f)
    return ShardConfig(**shard_config_dict)


def get_train_test_fraction(total_size, eval_size):
    assert eval_size < total_size
    train_test_fraction = (total_size - int(eval_size))/total_size
    logging.info('Train test fraction: {}'.format(train_test_fraction))
    return train_test_fraction


def write_catalog_to_tfrecord_shards(df: pd.DataFrame, img_size, columns_to_save, save_dir, shard_size=1000):
    """Write galaxy catalog of id_str and file_loc across many tfrecords, and record in db.
    Useful to quickly load images for repeated predictions.

    Args:
        df (pd.DataFrame): Galaxy catalog with 'id_str' and 'fits_loc' fields
        db (sqlite3.Connection): database with `catalog` table to record df id_col and fits_loc
        img_size (int): height/width dimension of image matrix to rescale and save to tfrecords
        columns_to_save (list): Catalog data to save with each subject. Names will match tfrecord.
        save_dir (str): disk directory path into which to save tfrecords
        shard_size (int, optional): Defaults to 1000. Max subjects per shard. Final shard has less.
    """
    assert not df.empty
    assert 'id_str' in columns_to_save
    if not all(column in df.columns.values for column in columns_to_save):
        raise IndexError('Columns not found in df: {}'.format(set(columns_to_save) - set(df.columns.values)))

    df = df.copy().sample(frac=1).reset_index(drop=True)  # shuffle - note that this means acquired subjects will be in random order
    # split into shards
    shard_n = 0
    n_shards = (len(df) // shard_size) + 1
    df_shards = [df.iloc[n * shard_size:(n + 1) * shard_size] for n in range(n_shards)]

    logging.info(f'Writing shards to {save_dir}')
    for shard_n, df_shard in tqdm(enumerate(df_shards), total=len(df_shards), unit='shards'):
        save_loc = os.path.join(save_dir, 's{}_shard_{}.tfrecord'.format(img_size, shard_n))
        catalog_to_tfrecord.write_image_df_to_tfrecord(
            df_shard, 
            save_loc,
            img_size,
            columns_to_save,
            reader=catalog_to_tfrecord.get_reader(df['file_loc'])
        )



if __name__ == '__main__':

    """
    Some example commands:

    DECALS:

        (debugging)
        python create_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2/unlabelled_catalog.csv --eval-size 100 --shard-dir=data/decals/shards/decals_debug --max-labelled 500 --max-unlabelled=300 --img-size 32

        (the actual commands used for gz decals: debug above, full below)
        python create_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired/unlabelled_catalog.csv --eval-size 100 --shard-dir=data/decals/shards/decals_debug --max-labelled 500 --max-unlabelled=300 --img-size 32
        python create_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2_arc/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2_arc/unlabelled_catalog.csv --eval-size 10000 --shard-dir=data/decals/shards/all_2p5_unfiltered_n2  --img-size 300

    GZ2:
        python create_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5/unlabelled_catalog.csv --eval-size 1000 --shard-dir=data/gz2/shards/all_featp5_facep5_256 --img-size 256

    """

    parser = argparse.ArgumentParser(description='Make shards')

    # you should have already made these catalogs for your dataset
    parser.add_argument('--labelled-catalog', dest='labelled_catalog_loc', type=str,
                    help='Path to csv catalog of previous labels and file_loc, for shards')
    parser.add_argument('--unlabelled-catalog', dest='unlabelled_catalog_loc', type=str, default='',
                help='Path to csv catalog of previous labels and file_loc, for shards. Optional - skip (recommended) if all galaxies are labelled.')

    parser.add_argument('--eval-size', dest='eval_size', type=int,
        help='Split labelled galaxies into train/test, with this many test galaxies (e.g. 5000)')

    # Write catalog to shards (tfrecords as catalog chunks) here
    parser.add_argument('--shard-dir', dest='shard_dir', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--max-unlabelled', dest='max_unlabelled', type=int,
                    help='Max unlabelled galaxies (for debugging/speed')
    parser.add_argument('--max-labelled', dest='max_labelled', type=int,
                    help='Max labelled galaxies (for debugging/speed')
    parser.add_argument('--img-size', dest='size', type=int,
                    help='Size at which to save images (before any augmentations). 300 for DECaLS paper.')

    args = parser.parse_args()

    # log_loc = 'make_shards_{}.log'.format(time.time())
    logging.basicConfig(
        # filename=log_loc,
        # filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )

    logging.info('Using GZ DECaLS label schema by default - see create_shards.py for more options, or to add your own')
    # label_cols = label_metadata.decals_partial_label_cols
    label_cols = label_metadata.decals_label_cols
    # label_cols = label_metadata.gz2_partial_label_cols
    # label_cols = label_metadata.gz2_label_cols

    # labels will always be floats, int conversion confuses tf.data
    dtypes = dict(zip(label_cols, [float for _ in label_cols]))
    dtypes['id_str'] = str
    labelled_catalog = pd.read_csv(args.labelled_catalog_loc, dtype=dtypes)
    if args.unlabelled_catalog_loc is not '':
        unlabelled_catalog = pd.read_csv(args.unlabelled_catalog_loc, dtype=dtypes)
    else:
        unlabelled_catalog = pd.DataFrame()  # empty dataframe, for consistency only

    # limit catalogs to random subsets
    if args.max_labelled:
        labelled_catalog = labelled_catalog.sample(len(labelled_catalog))[:args.max_labelled]
    if args.max_unlabelled:  
        unlabelled_catalog = unlabelled_catalog.sample(len(unlabelled_catalog))[:args.max_unlabelled]

    logging.info('Labelled: {}, unlabelled: {}'.format(len(labelled_catalog), len(unlabelled_catalog)))

    # in memory for now, but will be serialized for later/logs
    train_test_fraction = get_train_test_fraction(len(labelled_catalog), args.eval_size)

    labelled_columns_to_save = ['id_str'] + label_cols
    logging.info('Saving columns for labelled galaxies: \n{}'.format(labelled_columns_to_save))

    shard_config = ShardConfig(shard_dir=args.shard_dir, size=args.size)

    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog,
        train_test_fraction=train_test_fraction,
        labelled_columns_to_save=labelled_columns_to_save
    )
