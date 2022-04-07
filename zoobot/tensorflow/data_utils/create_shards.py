"""
Save catalog columns and images to tfrecord shards.
Allowed to assume:
- Each catalog entry has an image under `file_loc`. Must be .png file (easy to extend to others if you need, submit a PR)
- Each catalog entry has an identifier under `id_str`

For a full working example, see `zoobot/tensorflow/examples/decals_dr5_to_shards.py`

"""

import os
import shutil
import logging
import json
from typing import List

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from zoobot.tensorflow.data_utils import catalog_to_tfrecord, checks


class ShardConfig():
    """
    Assumes that you have:
    - a directory of png files  (e.g. `png_native`)
    - a catalog of files, with file locations under the column 'png_loc' (relative to repo root)

    Checks that catalog paths match real png files
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
            size (int, optional): Defaults to 128. Resolution to save png to tfrecord (i.e. width in pixels)
            final_size (int, optional): Defaults to 64. Resolution to load from tfrecord into model
            shard_size (int, optional): Defaults to 4096. Galaxies per shard.
        """
        self.size = size
        self.shard_size = shard_size
        self.shard_dir = shard_dir

        self.channels = 3  # save 3-band image to tfrecord. Augmented later by model input func.

        # paths for fixed tfrecords for initial training and (permanent) evaluation
        self.train_dir = os.path.join(self.shard_dir, 'train_shards') 
        self.val_dir = os.path.join(self.shard_dir, 'val_shards')
        self.test_dir = os.path.join(self.shard_dir, 'test_shards')

        # paths for catalogs. Used to look up .png locations during active learning.
        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        self.config_save_loc = os.path.join(self.shard_dir, 'shard_config.json')


    def train_tfrecord_locs(self):
        return [os.path.join(self.train_dir, loc) for loc in os.listdir(self.train_dir)
            if loc.endswith('.tfrecord')]


    def val_tfrecord_locs(self):
        return [os.path.join(self.val_dir, loc) for loc in os.listdir(self.val_dir)
            if loc.endswith('.tfrecord')]

    
    def test_tfrecord_locs(self):
        return [os.path.join(self.test_dir, loc) for loc in os.listdir(self.test_dir)
            if loc.endswith('.tfrecord')]


    def prepare_shards(self, labelled_catalog: pd.DataFrame, unlabelled_catalog: pd.DataFrame, labelled_columns_to_save: List,  val_fraction=0.1, test_fraction=0.2):
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
        if unlabelled_catalog is not None:
            unlabelled_catalog['file_loc'] = unlabelled_catalog['file_loc'].str.replace('/media/walml/beta/decals/png_native/dr5', '/share/nas/walml/galaxy_zoo/decals/dr5/png')

        assert 'id_str' in labelled_columns_to_save

        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)
        os.mkdir(self.train_dir)
        os.mkdir(self.val_dir)
        os.mkdir(self.test_dir)

        # check that file paths resolve correctly
        logging.info('Example file locs: \n{}'.format(labelled_catalog['file_loc'][:3].values))
        checks.check_no_missing_files(labelled_catalog['file_loc'], max_to_check=2000)
        if unlabelled_catalog is not None:
            checks.check_no_missing_files(unlabelled_catalog['file_loc'], max_to_check=2000)

        logging.info('\nLabelled subjects: {}'.format(len(labelled_catalog)))
        labelled_catalog.to_csv(self.labelled_catalog_loc)

        if unlabelled_catalog is not None:
            logging.info('Unlabelled subjects: {}'.format(len(unlabelled_catalog)))
            unlabelled_catalog.to_csv(self.unlabelled_catalog_loc)

        logging.info('Val fraction: {:.3f}. Test fraction: {:.3f}'.format(val_fraction, test_fraction))
        train_df, hidden_df = train_test_split(labelled_catalog, test_size=val_fraction + test_fraction)  # sklearn understands test size float as a fraction
        val_df, test_df = train_test_split(hidden_df, test_size=test_fraction / (val_fraction + test_fraction))  # both fractions of full catalog, now taking a slice of a slice

        logging.info('\nTraining subjects: {}'.format(len(train_df)))
        logging.info('Val subjects: {}'.format(len(val_df)))
        logging.info('Test subjects: {}'.format(len(test_df)))
        if len(train_df) < len(val_df):
            logging.warning('More val subjects than training subjects - is this intended?')
        train_df.to_csv(os.path.join(self.train_dir, 'train_df.csv'))
        val_df.to_csv(os.path.join(self.val_dir, 'val_df.csv'))
        test_df.to_csv(os.path.join(self.test_dir, 'test_df.csv'))

        logging.info('Writing {} train, {} val, and {} test galaxies to shards'.format(len(train_df), len(val_df), len(test_df)))
        for (df, save_dir) in [(train_df, self.train_dir), (val_df, self.val_dir), (test_df, self.test_dir)]:
            write_catalog_to_tfrecord_shards(
                df,
                img_size=self.size,
                columns_to_save=labelled_columns_to_save,
                save_dir=save_dir,
                shard_size=self.shard_size
            )

        if unlabelled_catalog is not None:
            logging.info('Writing {} unlabelled galaxies to shards (optional)'.format(len(unlabelled_catalog)))
            columns_to_save = ['id_str']
            write_catalog_to_tfrecord_shards(
                unlabelled_catalog,
                self.size,
                columns_to_save,
                self.shard_dir,
                self.shard_size
            )
        else:
            self.unlabelled_catalog_loc = ''  # record that no unlabelled catalog was used 

        assert self.ready()

        # serialized for later/logs
        self.write()


    def ready(self):
        assert os.path.isdir(self.shard_dir)
        assert os.path.isdir(self.train_dir)
        assert os.path.isdir(self.val_dir)
        assert os.path.isdir(self.test_dir)
        assert os.path.isfile(self.labelled_catalog_loc)
        if self.unlabelled_catalog_loc is '':
            logging.info('No unlabelled_catalog has been used')
        else:
            assert os.path.isfile(self.unlabelled_catalog_loc)
        return True

    def to_dict(self):
        return {
            'size': self.size,
            'shard_size': self.shard_size,
            'shard_dir': self.shard_dir,
            'channels': self.channels,
            'train_dir': self.train_dir,
            'val_dir': self.val_dir,
            'test_dir': self.test_dir,
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
        'val_dir',
        'test_dir',
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

