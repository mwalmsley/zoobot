"""
Save catalog columns and images to tfrecord shards.
No knowledge of oracle allowed! Similarly, no knowledge of labels allowed. 
Allowed to assume:
- Each catalog entry has an image under `file_loc`
- Each catalog entry has an identifier under `id_str`
"""
import argparse
import os
import shutil
import logging
import json
import time
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

# from shared_astro_utils import object_utils

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
        shard_dir,  # to hold a new folder, named after the shard config 
        size,  # IMPORTANT image size
        shard_size=4096,
        **overflow_args  # TODO review removing this
        ):
        """
        Args:
            shard_dir (str): directory into which to save shards
            size (int, optional): Defaults to 128. Resolution to save fits to tfrecord
            final_size (int, optional): Defaults to 64. Resolution to load from tfrecord into model
            shard_size (int, optional): Defaults to 4096. Galaxies per shard.
        """
        self.size = size
        self.shard_size = shard_size
        self.shard_dir = shard_dir

        self.channels = 3  # save 3-band image to tfrecord. Augmented later by model input func.

        self.db_loc = os.path.join(self.shard_dir, 'static_shard_db.db')  # record shard contents

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


    def prepare_shards(self, labelled_catalog, unlabelled_catalog, train_test_fraction, columns_to_save):
        """[summary]
        
        Args:
            labelled_catalog (pd.DataFrame): labelled galaxies, including fits_loc column
            unlabelled_catalog (pd.DataFrame): unlabelled galaxies, including fits_loc column
            train_test_fraction (float): fraction of labelled catalog to use as training data
            columns_to_save list: Save catalog cols to tfrecord, under same name. 
        """
        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)
        os.mkdir(self.train_dir)
        os.mkdir(self.eval_dir)

        # check that file paths resolve correctly
        print(labelled_catalog['file_loc'][:3].values)
        checks.check_no_missing_files(labelled_catalog['file_loc'], max_to_check=2000)
        checks.check_no_missing_files(unlabelled_catalog['file_loc'], max_to_check=2000)

        # assume the catalog is true, don't modify halfway through
        logging.info('\nLabelled subjects: {}'.format(len(labelled_catalog)))
        logging.info('Unlabelled subjects: {}'.format(len(unlabelled_catalog)))
        logging.info(f'Train-test fraction: {train_test_fraction}')
        labelled_catalog.to_csv(self.labelled_catalog_loc)
        unlabelled_catalog.to_csv(self.unlabelled_catalog_loc)

        # save train/test split into training and eval shards
        # train_df, eval_df = catalog_to_tfrecord.split_df(labelled_catalog, train_test_fraction=train_test_fraction)
        # temp change: everything labelled as train, everything unlabelled (just 4k) as eval
        train_df = labelled_catalog
        eval_df = unlabelled_catalog
        logging.info('\nTraining subjects: {}'.format(len(train_df)))
        logging.info('Eval subjects: {}'.format(len(eval_df)))
        if len(train_df) < len(eval_df):
            print('More eval subjects than training subjects - is this intended?')
        train_df.to_csv(os.path.join(self.train_dir, 'train_df.csv'))
        eval_df.to_csv(os.path.join(self.eval_dir, 'eval_df.csv'))

        # training and eval galaxies are labelled and should never be read by db
        # just write them directly as shards, don't enter into db
        for (df, save_dir) in [(train_df, self.train_dir), (eval_df, self.eval_dir)]:
            write_catalog_to_tfrecord_shards(
                df,
                img_size=self.size,
                columns_to_save=columns_to_save,
                save_dir=save_dir,
                shard_size=self.shard_size
            )

        # write unlabelled galaxies to shards (optional)
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
        assert os.path.isfile(self.db_loc)
        assert os.path.isfile(self.labelled_catalog_loc)
        assert os.path.isfile(self.unlabelled_catalog_loc)
        return True


    # def to_dict(self):
    #     return object_utils.object_to_dict(self)

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
        'config_save_loc',
        'db_loc']
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
    for shard_n, df_shard in tqdm(enumerate(df_shards), total=len(df_shards)):
        save_loc = os.path.join(save_dir, 's{}_shard_{}.tfrecord'.format(img_size, shard_n))
        catalog_to_tfrecord.write_image_df_to_tfrecord(
            df_shard, 
            save_loc,
            img_size,
            columns_to_save,
            reader=catalog_to_tfrecord.get_reader(df['file_loc']),
            append=False)



if __name__ == '__main__':

    """
    Sim shards: add simulation_context
        # python zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/decals_multiq/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/decals_multiq/unlabelled_catalog.csv --eval-size 5000 --shard-dir=data/decals/shards/decals_multiq_128_sim --img-size 128
    
    DECALS all:
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired/unlabelled_catalog.csv --eval-size 10000 --shard-dir=data/decals/shards/all_2p5_unfiltered_retired --img-size 300
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired/unlabelled_catalog.csv --eval-size 100 --shard-dir=data/decals/shards/all_2p5_unfiltered_retired --max-labelled 500 --max-unlabelled=300 --img-size 32

        python zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired_arc/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired_arc/unlabelled_catalog.csv --eval-size 10000 --shard-dir=data/decals/shards/all_2p5_unfiltered_retired  --img-size 300
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired_arc/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired_arc/unlabelled_catalog.csv --eval-size 100 --shard-dir=data/decals/shards/all_2p5_unfiltered_retired --max-labelled 500 --max-unlabelled=300 --img-size 32

    GZ2 sim:
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5/simulation_context/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5/simulation_context/unlabelled_catalog.csv --eval-size 1000 --shard-dir=data/gz2/shards/all_featp5_facep5_sim_256 --img-size 256
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5_2p5/simulation_context/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5_2p5/simulation_context/unlabelled_catalog.csv --eval-size 4000 --shard-dir=data/gz2/shards/all_featp5_facep5_sim_300 --img-size 300
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_2p5_arc_unfiltered/simulation_context/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_2p5_arc_unfiltered/simulation_context/unlabelled_catalog.csv --eval-size 15000 --shard-dir=data/gz2/shards/all_sim_2p5_unfiltered_300 --img-size 300
        
        $PYTHON zoobot/active_learning/make_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5_arc/simulation_context/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5_arc/simulation_context/unlabelled_catalog.csv --eval-size 1000 --shard-dir=data/gz2/shards/all_sim_2p5_unfiltered_300 --img-size 300
    
    GZ2 all:
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5/unlabelled_catalog.csv --eval-size 1000 --shard-dir=data/gz2/shards/all_featp5_facep5_256 --img-size 256

    Testing:
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/decals/prepared_catalogs/decals_multiq/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/decals_multiq/unlabelled_catalog.csv --eval-size=100 --shard-dir=data/decals/shards/debug_sim --max-labelled 500 --max-unlabelled=300 --img-size 32
        python zoobot/active_learning/make_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_2p5_unfiltered/simulation_context/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_2p5_unfiltered/simulation_context/unlabelled_catalog.csv --shard-dir=data/gz2/shards/debug_sim --eval-size 1000 --max-labelled 5000 --max-unlabelled=3000 --img-size 64
    """

    # only responsible for making the shards. 
    # Assumes catalogs are shuffled and have id_str, file_loc, label, total_votes columns

    parser = argparse.ArgumentParser(description='Make shards')
    # to create for GZ2, see zoobot/get_catalogs/gz2
    # to create for DECALS, see github/zooniverse/decals and apply zoobot/active_learning/make_decals_catalog to `joint_catalog_for_upload`
    parser.add_argument('--labelled-catalog', dest='labelled_catalog_loc', type=str,
                    help='Path to csv catalog of previous labels and file_loc, for shards')

    parser.add_argument('--unlabelled-catalog', dest='unlabelled_catalog_loc', type=str,
                help='Path to csv catalog of previous labels and file_loc, for shards')

    parser.add_argument('--eval-size', dest='eval_size', type=int)

    # Write catalog to shards (tfrecords as catalog chunks) here for use in active learning
    parser.add_argument('--shard-dir', dest='shard_dir', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--max-unlabelled', dest='max_unlabelled', type=int,
                    help='Max galaxies (for debugging/speed')
    parser.add_argument('--max-labelled', dest='max_labelled', type=int,
                    help='Max galaxies (for debugging/speed')
    parser.add_argument('--img-size', dest='size', type=int,
                    help='Max galaxies (for debugging/speed')

    args = parser.parse_args()

    # label_cols = label_metadata.decals_partial_label_cols
    label_cols = label_metadata.decals_label_cols
    # label_cols = label_metadata.gz2_partial_label_cols
    # label_cols = label_metadata.gz2_label_cols

    # log_loc = 'make_shards_{}.log'.format(time.time())
    logging.basicConfig(
        # filename=log_loc,
        # filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )

    # labels will always be floats, int conversion confuses tf.data
    dtypes = dict(zip(label_cols, [float for _ in label_cols]))
    dtypes['id_str'] = str
    labelled_catalog = pd.read_csv(args.labelled_catalog_loc, dtype=dtypes)
    unlabelled_catalog = pd.read_csv(args.unlabelled_catalog_loc, dtype=dtypes)

    # limit catalogs to random subsets
    if args.max_labelled:
        labelled_catalog = labelled_catalog.sample(len(labelled_catalog))[:args.max_labelled]
    if args.max_unlabelled:  
        unlabelled_catalog = unlabelled_catalog.sample(len(unlabelled_catalog))[:args.max_unlabelled]

    logging.info('Labelled: {}, unlabelled: {}'.format(len(labelled_catalog), len(unlabelled_catalog)))

    # in memory for now, but will be serialized for later/logs
    train_test_fraction = catalog_to_tfrecord.get_train_test_fraction(len(labelled_catalog), args.eval_size)

    columns_to_save = ['id_str'] + label_cols
    logging.info('Saving {} columns)'.format(columns_to_save))

    shard_config = ShardConfig(shard_dir=args.shard_dir, size=args.size)

    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog,
        train_test_fraction=train_test_fraction,
        columns_to_save=columns_to_save
    )
    
    # finally, tidy up by moving the log into the shard directory
    # could not be create here because shard directory did not exist at start of script
    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha
    # shutil.move(log_loc, os.path.join(args.shard_dir, '{}.log'.format(sha)))
