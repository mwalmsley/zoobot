import glob
import os
import logging
import functools
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

"""Useful to convert cnn predictions with a single forward pass and a single class (e.g. class_a: [[0.3]]) to simple floats rather than lists (e.g. class_a: 0.3)"""

def raw_loc_to_clean_loc(raw_loc):
    if '_full_features_' in raw_loc:  # convention for cnn features
        return raw_loc.replace('_full_features_', '_full_cleaned_').replace('.csv', '.parquet')
    else:
        return raw_loc.split('.')[-2] + '_cleaned.' + raw_loc.split('.')[-1]


def clean_feature_csv(loc, image_format = 'png', overwrite=False):
        
    logging.info('Reformatting {}'.format(loc))
    clean_loc = raw_loc_to_clean_loc(loc)
    if overwrite or not os.path.isfile(clean_loc):
        logging.info('Cleaning {}'.format(loc))

        df = pd.read_csv(loc)

        cols_to_clean = [col for col in df.columns.values if col not in ['id_str', 'image_loc']]

        for col in cols_to_clean:
            df[col] = df[col].apply(lambda x: float(x.replace('[', '').replace(']', '')))  # extract from list e.g. [0.1456] to 0.1456

        df['id_str'] = list(df['image_loc'].apply(lambda x: os.path.basename(x).replace('.png', '')))  # assumes png format

        df.to_parquet(clean_loc)

    else:
        logging.info('Skipping {} - file exists and overwrite is {}'.format(loc, overwrite))


def concat(clean_locs):
    data = []
    for loc in clean_locs:
        data.append(pd.read_parquet(loc))
    df = pd.concat(data)
    logging.info('Total galaxies in reformatted and concat df: {}'.format(len(df)))
    return df


def main(raw_search_str, clean_search_str, reformatted_parquet_loc, overwrite=False):
        
    logging.info('Raw files: {}'.format(raw_search_str))
    logging.info('Reformatted files: {}'.format(clean_search_str))
    logging.info('Destination parquet: {}'.format(reformatted_parquet_loc))

    raw_locs = glob.glob(raw_search_str)
    assert raw_locs
    logging.info('Raw csvs to reformat: {} e.g. {}'.format(len(raw_locs), raw_locs[0]))

    pool = Pool(processes=20)

    pbar = tqdm(total=len(raw_locs))
    clean_feature_csv_partial = functools.partial(clean_feature_csv, overwrite=overwrite)  # can't use lambda with multiprocessing
    for _ in pool.imap_unordered(clean_feature_csv_partial, raw_locs):
        pbar.update()

    clean_locs = glob.glob(clean_search_str)
    assert clean_locs
    df = concat(clean_locs)

    print('Saving to {}'.format(reformatted_parquet_loc))
    df.to_parquet(reformatted_parquet_loc, index=False)



if __name__ == '__main__':

    # raw_search_str = '/media/walml/beta1/cnn_features/gz2/*_full_features_*.csv'
    # clean_search_str = '/media/walml/beta1/cnn_features/gz2/*_cleaned_*.parquet'
    # reformatted_parquet_loc = '/media/walml/beta1/cnn_features/gz2/cnn_features_concat.parquet'

    logging.basicConfig(level=logging.INFO)

    overwrite = True

    # run_name = 'dr5_color'
    run_name = 'dr5_rings'

    raw_search_str = '/share/nas/walml/repos/zoobot/data/results/{}_*_raw.csv'.format(run_name)
    clean_search_str = raw_loc_to_clean_loc(raw_search_str)
    assert raw_search_str != clean_search_str
    reformatted_parquet_loc = os.path.join(os.path.dirname(raw_search_str), '{}_cleaned_concat.parquet'.format(run_name))

    main(raw_search_str, clean_search_str, reformatted_parquet_loc, overwrite=overwrite)
