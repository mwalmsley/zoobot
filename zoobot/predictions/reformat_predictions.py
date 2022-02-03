import glob
import os
import logging
import functools
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

"""
The CNN prediction code (e.g. make_predictions.py) is designed for multiple (MC Dropout) forward passes for mutiple classes.
Each galaxy can therefore have a num passes x num classes dimensional-prediction
For example, [[pass_0_class_a, pass_1 class_a], [pass_0_class_b, pass_1_class_b]]

This is overkill where we use a single forward pass and a single class.
This script converts the predictions in a make_predictions.py output csv (e.g. [[0.3]]) to be a simple float (e.g. 0.3)

I might remove this by having make_predictions.py etc notice when there is one class and one forward pass and do this before saving to disk TODO
"""

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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    overwrite = True

    run_name = 'dr5_rings'  # the text identifying each of the output prediction csv's e.g. dr5_rings_full_features_0_5000.csv, etc.

    raw_search_str = 'data/results/{}_*_raw.csv'.format(run_name)
    clean_search_str = raw_loc_to_clean_loc(raw_search_str)  # simply gets new name e.g. 'data/results/{}_*_clean.csv'
    assert raw_search_str != clean_search_str

    # each cleaned csv will be concatenated and saved here
    reformatted_parquet_loc = os.path.join(os.path.dirname(raw_search_str), '{}_cleaned_concat.parquet'.format(run_name))

    main(raw_search_str, clean_search_str, reformatted_parquet_loc, overwrite=overwrite)
