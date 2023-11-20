import logging

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from PIL import Image  # necessary to avoid PIL.Image error assumption in web_datasets

from galaxy_datasets.shared import label_metadata
from galaxy_datasets import gz2

from sklearn.model_selection import train_test_split

from zoobot.pytorch.datasets import webdataset_utils


def dataset_to_webdataset(dataset_name, dataset_func, label_cols, divisor=4096):

    train_catalog, _ = dataset_func(root=f'/home/walml/data/galaxy-datasets/{dataset_name}', download=True, train=True)
    test_catalog, _ = dataset_func(root=f'/home/walml/data/galaxy-datasets/{dataset_name}', download=False, train=False)

    catalogs_to_webdataset(dataset_name, label_cols, train_catalog, test_catalog, divisor=divisor)


def catalogs_to_webdataset(dataset_name, label_cols, train_catalog, test_catalog, divisor=4096):
    for (catalog_name, catalog) in [('train', train_catalog), ('test', test_catalog)]:
        n_shards = len(catalog) // divisor
        logging.info(n_shards)

        catalog = catalog[:n_shards*divisor]
        logging.info(len(catalog))

        save_loc = f"/home/walml/data/wds/{dataset_name}/{dataset_name}_{catalog_name}.tar"  # .tar replace automatically
        
        webdataset_utils.df_to_wds(catalog, label_cols, save_loc, n_shards=n_shards)

        # webdataset_utils.load_wds_directly(save_loc)

        # webdataset_utils.load_wds_with_augmentation(save_loc)

        # webdataset_utils.load_wds_with_webdatamodule([save_loc], label_cols)


def main():

    # for converting my galaxy-dataset datasets
    # dataset_name = 'gz2'
    # dataset_func = gz2
    # label_cols = label_metadata.gz2_ortho_label_cols
    # dataset_to_webdataset(dataset_name, label_cols, dataset_func)

 

    # for converting other catalogs e.g. DESI
    dataset_name = 'desi_labelled'
    label_cols = label_metadata.decals_all_campaigns_ortho_label_cols
    columns = [
        'dr8_id', 'brickid', 'objid', 'ra', 'dec'
    ]
    df = pd.read_parquet('/home/walml/repos/decals-rings/data/master_all_file_index_passes_file_checks.parquet', columns=columns)
    # desi pipeline shreds sources. Be careful to deduplicate.

    columns = ['id_str'] + label_cols
    votes = pd.concat([
        pd.read_parquet(f'/media/walml/beta/galaxy_zoo/decals/dr8/catalogs/training_catalogs/{campaign}_ortho_v5_labelled_catalog.parquet', columns=columns)
        for campaign in ['dr12', 'dr5', 'dr8']
    ], axis=0)
    assert votes['id_str'].value_counts().max() == 1, votes['id_str'].value_counts()
    votes['dr8_id'] = votes['id_str']
    df = pd.merge(df, votes[['dr8_id']], on='dr8_id', how='inner')

    df['relative_file_loc'] = df.apply(lambda x: f"{x['brickid']}/{x['brickid']}_{x['objid']}.jpg", axis=1) 
    df['file_loc'] = '/home/walml/data/desi/jpg/' + df['relative_file_loc']

    df_dedup = remove_close_sky_matches(df)
    print(len(df_dedup))
    # df_dedup2 = remove_close_sky_matches(df_dedup)
    # print(len(df_dedup2))
    df_dedup.to_parquet('/home/walml/data/desi/master_all_file_index_labelled_dedup_20arcsec.parquet')

    df_dedup = pd.read_parquet('/home/walml/data/desi/master_all_file_index_labelled_dedup_20arcsec.parquet')

    # columns = ['id_str', 'smooth-or-featured-dr12_total-votes', 'smooth-or-featured-dr5_total-votes', 'smooth-or-featured-dr8_total-votes']

    df_dedup_with_votes = pd.merge(df_dedup, votes, how='inner', on='dr8_id')

    train_catalog, test_catalog = train_test_split(df_dedup_with_votes, test_size=0.2, random_state=42)
    train_catalog.to_parquet('/home/walml/data/wds/desi_labelled/train_catalog_v1.parquet', index=False)
    test_catalog.to_parquet('/home/walml/data/wds/desi_labelled/test_catalog_v1.parquet', index=False)

    catalogs_to_webdataset(dataset_name, label_cols, train_catalog, test_catalog, divisor=4096)

    

    

def remove_close_sky_matches(df, seplimit=20*u.arcsec, col_to_prioritise='ra'):

    catalog = SkyCoord(ra=df['ra'].values * u.deg, dec=df['dec'].values * u.deg)

    search_coords = catalog

    idxc, idxcatalog, d2d, _ = catalog.search_around_sky(search_coords, seplimit=seplimit)
    # idxc is index in search coords
    # idxcatalog is index in catalog
    # steps through all indexes in both that are within seplimit
    # d2d gives the distance (not used here)

    # includes self-match, so remove these
    idxc = idxc[d2d > 0]
    idxcatalog = idxcatalog[d2d > 0]
    d2d = d2d[d2d > 0]

    indices_to_drop = []
    for search_index_val in pd.unique(idxc):
        matched_indices = idxcatalog[idxc == search_index_val]
        matched_indices_including_self = matched_indices.tolist() + [search_index_val]

        # use RA as tiebreaker
        matching_galaxies = df.iloc[matched_indices_including_self]
        highest = matching_galaxies.index[np.argmax(matching_galaxies[col_to_prioritise])]
        these_indices_to_drop = list(set(matched_indices_including_self) - set([highest]))
        indices_to_drop += these_indices_to_drop

    indices_to_drop = set(indices_to_drop)
    all_indices = np.arange(len(df))  # index is like this, for sure
    indices_to_keep = set(all_indices) - indices_to_drop
    df_dedup = df.iloc[list(indices_to_keep)]
    return df_dedup





if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    main()





    # df = df[:100000]
    # df['total_votes'] = df['smooth-or-featured-dr12_total-votes'] + df['smooth-or-featured-dr5_total-votes'] + df['smooth-or-featured-dr8_total-votes']
    # df['total_votes'] = df['total_votes'].fillna(0)
    # df['random'] = np.random.rand(len(df))