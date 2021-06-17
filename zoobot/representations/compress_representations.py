import os
import glob
import pickle
import logging

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from sklearn.decomposition import IncrementalPCA


def create_pca_embedding(features, n_components, variance_plot_loc=None):
    assert len(features) > 0
    pca = IncrementalPCA(n_components=n_components, batch_size=20000)
    reduced_embed = pca.fit_transform(features)
    if np.isnan(reduced_embed).any():
        raise ValueError(f'embed is {np.isnan(reduced_embed).mean()} nan')

    if variance_plot_loc:  # only need for last one
        plt.plot(range(n_components), pca.explained_variance_ratio_)
        plt.xlabel('Nth Component')
        plt.ylabel('Explained Variance')
        plt.tight_layout()
        plt.savefig(variance_plot_loc)

    return pd.DataFrame(data=reduced_embed, columns=['feat_{}_pca'.format(n) for n in range(n_components)])



def main(features_cleaned_and_concat_loc, catalog_loc, name, output_dir):

    # made by reformat_cnn_features.py
    df = pd.read_parquet(features_cleaned_and_concat_loc)
    print(df.iloc[0]['filename'])

    """join to catalog"""
    catalog = pd.read_parquet(catalog_loc)
    catalog['filename'] = catalog['png_loc']  # will use this for first merge w/ features, then use dr8_id or galaxy_id going forwards
    print(catalog.iloc[0]['filename'])
    df = pd.merge(df, catalog, on='filename', how='inner').reset_index(drop=True)  # applies previous filters implicitly
    df = df.sample(len(df), random_state=42).reset_index()
    assert len(df) > 0
    logging.info(len(df))

    # # rename dr8 catalog cols
    # df = df.rename(columns={
    #     'weighted_radius', 'estimated_radius',  # TODO I have since improved this column, need to update
    #     'dr8_id': 'galaxy_id'
    # })

    # rename dr5 catalog cols
    df = df.rename(columns={
        'petro_th50': 'estimated_radius',  # TODO I have since improved this column, need to update
        'iauname': 'galaxy_id'
    })

    df.to_parquet(os.path.join(output_dir, '{}_full_features_and_safe_catalog.parquet'.format(name)), index=False)

    feature_cols = [col for col in df.columns.values if col.startswith('feat')]
    
    features = df[feature_cols].values
    
    components_to_calculate = [5, 10, 30]
    for n_components in tqdm.tqdm(components_to_calculate):

        if n_components == np.max(components_to_calculate):
            variance_plot_loc = 'explained_variance.pdf'
        else:
            variance_plot_loc = None

        embed_df = create_pca_embedding(features, n_components, variance_plot_loc=variance_plot_loc)
        embed_df['galaxy_id'] = df['galaxy_id']
        embed_df.to_parquet(os.path.join(output_dir, '{}_pca{}_and_ids.parquet'.format(name, n_components)), index=False)
    

if __name__ == '__main__':

    sns.set_context('notebook')

    features_cleaned_and_concat_loc = '/share/nas/walml/repos/zoobot/data/results/dr5_color_cnn_features_concat.parquet'
    catalog_loc = '/share/nas/walml/dr5_nsa_v1_0_0_to_upload.parquet'

    name = 'dr5_color'
    output_dir = '/share/nas/walml/repos/zoobot/data/results'
    
    main(features_cleaned_and_concat_loc, catalog_loc, name=name, output_dir=output_dir)




    # catalog_loc = '/raid/scratch/walml/repos/download_DECaLS_images/working_dr8_master.parquet'

    # columns=['png_loc', 'weighted_radius', 'ra', 'dec', 'dr8_id']  # dr8 catalog cols

    # TODO I think this has not been correctly filtered for bad images. Run the checks again, perhaps w/ decals downloader? Or check data release approach
    # dr5_df = pd.read_parquet('dr5_b0_full_features_and_safe_catalog.parquet')
    # """Rename a few columns"""
    # print(dr5_df.head())
    # dr5_df['estimated_radius'] = dr5_df['petro_th50']
    # dr5_df['galaxy_id'] = dr5_df['iauname']


    # df = pd.concat([dr5_df, dr8_df], axis=0).reset_index(drop=True)  # concat rowwise, some cols will have nans - but not png_loc or feature cols
    # important to reset index else index is not unique, would be like 0123...0123...


    # df[not_feature_cols].to_parquet('dr5_dr8_catalog_with_radius.parquet')

    # TODO rereun including ra/dec and check for duplicates/very close overlaps