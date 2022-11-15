import os
from urllib.parse import urlparse

import pandas as pd

if __name__ == '__main__':

    repo_dir = '/home/walml/repos'

    df = pd.read_parquet(os.path.join(repo_dir, 'gz-downloads/gz_cosmic_dawn_early_aggregation_ortho.parquet'))
  
    file_key = pd.read_csv('data/cosmic_dawn_file_key.csv')
    file_key['filename'] = file_key['locations'].apply(lambda x: f'{os.path.basename(urlparse(x).path)}')
    # file_key['file_loc'] =  file_key['filename'].apply(lambda x: os.path.join(repo_dir, 'zoobot/data/example_images/cosmic_dawn', x))  # absolute path, annoying to share to cluster
    file_key['file_loc'] =  file_key['filename'].apply(lambda x: os.path.join('data/example_images/cosmic_dawn', x))  # relative path

    df = pd.merge(df, file_key[['id_str', 'file_loc']], on='id_str', how='inner')

    df.to_parquet(os.path.join(repo_dir, 'zoobot/data/gz_cosmic_dawn_early_aggregation_ortho_with_file_locs.parquet'), index=False)
