import os

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from sklearn.decomposition import IncrementalPCA


def create_pca_embedding(features: np.array, n_components: int, variance_plot_loc=None):
    """
    Compress galaxy representations into a lower dimensionality using Incremental PCA.
    These compressed representations are easier and faster to work with.

    Args:
        features (np.array): galaxy representations, of shape (galaxies, feature_dimensions)
        n_components (int): number of PCA components to use. Sets output dimension.
        variance_plot_loc (str, optional): If not None, save plot of variance vs. PCA components here. Defaults to None.

    Raises:
        ValueError: features includes np.nan values (PCA would break)

    Returns:
        np.array: PCA-compressed representations, of shape (galaxies, pca components)
    """
    assert len(features) > 0
    pca = IncrementalPCA(n_components=n_components, batch_size=20000)
    reduced_embed = pca.fit_transform(features)
    if np.isnan(reduced_embed).any():
        raise ValueError(f'embed is {np.isnan(reduced_embed).mean()} nan')

    if variance_plot_loc:
        plt.plot(range(n_components), pca.explained_variance_ratio_)
        plt.xlabel('Nth Component')
        plt.ylabel('Explained Variance')
        plt.tight_layout()
        plt.savefig(variance_plot_loc)

    return pd.DataFrame(data=reduced_embed, columns=['feat_{}_pca'.format(n) for n in range(n_components)])

"""
The code below is deprecated because I no longer use dataframes for this, but rather .hdf5. However, the approach might be useful to read."""

# def main(df: pd.DataFrame, name: str, output_dir: str, components_to_calculate=[5, 10, 30], id_col='iauname'):
#     """
#     Wrapper around :meth:`create_pca_embedding`.
#     Creates and saves several embeddings using (by default) 5, 10, and 30 PCA components.

#     Args:
#         df (pd.DataFrame): with columns of id_col (below) and feat_* (e.g. feat_0_pred, feat_1_pred, ...) recording representations for each galaxy (row)
#         name (str): Text to identify saved outputs. No effect on results.
#         output_dir (str): Directory in which to save results. No effect on results.
#         id_col (str, optional): Name of column containing unique strings identifying each galaxy. Defaults to 'iauname', matching DECaLS catalog. 'id_str' may be useful to match GZ2 catalog.
#     """
#     feature_cols = [col for col in df.columns.values if col.startswith('feat')]
    
#     features = df[feature_cols].values
    
#     for n_components in tqdm.tqdm(components_to_calculate):

#         # only need to bother with the variance plot for the highest num. components
#         if n_components == np.max(components_to_calculate):
#             variance_plot_loc = os.path.join(output_dir, name + '_explained_variance.pdf')
#         else:
#             variance_plot_loc = None

#         embed_df = create_pca_embedding(features, n_components, variance_plot_loc=variance_plot_loc)
#         embed_df[id_col] = df[id_col]  # pca embedding doesn't shuffle, so can copy the id col across to new df
#         embed_df.to_parquet(os.path.join(output_dir, '{}_pca{}_and_ids.parquet'.format(name, n_components)), index=False)


# if __name__ == '__main__':

#     sns.set_context('notebook')

#     output_dir = '/Users/walml/repos/zoobot/data/results'
#     assert os.path.isdir(output_dir)

#     name = 'decals_dr5_oct_21' 
#     features_loc = '/Volumes/beta/cnn_features/decals/cnn_features_decals.parquet'  # TODO point this to your representations download from Zenodo
#     df = pd.read_parquet(features_loc)
#     # TODO replace second arg with your image download folder
#     df['png_loc'] = df['png_loc'].str.replace('/media/walml/beta1/decals/png_native/dr5', '/Volumes/beta/decals/png_native/dr5')  
#     id_col = 'iauname'

#     """
#     Similarly for GZ2
#     """
#     # name = 'gz2' 
#     # features_loc = '/Volumes/beta/cnn_features/gz2/cnn_features_gz2.parquet'  # TODO point this to your download from Zenodo
#     # df = pd.read_parquet(features_loc)
#     # # TODO replace second arg with your image download folder
#     # df['png_loc'] = df['png_loc'].str.replace('/media/walml/beta1/galaxy_zoo/gz2/png', '/Volumes/beta/galaxy_zoo/gz2/png')  
#     # id_col = 'id_str'

#     main(df, name, output_dir, id_col=id_col)
