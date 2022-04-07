from cProfile import label
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import h5py


def hdf5s_to_prediction_df(hdf5_locs: List):
    """
    Load predictions saved as hdf5 into pd.DataFrame with id_str and label_cols columns

    Best used when not using test-time dropout i.e. when you have a single forward pass per galaxy. 
    Otherwise, with test-time dropout on, aanswer columns (keyed by each value in label_cols) will be
    `np.ndarray` of length (n forward passes) for each galaxy and answer, *not* 1D scalars.
    This is quite awkward to work with. I suggest using `load_hdf5s` directly when using test-time dropout

    Args:
        hdf5_locs (List): _description_

    Returns:
        _type_: _description_
    """
    galaxy_id_df, predictions, label_cols = load_hdf5s(hdf5_locs)

    predictions = predictions.squeeze()
    if len(predictions.shape) > 2:
        logging.warning(
            'Predictions are of shape {}, greater than rank 2. \
            I suggest using load_hdf5s directly to work with np.arrays, not with DataFrame - see docstring'
        )
    prediction_df = pd.DataFrame(data=predictions, columns=label_cols)
    # copy over metadata (indices will align)
    prediction_df['id_str'] = galaxy_id_df['id_str']
    prediction_df['hdf5_loc'] = galaxy_id_df['hdf5_loc']
    return prediction_df


def load_hdf5s(hdf5_locs: List):
    """
    Load hdf5 predictions (with metadata) from disk.
    
    Each hdf5 includes the following datasets (accessed with f[dataset_name]):
        - predictions: np.array of shape (galaxy, answer, forward pass) containing model outputs (usually Dirichlet concentrations)
        - id_str: unique string (usually the file location) specifying which galaxy each prediction refers to. Indices match `predictions`
        - label_cols: semantic names for the answer dimension of `predictions` e.g. smooth-or-featured-dr8_smooth, smooth-or-featured-dr8_featured-or-disk, ...
    
    See predictions.predictions_to_hdf5 for the save function.

    hdf5 is useful because predictions are three-dimensional (e.g. 10 galaxies, 4 answers, 5 dropout passes)
    while csv's and similar are designed for two-dimensional data (e.g 10 galaxies, 4 answers).

    Args:
        hdf5_locs (list): hdf5 files to load

    Returns:
        pd.DataFrame: with rows of id_str, hdf5_loc, indexed like predictions (below)
        np.array: model predictions, usually dirichlet concentrations, like (galaxy, answer, forward pass)
    """

    if isinstance(hdf5_locs, str):
        logging.warning('Passed a single hdf5 loc to load_hdf5s - assuming this is a single file to load, not list of files to load')
        hdf5_locs = [hdf5_locs]  # pretend user passed a list

    predictions = []
    prediction_metadata = []
    template_label_cols = None  # will use this var to check consistency of label_cols across each hdf5_loc
    for loc in hdf5_locs:
        with h5py.File(loc, 'r') as f:
            logging.info(f.keys())
            these_predictions = f['predictions'][:]
            these_prediction_metadata = {
                'id_str': f['id_str'].asstr()[:],
                'hdf5_loc': [os.path.basename(loc) for _ in these_predictions]
        }
            predictions.append(these_predictions)
            prediction_metadata.append(these_prediction_metadata)

            if template_label_cols is None:  # first file to load, use this as the expected template
                template_label_cols = f['label_cols'].asstr()[:]
                logging.info('Using label columns {} from first hdf5 {}'.format(template_label_cols, loc))
            else:
                these_label_cols = f['label_cols'].asstr()[:]
                if these_label_cols != template_label_cols:
                    raise ValueError('Label columns {} of hdf5 {} do not match first label columns {}'.format(loc, f['label_cols'], template_label_cols))


    predictions = np.concatenate(predictions, axis=0)
    prediction_metadata = {
        'id_str': [p for metadata in prediction_metadata for p in metadata['id_str']],
        'hdf5_loc': [l for metadata in prediction_metadata for l in metadata['hdf5_loc']]
    }
    assert len(prediction_metadata['id_str']) == len(predictions)

    galaxy_id_df = pd.DataFrame(data=prediction_metadata)

    return galaxy_id_df, predictions, template_label_cols


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # simple tests

    # load_hdf5s('/nvme1/scratch/walml/repos/understanding_galaxies/scaled_image_predictions.hdf5')
    df = hdf5s_to_prediction_df('/nvme1/scratch/walml/repos/understanding_galaxies/scaled_image_predictions.hdf5')
    print(df)
