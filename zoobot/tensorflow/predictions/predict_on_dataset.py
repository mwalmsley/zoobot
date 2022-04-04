import os
import glob
import json
import logging
import time
import datetime
from typing import List
from pathlib import Path

import numpy as np
import tensorflow as tf

from zoobot.shared import save_predictions


def predict(ds: tf.data.Dataset, model: tf.keras.Model, n_samples: int, label_cols: List, save_loc: str):
    """
    Make and save predictions by model on image dataset.
    Args:
        ds (tf.data.Dataset): dataset yielding batches of (images, id_strs). Preprocessing already applied. id_strs may be the original path to image, or any other galaxy identifier.
        model (tf.keras.Model): trained model with which to make predictions
        n_samples (int): number of repeat predictions. Useful to marginalise over augmentations or MC Dropout.
        label_cols (list): Semantic labels for final model output dimension (e.g. ["smooth", "bar", "merger"]). Only used for output csv/hdf5 notes.
        save_loc (str): path to save csv of predictions.
    """

    # to make sure images and id_str line up, load id_str back out from dataset
    id_str_ds = ds.map(lambda _, id_str: id_str)
    image_id_strs = [id_str.numpy().decode('utf-8') for id_str_batch in id_str_ds for id_str in id_str_batch]

    logging.info('Beginning predictions')
    start = datetime.datetime.fromtimestamp(time.time())
    logging.info('Starting at: {}'.format(start.strftime('%Y-%m-%d %H:%M:%S')))

    predictions = np.stack([model.predict(ds) for n in range(n_samples)], axis=-1)
    logging.info('Predictions complete - {}'.format(predictions.shape))

    if save_loc.endswith('.csv'):      # save as pandas df
        save_predictions.predictions_to_csv(predictions, image_id_strs, label_cols, save_loc)
    elif save_loc.endswith('.hdf5'):
        save_predictions.predictions_to_hdf5(predictions, image_id_strs, label_cols, save_loc)
    else:
        logging.warning('Save format of {} not recognised - assuming csv'.format(save_loc))
        save_predictions.predictions_to_csv(predictions, image_id_strs, label_cols, save_loc)

    logging.info(f'Predictions saved to {save_loc}')

    end = datetime.datetime.fromtimestamp(time.time())
    logging.info('Completed at: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S')))
    logging.info('Time elapsed: {}'.format(end - start))



def paths_in_folder(folder: str, file_format: str, recursive=False):
    """
    Find all files of ``file_format`` in ``folder``, optionally recursively.
    Useful for getting list of image paths.

    Args:
        folder (str): path to folder to search
        file_format (str): include only files ending with this file format
        recursive (bool, optional): If True, search recursively i.e. all subfolders. Defaults to False.

    Returns:
        list: relative paths to all files of ``file_format`` in ``folder``.

    """
    assert os.path.isdir(folder)
    if recursive:  # in all subfolders, recursively
        unordered_paths = [str(path) for path in Path(folder).rglob('*.{}'.format(file_format))]
    else:
        unordered_paths = list(glob.glob('{}/*.{}'.format(folder, file_format)))  # only in that folder
    return unordered_paths
