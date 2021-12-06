import os
import glob
import json
import logging
import time
import datetime
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import h5py


def predict(image_ds: tf.data.Dataset, model: tf.keras.Model, n_samples: int, label_cols: List, save_loc: str):
    """
    Make and save predictions by model on image dataset.

    Args:
        image_ds (tf.data.Dataset): image dataset yielding batches of (images, paths)
        model (tf.keras.Model): [description]
        n_samples (int): number of repeat predictions. Useful to marginalise over augmentations or MC Dropout.
        label_cols (list): Semantic labels for final model output dimension (e.g. ["smooth", "bar", "merger"]). Only used for output csv/hdf5 notes.
        save_loc (str): path to save csv of predictions.
    """

    # order of images is not always the same as order of paths, so load paths (saved under id_str key) back out
    path_ds = image_ds.map(lambda image, paths: paths)
    image_paths = [path.numpy().decode('utf-8') for path_batch in path_ds for path in path_batch]

    logging.info('Beginning predictions')
    start = datetime.datetime.fromtimestamp(time.time())
    logging.info('Starting at: {}'.format(start.strftime('%Y-%m-%d %H:%M:%S')))

    predictions = np.stack([model.predict(image_ds) for n in range(n_samples)], axis=-1)
    logging.info('Predictions complete - {}'.format(predictions.shape))

    if save_loc.endswith('.csv'):      # save as pandas df
        predictions_to_csv(predictions, image_paths, label_cols, save_loc)
    elif save_loc.endswith('.hdf5'):
        predictions_to_hdf5(predictions, image_paths, label_cols, save_loc)
    else:
        logging.warning('Save format of {} not recognised - assuming csv'.format(save_loc))
        predictions_to_csv(predictions, image_paths, label_cols, save_loc)

    logging.info(f'Predictions saved to {save_loc}')

    end = datetime.datetime.fromtimestamp(time.time())
    logging.info('Completed at: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S')))
    logging.info('Time elapsed: {}'.format(end - start))


def predictions_to_hdf5(predictions, image_paths, label_cols, save_loc):
    assert save_loc.endswith('.hdf5')
    with h5py.File(save_loc, "w") as f:
        f.create_dataset(name='predictions', data=predictions)
        # https://docs.h5py.org/en/stable/special.html#h5py.string_dtype
        dt = h5py.string_dtype(encoding='utf-8')
        # predictions_dset.attrs['label_cols'] = label_cols  # would be more conventional but is a little awkward
        f.create_dataset(name='image_paths', data=image_paths, dtype=dt)
        f.create_dataset(name='label_cols', data=label_cols, dtype=dt)


def predictions_to_csv(predictions, image_paths, label_cols, save_loc):
    assert save_loc.endswith('.csv')
    data = [prediction_to_row(predictions[n], image_paths[n], label_cols=label_cols) for n in range(len(predictions))]
    predictions_df = pd.DataFrame(data)
    # logging.info(predictions_df)
    predictions_df.to_csv(save_loc, index=False)


def prediction_to_row(prediction: np.ndarray, image_loc: str, label_cols: List):
    """
    Convert prediction on image into dict suitable for saving as csv
    Predictions are encoded as a json e.g. "[1., 0.9]" for 2 repeat predictions on one galaxy
    This makes them easy to read back with df[col] = df[col].apply(json.loads)

    Args:
        prediction (np.ndarray): model output for one galaxy, including repeat predictions e.g. [[1., 0.9], [0.3, 0.24]] for model with output_dim=2 and 2 repeat predictions
        image_loc (str): path to image
        label_cols (list): semantic labels for model output dim e.g. ['smooth', 'bar'].

    Returns:
        dict: of the form {'image_loc': 'path', 'smooth_pred': "[1., 0.9]", 'bar_pred: "[0.3, 0.24]"}
    
    """
    row = {
        'image_loc': image_loc
    }
    for n in range(len(label_cols)):
        answer = label_cols[n]
        row[answer + '_pred'] = json.dumps(list(prediction[n].astype(float)))
    return row


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
