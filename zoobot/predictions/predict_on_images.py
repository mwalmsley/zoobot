import os
import glob
import json
import logging
import time
import datetime
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from zoobot import label_metadata
from zoobot.data_utils import image_datasets
from zoobot.estimators import define_model, preprocess
from zoobot import schemas 

from pathlib import Path




def predict(image_ds: tf.data.Dataset, model: tf.keras.Model, n_samples: int, label_cols: List, save_loc: str):
    """
    Make and save predictions by model on image dataset.

    Args:
        image_ds (tf.data.Dataset): image dataset yielding batches of (images, paths)
        model (tf.keras.Model): [description]
        n_samples (int): number of repeat predictions. Useful to marginalise over augmentations or MC Dropout.
        label_cols (list): Semantic labels for final model output dimension (e.g. ["smooth", "bar", "merger"]). Only used for output csv headers.
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

    data = [prediction_to_row(predictions[n], image_paths[n], label_cols=label_cols) for n in range(len(predictions))]
    predictions_df = pd.DataFrame(data)
    # logging.info(predictions_df)

    predictions_df.to_csv(save_loc, index=False)
    logging.info(f'Predictions saved to {save_loc}')

    end = datetime.datetime.fromtimestamp(time.time())
    logging.info('Completed at: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S')))
    logging.info('Time elapsed: {}'.format(end - start))


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
