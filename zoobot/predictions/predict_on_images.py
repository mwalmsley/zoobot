import os
import glob
import json
import logging
import time
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from zoobot import label_metadata
from zoobot.data_utils import image_datasets
from zoobot.estimators import define_model, preprocess
from zoobot import schemas 

from pathlib import Path


def prediction_to_row(prediction, png_loc, label_cols):
    row = {
        'png_loc': png_loc
    }
    for n in range(len(label_cols)):
        answer = label_cols[n]
        row[answer + '_concentration'] = json.dumps(list(prediction[n].astype(float)))
        # row[answer + '_concentration_mean'] = float(prediction[n].mean())
    return row


def paths_in_folder(folder, file_format, recursive=False):
    assert os.path.isdir(folder)
    if recursive:  # in all subfolders, recursively
        unordered_image_paths = [str(path) for path in Path(folder).rglob('*.{}'.format(file_format))]
    else:
        unordered_image_paths = list(glob.glob('{}/*.{}'.format(folder, file_format)))  # only in that folder
    return unordered_image_paths



def predict(image_ds, model, n_samples, label_cols, save_loc):

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
    logging.info(predictions_df)

    predictions_df.to_csv(save_loc, index=False)
    logging.info(f'Predictions saved to {save_loc}')

    end = datetime.datetime.fromtimestamp(time.time())
    logging.info('Completed at: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S')))
    logging.info('Time elapsed: {}'.format(end - start))

    # you can ignore warnings about not loading the optimizer's state
    # the optimizer isn't needed to make predictions, only for training
