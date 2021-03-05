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


def prediction_to_row(prediction, png_loc, label_cols):
    row = {
        'png_loc': png_loc
    }
    for n in range(len(label_cols)):
        answer = label_cols[n]
        row[answer + '_concentration'] = json.dumps(list(prediction[n].astype(float)))
        # row[answer + '_concentration_mean'] = float(prediction[n].mean())
    return row


def predict(
    label_cols, file_format, folder_to_predict, checkpoint_dir, save_loc, n_samples,
    batch_size, initial_size, crop_size, resize_size
    ):

    assert os.path.isdir(folder_to_predict)
    unordered_image_paths = list(glob.glob('{}/*.{}'.format(folder_to_predict, file_format)))  # in that folder
    # image_paths = list(glob.glob('*/*.{}'.format(file_format)))  # next folder down only, not recursive
    # image_paths = list(glob.glob('*.{}'.format(file_format)))  # this folder only
    assert len(unordered_image_paths) > 0

    raw_image_ds = image_datasets.get_image_dataset([str(x) for x in unordered_image_paths], file_format, initial_size, batch_size)
    # order of images is not always the same as order of paths, so load paths (saved under id_str key) back out
    path_ds = raw_image_ds.map(lambda x: x['id_str'])
    image_paths = [path.numpy().decode('utf-8') for path_batch in path_ds for path in path_batch]
    assert set(image_paths) == set(unordered_image_paths)

    preprocessing_config = preprocess.PreprocessingConfig(
        label_cols=[],
        input_size=initial_size,
        channels=3,
        greyscale=True
    )
    image_ds = preprocess.preprocess_dataset(raw_image_ds, preprocessing_config)
    # images_only = image_ds.map(lambda x: x['matrix'])
    # for im in images_only.take(5):
    #     print(im.shape)
    # exit()

    model = define_model.load_model(
        checkpoint_dir=checkpoint_dir,
        include_top=True,
        input_size=initial_size,
        crop_size=crop_size,
        resize_size=resize_size,
        expect_partial=True  # optimiser state will load as we're not using it for prediction
    )
    # model = define_model.get_model(len(schema.label_cols), initial_size, crop_size, resize_size)
    # load_status = model.load_weights(checkpoint_dir)
    # load_status.assert_nontrivial_match()
    # load_status.assert_existing_objects_matched()

    logging.info('Beginning predictions')
    start = datetime.datetime.fromtimestamp(time.time())
    logging.info('Starting at: {}'.format(start.strftime('%Y-%m-%d %H:%M:%S')))

    # for debugging
    # predictions = model.predict(png_ds.take(1))  # np out
    # print(predictions[:, 0] / predictions[:, :2].sum(axis=1))

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
