import os
import glob
import json
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from zoobot import label_metadata
from zoobot.data_utils import tfrecord_datasets
from zoobot.training import losses, training_config
from zoobot.estimators import preprocess, define_model


def prediction_to_row(prediction, id_str, schema):
    row = {
        'id_str': id_str
    }
    for n in range(len(schema.label_cols)):
        answer = schema.label_cols[n]
        row[answer + '_concentration'] = json.dumps(list(prediction[n].astype(float)))
        row[answer + '_concentration_mean'] = float(prediction[n].mean())
    return row


def predict(schema, tfrecord_locs, checkpoint_dir, save_loc, n_samples, batch_size, initial_size, crop_size, final_size, channels=3):

    raw_dataset = tfrecord_datasets.get_dataset(
        tfrecord_locs,
        label_cols=schema.label_cols,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        drop_remainder=False
    )
    id_strs_batched = [batch['id_str'] for batch in raw_dataset]
    id_strs = [id_str.numpy().squeeze()[2:-1] for batch in id_strs_batched for id_str in batch]

    input_config = preprocess.PreprocessingConfig(
        name='predict',
        label_cols=[],  # no labels
        initial_size=initial_size,
        final_size=final_size,
        batch_size=batch_size,
        greyscale=True,
        channels=3
    )
    dataset = preprocess.preprocess_dataset(raw_dataset, input_config)

    # TODO refactor
    model = define_model.get_model(schema, initial_size, crop_size, final_size)
    load_status = model.load_weights(checkpoint_dir)
    load_status.assert_nontrivial_match()
    load_status.assert_existing_objects_matched()

    logging.info('Beginning predictions')
    # predictions must fit in memory
    predictions = np.stack([model.predict(dataset) for n in range(n_samples)], axis=-1)
    logging.info('Predictions complete - {}'.format(predictions.shape))

    data = [prediction_to_row(predictions[n], id_strs[n], schema) for n in range(len(predictions))]
    predictions_df = pd.DataFrame(data)

    # if version == 'decals':
    #     catalog['iauname'] = catalog['iauname'].astype(str)  # or dr7objid

    # df = pd.merge(catalog, predictions_df, how='inner', on='id_str')
    # assert len(df) == len(predictions_df)

    predictions_df.to_csv(save_loc, index=False)
    logging.info(f'Predictions saved to {save_loc}')
