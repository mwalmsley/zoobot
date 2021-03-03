import os
import glob
import json
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from zoobot import label_metadata
from zoobot.training import losses, training_config
from zoobot.estimators import input_utils, define_model


def prediction_to_row(prediction, id_str, schema):
    row = {
        'id_str': id_str
    }
    for n in range(schema.label_cols):
        answer = schema.label_cols[n]
        row[answer + '_concentration'] = json.dumps(list(prediction[n].astype(float)))
        row[answer + '_concentration_mean'] = float(prediction[n].mean())
    return row


def predict(schema, catalog_loc, tfrecord_locs, checkpoint_dir, save_loc, n_samples, batch_size, initial_size, crop_size, final_size, channels=3):
    # logging.info(f'{catalog_loc}, {tfrecord_locs}, {checkpoint_dir}, {save_loc}, {n_samples}, {batch_size}')
    # logging.info(len(tfrecord_locs))

    catalog = pd.read_csv(catalog_loc, dtype={'subject_id': str})  # original catalog

    eval_config = training_config.get_eval_config(
        tfrecord_locs, [], batch_size, initial_size, final_size, channels  # label cols = [] as we don't know them, in general
    )
    eval_config.drop_remainder = False
    dataset = input_utils.get_input(config=eval_config)

    feature_spec = input_utils.get_feature_spec({'id_str': 'string'})
    id_str_dataset = input_utils.get_dataset(
        tfrecord_locs, feature_spec, batch_size=1, shuffle=False, repeat=False, drop_remainder=False
    )
    id_strs = [str(d['id_str'].numpy().squeeze())[2:-1] for d in id_str_dataset]

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

    df = pd.merge(catalog, predictions_df, how='inner', on='id_str')
    assert len(df) == len(predictions_df)

    df.to_csv(save_loc, index=False)
    logging.info(f'Predictions saved to {save_loc}')
