import os
import logging
import glob

import tensorflow as tf

from zoobot import label_metadata, schemas
from zoobot.predictions import predict_on_images


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # possible driver errors if you don't include this, try it and see
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # decals schema, replace the args if you defined your own
    schema = schemas.Schema(
        label_cols=label_metadata.decals_label_cols,
        questions=label_metadata.decals_questions,
        version='decals'
    )

    initial_size = 300
    crop_size = int(initial_size * 0.75)
    final_size = 224
    channels = 3
    
    batch_size = 8  # 128 for the paper, but you'll need a good GPU
    n_samples = 5

    # TODO you'll want to replace these with your own paths
    catalog_loc = 'data/gz2/gz2_master_catalog.csv'
    tfrecord_locs = glob.glob(f'/home/walml/repos/zoobot/results/temp/gz2_all_actual_sim_2p5_unfiltered_300_eval_shards/*.tfrecord')
    checkpoint_dir = 'results/temp/all_actual_sim_2p5_unfiltered_300_small_first_baseline_1q_effnetv2/models/final'
    save_loc = 'temp/all_actual_sim_2p5_unfiltered_300_small_first_baseline_1q_effnetv2.csv'

    predict_on_images.predict(
        schema=schema,
        catalog_loc=catalog_loc,
        tfrecord_locs=tfrecord_locs,
        checkpoint_dir=checkpoint_dir,
        save_loc=save_loc,
        n_samples=n_samples,
        batch_size=batch_size,
        initial_size=initial_size,
        crop_size=crop_size,
        final_size=final_size,
        channels=3
    )
