import os
import logging
import glob

import tensorflow as tf

from zoobot import label_metadata, schemas
from zoobot.predictions import predict_on_images


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    # replace the question_answer_pairs if you defined your own
    schema = schemas.Schema(
        question_answer_pairs=label_metadata.decals_pairs,
        dependencies=label_metadata.get_gz2_and_decals_dependencies(label_metadata.decals_pairs)
    )

    initial_size = 64  # 300 for paper
    crop_size = int(initial_size * 0.75)
    final_size = 32  # 224 for paper
    channels = 3
    
    batch_size = 8  # 128 for paper, you'll need a good GPU
    n_samples = 5

    # TODO you'll want to replace these with your own paths
    catalog_loc = '/home/walml/repos/zoobot_private/data/decals/decals_master_catalog.csv'
    tfrecord_locs = glob.glob(f'/home/walml/repos/zoobot_private/data/decals/shards/all_2p5_unfiltered_retired/eval_shards/*.tfrecord')
    checkpoint_dir = '/home/walml/repos/zoobot_private/results/debug/models/final'
    save_loc = '/home/walml/repos/zoobot_private/temp/debug_predictions.csv'

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
