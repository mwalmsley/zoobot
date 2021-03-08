import os
import logging
import glob

import tensorflow as tf

from zoobot import label_metadata, schemas
from zoobot.predictions import predict_on_tfrecords, predict_on_images


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    # replace if you defined your own schema and retrained a model from scratch
    label_cols = label_metadata.decals_label_cols  

    initial_size = 300  # 300 for paper, from tfrecord or from png (png will be resized when loaded, before preprocessing)
    crop_size = int(initial_size * 0.75)
    resize_size = 224  # 224 for paper
    channels = 3
    
    batch_size = 128  # 128 for paper, you'll need a good GPU. 8 for debugging
    n_samples = 5

    # TODO you'll want to replace these with your own paths
    # checkpoint_dir = '/home/walml/repos/zoobot_private/results/debug/models/final'
    # save_loc = '/home/walml/repos/zoobot_private/temp/debug_predictions.csv'
    checkpoint_dir = '/raid/scratch/walml/galaxy_zoo/models/decals_dr_train_labelled_m0/in_progress'
    save_loc = '/raid/scratch/walml/galaxy_zoo/temp/debug_predictions.csv'

    """
    Make predictions on a folder of images (png or jpeg)
    """

    # to make predictions on folders of images e.g. png:
    # folder_to_predict = '/media/walml/beta1/decals/png_native/dr5/J000'
    folder_to_predict = '/raid/scratch/walml/galaxy_zoo/decals/png/J000'
    file_format = 'png'  # jpg or png supported. FITS is NOT supported (PRs welcome)
    predict_on_images.predict(
        label_cols=label_cols,
        file_format=file_format,
        folder_to_predict=folder_to_predict,
        checkpoint_dir=checkpoint_dir,
        save_loc=save_loc,
        n_samples=n_samples,  # number of dropout forward passes
        batch_size=batch_size,
        initial_size=initial_size,
        crop_size=crop_size,
        resize_size=resize_size
    )
    # note that this will only work well if the images are exactly equivalent to those on which the model was trained
    # for the pretrained models, these are galaxy zoo images made from decals observations
    # to use the pretrained models on any other images, you must finetune (see README.md)

    """
    Or, make predictions on TFRecords (made using create_shards.py)

    (Only use TFRecords if you intend to make many predictions, as the time taken to make them may outweigh the faster predictions)
    """
    # tfrecord_locs = glob.glob(f'/home/walml/repos/zoobot_private/data/decals/shards/all_2p5_unfiltered_retired/eval_shards/*.tfrecord')
    # tfrecord_locs = ['/raid/scratch/walml/galaxy_zoo/shards/all_2p5_unfiltered_n2/s300_shard_0.tfrecord']
    # predict_on_tfrecords.predict(
    #     label_cols=label_cols,  # used for saving predictions, not reading the tfrecords 
    #     tfrecord_locs=tfrecord_locs,
    #     checkpoint_dir=checkpoint_dir,
    #     save_loc=save_loc,
    #     n_samples=n_samples,
    #     batch_size=batch_size,
    #     initial_size=initial_size,
    #     crop_size=crop_size,
    #     resize_size=resize_size,
    #     channels=3
    # )
    # the same caveat applies
