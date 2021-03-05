

import os
import logging
import glob

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from zoobot import label_metadata, schemas
from zoobot.data_utils import image_datasets
from zoobot.estimators import preprocess, define_model
from zoobot.predictions import predict_on_tfrecords, predict_on_images
from zoobot.training import training_config
from zoobot.transfer_learning import utils


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)


    # TODO you'll want to replace these with your own paths
    pretrained_checkpoint = '/home/walml/repos/zoobot_private/results/debug/models/final'
    log_dir = '/home/walml/repos/zoobot_private/results/temp/finetune_debug'
    epochs = 100

    # can vary
    initial_size = 424  # loading 424x424 pixel png, not 300x300 tfrecord
    channels = 3
    batch_size = 8  # 128 for paper, you'll need a good GPU
    n_samples = 5

    # should match how the model was trained
    original_output_dim = 34
    crop_size = int(initial_size * 0.75)
    resize_size = 32  # 224 for paper

    # get datasets
    file_format = 'png'
    ring_catalog = pd.read_csv('/home/walml/repos/recommender_hack/ring_tag_catalog_all.csv')  # TODO change path
    # select only a few non-rings to have balanced classes (hacky version for this demo, we're throwing away information)
    rings = ring_catalog.query('tag_count > 0')
    not_rings = ring_catalog.query('tag_count == 0')
    ring_catalog_balanced = pd.concat([rings, not_rings.sample(len(rings))]).reset_index()

    paths = list(ring_catalog_balanced['local_png_loc'])
    labels = list(ring_catalog_balanced['tag_count'] > 0)
    paths_train, paths_val, labels_train, labels_val= train_test_split(paths, labels, test_size=0.2, random_state=1)
    num_classes = 2  # for retrained head

    raw_train_dataset = image_datasets.get_image_dataset(paths_train, file_format=file_format, initial_size=initial_size, batch_size=batch_size, labels=labels_train)
    raw_train_dataset = raw_train_dataset.shuffle(64)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, initial_size=initial_size, batch_size=batch_size, labels=labels_val)
  
    input_config = preprocess.PreprocessingConfig(
        name='from_png',
        label_cols=['label'],  # image_datasets.get_image_dataset will put the labels arg under 'label' key for each batch
        input_size=initial_size,
        channels=3,
        greyscale=True
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, input_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, input_config)

    # get headless model (inc. augmentations)
    model = define_model.load_model(
      pretrained_checkpoint,
      include_top=False,
      input_size=initial_size,  # preprocessing above did not change size
      crop_size=crop_size,  # model augmentation layers apply a crop...
      resize_size=resize_size,  # ...and then apply a resize
      output_dim=original_output_dim
    )

    utils.freeze_model(model)  # not including the new head, which will be trained
    new_head = utils.classification_head(num_classes=num_classes)
    model.add(new_head)
    # utils.get_headless_frozen_model(base_model, new_head)

    loss = tf.keras.losses.binary_crossentropy

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam()  # normal learning rate is okay
    )
    model.summary()

    train_config = training_config.TrainConfig(
      log_dir=log_dir,
      epochs=epochs,
      patience=10
    )

    # inplace on model
    training_config.train_estimator(
      model, 
      train_config,  # e.g. how to train epochs, patience
      input_config,  # how to preprocess data before model (currently, barely at all)
      train_dataset,
      val_dataset
    )
