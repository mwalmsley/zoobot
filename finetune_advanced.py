

import os
import logging
import glob
import random

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split

from zoobot import label_metadata, schemas
from zoobot.data_utils import image_datasets
from zoobot.estimators import preprocess, define_model, alexnet_baseline, small_cnn_baseline
from zoobot.predictions import predict_on_tfrecords, predict_on_images
from zoobot.training import training_config
from zoobot.transfer_learning import utils
from zoobot.estimators import custom_layers


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    """
    Set up your finetuning dataset
    
    Here, I'm using galaxies tagged or not tagged as "ring" by Galaxy Zoo volunteers.

    This time, instead of balancing the classes by dropping most of the non-rings, I'm repeating the rings by a factor of ~8
    This includes more information but needs some footwork to make sure that no repeated ring ends up in both the train and test sets
    """
    requested_img_size = 300 # images will be resized from disk (424) to this before preprocessing
    channels = 3
    batch_size = 64  # 128 for paper, you'll need a good GPU. 64 for 2070 RTX, not sure if this will mess up batchnorm tho.

    file_format = 'png'
    ring_catalog = pd.read_csv('data/ring_catalog_with_morph.csv', dtype={'ring': int})
    ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].str.replace('/media/walml/beta1/decals/png_native/dr5', '/raid/scratch/walml/galaxy_zoo/decals/png')

    # apply selection cuts
    feat = ring_catalog['smooth-or-featured_featured-or-disk_fraction'] > 0.6
    face = ring_catalog['disk-edge-on_no_fraction'] > 0.75
    not_spiral = ring_catalog['has-spiral-arms_no_fraction'] > 0.5
    ring_catalog = ring_catalog[feat & face & not_spiral].reset_index(drop=True)
    logging.info('Labels after selection cuts: \n{}'.format(pd.value_counts(ring_catalog['ring'])))

    rings = ring_catalog.query('ring == 1')
    resampling_fac = int((len(ring_catalog) - len(rings)) / (len(rings)))
    logging.info('Resampling fraction: {}'.format(resampling_fac))

    split_index = int(len(rings) * 0.8)
    rings_train = pd.concat([rings[:split_index]] * resampling_fac)  # resample (repeat) rings
    rings_val = rings[split_index:]  # do not resample validation, instead will select same num. of non-rings

    seed = 1
    not_rings = ring_catalog.query('ring == 0')
    not_rings_train = not_rings[:len(rings_train)]
    not_rings_val = not_rings[len(rings_train):].sample(len(rings_val), replace=False, random_state=seed)

    ring_catalog_train = pd.concat([rings_train, not_rings_train])
    ring_catalog_val = pd.concat([rings_val, not_rings_val.sample(len(rings_val), random_state=seed)])  # only need one val batch

    # shuffle
    ring_catalog_train = ring_catalog_train.sample(len(ring_catalog_train), random_state=seed)
    ring_catalog_val = ring_catalog_val.sample(len(ring_catalog_val), random_state=seed)

    logging.info('Train labels: \n {}'.format(pd.value_counts(ring_catalog_train['ring'])))
    logging.info('Val labels: \n {}'.format(pd.value_counts(ring_catalog_val['ring'])))

    paths_train, paths_val = list(ring_catalog_train['local_png_loc']), list(ring_catalog_val['local_png_loc'])
    labels_train, labels_val = list(ring_catalog_train['ring']), list(ring_catalog_val['ring'])

    # check that no repeated rings ended up in both the train and val sets
    assert set(paths_train).intersection(set(paths_val)) == set()

    raw_train_dataset = image_datasets.get_image_dataset(paths_train, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_train)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_val)

    # small datasets that fit in memory can be cached before augmentations
    # this speeds up training
    raw_train_dataset = raw_train_dataset.cache()
    raw_val_dataset = raw_val_dataset.cache()
    # do a dummy read to trigger the cache
    _ = [x for x in raw_train_dataset.as_numpy_iterator()]
    _ = [x for x in raw_val_dataset.as_numpy_iterator()]
    logging.info('Cache complete')
  
    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=['label'],  # image_datasets.get_image_dataset will put the labels arg under the 'label' key for each batch
        input_size=requested_img_size,
        normalise_from_uint8=True,  # divide by 255
        make_greyscale=True,  # take the mean over RGB channels
        permute_channels=False  # swap channels around randomly (no need when making greyscale anwyay)
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, preprocess_config)


    """
    Load the pretrained model (without the "head" output layer), freeze it, and add a new head.
    This is exactly the same as finetune_minimal.py
    """

    pretrained_checkpoint = 'data/pretrained_models/gz_decals_full_m0/in_progress'
    # should match how the model was trained
    crop_size = int(requested_img_size * 0.75)
    resize_size = 224  # 224 for paper

    log_dir = 'results/finetune_advanced'  # TODO you'll want to replace this with your own path
    log_dir_head = os.path.join(log_dir, 'head_only')
    for d in [log_dir, log_dir_head]:
      if not os.path.isdir(d):
        os.mkdir(d)

    # get headless model (inc. augmentations)
    logging.info('Loading pretrained model from {}'.format(pretrained_checkpoint))
    base_model = define_model.load_model(
      pretrained_checkpoint,
      include_top=False,  # do not include the head used for GZ DECaLS - we will add our own head
      input_size=requested_img_size,  # the preprocessing above did not change size
      crop_size=crop_size,  # model augmentation layers apply a crop...
      resize_size=resize_size,  # ...and then apply a resize
      output_dim=None  # headless so no effect
    )

    base_model.trainable = False  # freeze the headless model (no training allowed)

    # I am not using test-time dropout (MC Dropout) on the head as 0.75 would be way too aggressive and reduce performance
    new_head = tf.keras.Sequential([
      layers.InputLayer(input_shape=(7,7,1280)),  # base model dim before GlobalAveragePooling (ignoring batch)
      layers.GlobalAveragePooling2D(),
      # TODO the following layers will likely need some experimentation to find a good combination for your problem
      layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(1, activation="sigmoid", name='sigmoid_output')  # output should be one neuron w/ sigmoid for binary classification...
      # layers.Dense(3, activation="softmax", name="softmax_output")  # ...or N neurons w/ softmax for N-class classification
    ])

    # stick the new head on the pretrained base model
    model = tf.keras.Sequential([
      tf.keras.Input(shape=(requested_img_size, requested_img_size, 1)),
      base_model,
      new_head
    ])

    # Retrain the model. Only the new head will train as the rest is frozen.
    epochs = 120
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal learning rate is okay
        metrics=['accuracy']
    )
    model.summary()

    train_config = training_config.TrainConfig(
      log_dir=log_dir_head,
      epochs=epochs,
      patience=int(epochs/6)  # early stopping: if val loss does not improve for this many epochs in a row, end training
    )

    training_config.train_estimator(
      model,
      train_config,  # e.g. how to train epochs, patience
      train_dataset,
      val_dataset
    )

    losses = []
    for _ in range(5):
      losses.append(model.evaluate(val_dataset)[0])
    logging.info('Mean validation loss: {:.3f} (var {:.4f})'.format(np.mean(losses), np.var(losses)))
    # should train to a loss of around 0.54, equivalent to 75-80% accuracy on the (class-balanced) validation set

    """
    The head has been retrained.
    It may be possible to further improve performance by unfreezing the layers just before the head in the base model,
    and training with a very low learning rate (to avoid overfitting).

    If you want to focus on this step, you can comment the training above and instead simply load the previous model (including the finetuned head)
    """
    define_model.load_weights(model=model, weights_loc=os.path.join(log_dir_head, 'checkpoint'), expect_partial=True)

    # you can unfreeze layers like so:
    utils.unfreeze_model(model, unfreeze_names=['top'])
    # or more...
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7', 'block6'])
    # note that the number of free parameters increases very quickly!

    logging.info('Recompiling with lower learning rate and trainable upper layers')
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=1e-4),  # very low learning rate is crucial
        metrics=['accuracy']
    )

    model.summary()

    # losses = []
    # for _ in range(5):
    #   losses.append(model.evaluate(val_dataset)[0])
    # logging.info('Before unfrozen finetuning: {}'.format(np.mean(losses), np.var(losses))

    log_dir_full = os.path.join(log_dir, 'full')
    train_config_full = training_config.TrainConfig(
      log_dir=log_dir_full,
      epochs=50,
      patience=15
    )

    training_config.train_estimator(
      model,  # inplace
      train_config_full,
      train_dataset,
      val_dataset
    )

    losses = []
    for _ in range(5):
      losses.append(model.evaluate(val_dataset)[0])
    logging.info('After unfrozen finetuning: {:.3f} (var={:.4f})'.format(np.mean(losses), np.var(losses)))
