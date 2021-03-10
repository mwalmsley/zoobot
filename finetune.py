

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



    from zoobot.estimators.custom_layers import PermaRandomRotation, PermaRandomFlip, PermaDropout, PermaRandomCrop

    # PermaRandomRotation = tf.keras.utils.register_keras_serializable()(PermaRandomRotation)

    """Boilerplate"""
    logging.basicConfig(level=logging.INFO)
    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)


    """
    Set up your finetuning dataset
    
    Here, I'm using galaxies tagged or not tagged as "ring" by Galaxy Zoo volunteers.
    """
    initial_size = 300 # images will be resized from disk (424) to this before preprocessing
    channels = 3
    batch_size = 64  # 128 for paper, you'll need a good GPU. 64 for 2070 RTX, not sure if this will mess up batchnorm tho.

    file_format = 'png'
    ring_catalog = pd.read_csv('data/ring_catalog_with_morph.csv')  # TODO change path
    ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].str.replace('/media/walml/beta1/decals/png_native/dr5', '/raid/scratch/walml/galaxy_zoo/decals/png')

    # apply selection cuts
    feat = ring_catalog['smooth-or-featured_featured-or-disk_fraction'] > 0.6
    face = ring_catalog['disk-edge-on_no_fraction'] > 0.75
    not_spiral = ring_catalog['has-spiral-arms_no_fraction'] > 0.5
    ring_catalog = ring_catalog[feat & face & not_spiral].reset_index(drop=True)
    logging.info('Labels after selection cuts: \n{}'.format(pd.value_counts(ring_catalog['ring'])))

    # select only a few non-rings to have balanced classes (hacky version for this demo, we're throwing away information)
    # ring_catalog['label'] = ring_catalog['tag_count'] > 0
    resampling_fac = 8
    rings = ring_catalog.query('ring == 1')
    split_indices = list(range(len(rings)))
    np.random.default_rng(seed=42).shuffle(split_indices)  # inplace
    split_index = int(len(rings) * 0.8)
    rings_train, rings_val = rings[:split_index], rings[split_index:]
    not_rings = ring_catalog.query('ring == 0').sample(len(rings)*resampling_fac, replace=False, random_state=42)
    not_rings_train, not_rings_val = not_rings[:split_index*resampling_fac], not_rings[split_index*resampling_fac:]

    ring_catalog_train = pd.concat([rings_train] * resampling_fac + [not_rings_train])
    ring_catalog_val = pd.concat([rings_val, not_rings_val.sample(len(rings_val), random_state=42)])  # only need one val batch

    # shuffle
    ring_catalog_train = ring_catalog_train.sample(len(ring_catalog_train), random_state=42)
    ring_catalog_val = ring_catalog_val.sample(len(ring_catalog_val), random_state=42)

    logging.info('Train labels: \n {}'.format(pd.value_counts(ring_catalog_train['ring'])))
    logging.info('Val labels: \n {}'.format(pd.value_counts(ring_catalog_val['ring'])))


    # not_rings = ring_catalog.query('tag_count == 0').sample(len(rings), replace=False)
    # ring_catalog_balanced = pd.concat([rings, not_rings).reset_index()

    # paths = list(ring_catalog_balanced['local_png_loc'])
    # labels = list(ring_catalog_balanced['tag_count'] > 0)

    # logging.info('Labels: \n{}'.format(pd.value_counts(labels)))
    # paths_train, paths_val, labels_train, labels_val= train_test_split(paths, labels, test_size=0.2, random_state=1)

    paths_train, paths_val = list(ring_catalog_train['local_png_loc']), list(ring_catalog_val['local_png_loc'])
    labels_train, labels_val = list(ring_catalog_train['ring']), list(ring_catalog_val['ring'])

    # check for no overlap
    assert set(paths_train).intersection(set(paths_val)) == set()

    raw_train_dataset = image_datasets.get_image_dataset(paths_train, file_format=file_format, initial_size=initial_size, batch_size=batch_size, labels=labels_train)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, initial_size=initial_size, batch_size=batch_size, labels=labels_val)

    # small datasets that fit in memory can be cached before augmentations
    raw_train_dataset = raw_train_dataset.cache()
    raw_val_dataset = raw_val_dataset.cache()
    # do a dummy read to trigger the cache
    _ = [x for x in raw_train_dataset.as_numpy_iterator()]
    _ = [x for x in raw_val_dataset.as_numpy_iterator()]
    logging.info('Cache complete')
  
    input_config = preprocess.PreprocessingConfig(
        label_cols=['label'],  # image_datasets.get_image_dataset will put the labels arg under 'label' key for each batch
        input_size=initial_size,
        channels=3,
        greyscale=True
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, input_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, input_config)

    """
    Load the pretrained model (without the "head" output layer), freeze it, and add a new head with 2 softmax neurons
    """
    # TODO you'll want to replace these with your own paths
    # pretrained_checkpoint = '/home/walml/repos/zoobot_private/results/debug/models/final'
    # pretrained_checkpoint = '/media/walml/alpha/beta/decals/long_term/models/decals_dr_train_labelled_m0/in_progress'
    # log_dir = '/home/walml/repos/zoobot_private/results/temp/finetune_debug'
    pretrained_checkpoint = '/raid/scratch/walml/galaxy_zoo/models/decals_dr_train_labelled_m0/in_progress'

    # experiment_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured'
    # version = 1
    # if not os.path.isdir(experiment_dir):
    #   os.mkdir(experiment_dir)
    # while True:
    #   log_dir = os.path.join(experiment_dir, 'v{}'.format(version))
    #   if os.path.isdir(log_dir):
    #     version+=1
    #   else:
    #     # os.mkdir(log_dir)
    #     break
    # log_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured_keras'
    log_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured_temp'

    head_epochs = 25
    full_epochs = 25

    # should match how the model was trained
    original_output_dim = 34
    crop_size = int(initial_size * 0.75)
    resize_size = 224  # 224 for paper

    # get headless model (inc. augmentations)
    logging.info('Loading pretrained model from {}'.format(pretrained_checkpoint))
    base_model = define_model.load_model(
      pretrained_checkpoint,
      include_top=False,
      input_size=initial_size,  # preprocessing above did not change size
      crop_size=crop_size,  # model augmentation layers apply a crop...
      resize_size=resize_size,  # ...and then apply a resize
      output_dim=original_output_dim
    )
    # print('Before freeze')
    # base_model.summary()
    # utils.freeze_model(model)  # not including the new head, which will be trained
    base_model.trainable = False
    # print('After freeze')
    # base_model.summary()

    # l2_lambda = 0.01
    # regularizer = regularizers.l2(l2=l2_lambda)  # v bad
    regularizer = None
  
    new_head = tf.keras.Sequential([
      layers.InputLayer(input_shape=(7,7,1280)),  # base model dim before GlobalAveragePooling (ignoring batch)
      layers.GlobalAveragePooling2D(),
      layers.Dropout(0.75),
      # layers.BatchNormalization(),
      # layers.Dense(128, activation='relu'),
      layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
      layers.Dropout(0.75),
      layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
      layers.Dropout(0.75),
      layers.Dense(1, activation="sigmoid", name='sigmoid_output')
      # layers.Dense(2, activation="softmax", name="softmax_output")  # predicting 2 classes (0=not ring, 1=ring)
    ])

    model = tf.keras.Sequential([
      tf.keras.Input(shape=(initial_size,initial_size,1)),
      base_model,
      new_head
    ])

    # """
    # Retrain the model. Only the new head will train as the rest is frozen.
    # """

    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Input(shape=(initial_size, initial_size, 1)))
    # define_model.add_preprocessing_layers(model, crop_size=crop_size,
    #                          resize_size=resize_size)  # inplace
    # # model.add(alexnet_baseline.alexnet_model(img_shape=(resize_size, resize_size, 1), n_classes=1))
    # model.add(small_cnn_baseline.small_cnn())

    model.step = tf.Variable(
      0, dtype=tf.int64, name='model_step', trainable=False)

    loss = tf.keras.losses.binary_crossentropy

    # effnet_layers = model.layers[0].layers[3].layers[0].layers
    # batchnorm_layers = [l for l in effnet_layers if isinstance(l, layers.BatchNormalization)]
    # print(batchnorm_layers[0].weights)

    # optimizer = tf.keras.optimizers.Adam(
    #   learning_rate=tf.Variable(0.005),
    #   beta_1=tf.Variable(0.9),
    #   beta_2=tf.Variable(0.999),
    #   epsilon=tf.Variable(1e-7),
    # )
    # optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
    # optimizer.decay = tf.Variable(0.0)  # Adam.__init__ assumes ``decay`` is a float object, so this needs to be converted to tf.Variable **after** __init__ method.

    # model.compile(
    #     loss=loss,
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),  # normal learning rate is okay
    #     # optimizer=optimizer,
    #     metrics=['accuracy']
    # )
    # print('With trainable head')
    # model.summary()

    # """Train only the new head"""



    # # ckpt = tf.train.Checkpoint(model=model)
    # # manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=3)

    # train_config = training_config.TrainConfig(
    #   log_dir=log_dir,
    #   epochs=head_epochs,
    #   patience=30
    # )

    # # inplace on model
    # training_config.train_estimator(
    #   model,  # inplace
    #   train_config,  # e.g. how to train epochs, patience
    #   input_config,  # how to preprocess data before model (currently, barely at all)
    #   train_dataset,
    #   val_dataset
    # )

    # losses = []
    # for _ in range(10):
    #   losses.append(model.evaluate(val_dataset)[0])
    # print(np.mean(losses), np.var(losses))

    # model.save(os.path.join(log_dir, 'keras'))

    # # # ckpt.step.assign(model.step)
    # manager.save()

    # load_status = ckpt.restore(manager.latest_checkpoint)
    # load_status.assert_consumed()
    # load_status.assert_existing_objects_matched()

    # from zoobot.estimators.efficientnet_standard import get_dropout
    # custom_objects = {
    #   "PermaRandomRotation": PermaRandomRotation,
    #   "PermaRandomFlip": PermaRandomFlip,
    #   "PermaRandomCrop": PermaRandomCrop,
    #   "PermaDropout": PermaDropout,
    #   "FixedDropout": get_dropout()
    # }
    # with tf.keras.utils.CustomObjectScope(custom_objects):
    #   tf.keras.models.load_model(os.path.join(log_dir, 'keras'))

    # losses = []
    # for _ in range(10):
    #   losses.append(model.evaluate(val_dataset)[0])
    # print(np.mean(losses), np.var(losses))


    # exit()

    # if allowed to finish, best model will be saved to log_dir/checkpoints/models/final for later use (see make_predictions.py)
    # else, latest model will be log_dir/checkpoints
    # partially_retrained_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured/v15/checkpoints/final_model/final'
    # partially_retrained_dir = os.path.join(log_dir, 'checkpoints')
    partially_retrained_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured/v42/checkpoints'
    # partially_retrained_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured/v15/checkpoints'  # the checkpoint, not the folder - tf understands
    # weights_loc = os.path.join(log_dir, 'checkpoints')
    define_model.load_weights(model=model, weights_loc=partially_retrained_dir, expect_partial=True)

    # effnet_layers = model.layers[0].layers[3].layers[0].layers
    # batchnorm_layers = [l for l in effnet_layers if isinstance(l, layers.BatchNormalization)]
    # print(batchnorm_layers[0].weights)

    # model.trainable = True on effnet? 
    # model.layers[3].trainable = True
    # utils.unfreeze_model(model)  # inplace

    # losses = []
    # for _ in range(10):
    #   losses.append(model.evaluate(val_dataset)[0])
    # print(np.mean(losses), np.var(losses))

    # # and do it again
    # define_model.load_weights(model=model, weights_loc=partially_retrained_dir, expect_partial=True)
    # losses = []
    # for _ in range(10):
    #   losses.append(model.evaluate(val_dataset)[0])
    # print(np.mean(losses), np.var(losses))

    # # and save and do it again
    # expected_save_dir = os.path.join(log_dir, 'checkpoints')
    # model.save_weights(expected_save_dir)
    # logging.info('Reloading from {}'.format(expected_save_dir))
    # define_model.load_weights(model=model, weights_loc=expected_save_dir, expect_partial=True)
    # losses = []
    # for _ in range(10):
    #   losses.append(model.evaluate(val_dataset)[0])
    # print(np.mean(losses), np.var(losses))

    model.summary()
    utils.unfreeze_model(model, unfreeze_names=['top'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7'])  # 1m free parameters, way too many for any learning rate
    model.summary()

    # # must recompile for trainable to update, and for lower learning rate
    print('recompiling')
    model.compile(
        loss=loss,
        # optimizer=tf.keras.optimizers.SGD(lr=0),  # dummy
        optimizer=tf.keras.optimizers.Adam(lr=1e-6),
        metrics=['accuracy']
    )
    # print('Unfrozen except for batch norm')
    model.summary()

    # utils.check_batchnorm_frozen(model)

    print('Before unfrozen finetuning')
    losses = []
    for _ in range(10):
      losses.append(model.evaluate(val_dataset)[0])
    print(np.mean(losses), np.var(losses))
    # exit()


    log_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured_unfreeze'
    train_config_full = training_config.TrainConfig(
      log_dir=log_dir,
      epochs=full_epochs,
      patience=10
    )

    
    print('pretending to train')
    training_config.train_estimator(
      model,  # inplace
      train_config_full,  # e.g. how to train epochs, patience
      input_config,  # how to preprocess data before model (currently, barely at all)
      train_dataset,
      val_dataset
    )

    print('after unfrozen finetuning')
    losses = []
    for _ in range(10):
      losses.append(model.evaluate(val_dataset)[0])
    print(np.mean(losses), np.var(losses))

    # effnet_layers = model.layers[0].layers[3].layers[0].layers
    # batchnorm_layers = [l for l in effnet_layers if isinstance(l, layers.BatchNormalization)]
    # print(batchnorm_layers[0].weights)