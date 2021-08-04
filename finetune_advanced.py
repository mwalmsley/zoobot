

import os
import logging
import glob
import random
import argparse
import time
import json

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split

from zoobot import label_metadata, schemas
from zoobot.estimators import preprocess, define_model, alexnet_baseline, small_cnn_baseline
from zoobot.predictions import predict_on_tfrecords, predict_on_images
from zoobot.training import training_config
from zoobot.transfer_learning import utils
from zoobot.estimators import custom_layers
from zoobot.datasets import rings


def main(batch_size, requested_img_size, train_dataset_size, max_galaxies_to_show=5000000, greyscale=True):


    # logging.warning('Using imagenet so setting greyscale = False')
    # greyscale = False
    if greyscale:
      channels = 1
    else:
      channels = 3
      logging.warning('Using color images, channels=3')

    """
    Set up your finetuning dataset
    
    Here, I'm using galaxies tagged or not tagged as "ring" by Galaxy Zoo volunteers.

    This time, instead of balancing the classes by dropping most of the non-rings, I'm repeating the rings by a factor of ~8
    This includes more information but needs some footwork to make sure that no repeated ring ends up in both the train and test sets
    """
    raw_train_dataset, raw_val_dataset, raw_test_dataset = rings.get_advanced_ring_image_dataset(
      batch_size=batch_size, requested_img_size=requested_img_size, train_dataset_size=train_dataset_size)

    # small datasets that fit in memory can be cached before augmentations
    # this speeds up training

    use_cache = False  # sequential if's for awkward None/int type
    if train_dataset_size is not None:  # when None, load all -> very many galaxies
      if train_dataset_size < 10000:
        use_cache = True

    if use_cache:
      raw_train_dataset = raw_train_dataset.cache()
      raw_val_dataset = raw_val_dataset.cache()
      raw_test_dataset = raw_test_dataset.cache()
      # read once (and don't use) to trigger the cache
      _ = [x for x in raw_train_dataset.as_numpy_iterator()]
      _ = [x for x in raw_val_dataset.as_numpy_iterator()]
      _ = [x for x in raw_test_dataset.as_numpy_iterator()]
      logging.info('Cache complete')
    else:
      logging.warning('Requested {} training images (if None, using all available). Skipping cache.'.format(train_dataset_size))
    
    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=['label'],  # image_datasets.get_image_dataset will put the labels arg under the 'label' key for each batch
        input_size=requested_img_size,
        normalise_from_uint8=True,  # divide by 255
        make_greyscale=greyscale,  # take the mean over RGB channels
        permute_channels=False  # swap channels around randomly (no need when making greyscale anwyay)
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, preprocess_config)
    test_dataset = preprocess.preprocess_dataset(raw_test_dataset, preprocess_config)

    """
    Load the pretrained model (without the "head" output layer), freeze it, and add a new head.
    This is exactly the same as finetune_minimal.py
    """

    pretrained_checkpoint = 'data/pretrained_models/decals_dr_trained_on_all_labelled_m0/in_progress'  # TODO can I just have a headless imagenet checkpoint?
    # should match how the model was trained
    crop_size = int(requested_img_size * 0.75)
    resize_size = 224  # 224 for paper

    run_name = 'from_scratch_{}'.format(time.time())
    log_dir = os.path.join('results/finetune_advanced', run_name)
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
      output_dim=None , # headless so no effect
      channels=channels
    )
    base_model.trainable = False  # freeze the headless model (no training allowed)

    # TODO temporarily replace with blank model - scratch mode
    # base_model = define_model.get_model(
    #   output_dim=None,
    #   input_size=requested_img_size,
    #   crop_size=crop_size,
    #   resize_size=resize_size,
    #   weights_loc=None,
    #   include_top=False,
    #   channels=channels
    # )
    ## TODO temporarily train the whole thing
    # base_model.trainable = True

    # # TODO temporarily replace with imagenet pretrained model - scratch mode
    # base_model = define_model.get_model(
    #   output_dim=None,
    #   input_size=requested_img_size,
    #   crop_size=crop_size,
    #   resize_size=resize_size,
    #   weights_loc=None,
    #   include_top=False,
    #   channels=channels,
    #   use_imagenet_weights=True
    # )
    # base_model.trainable = False  # freeze the headless model (no training allowed)




    # I am not using test-time dropout (MC Dropout) on the head as 0.75 would be way too aggressive and reduce performance
    new_head = tf.keras.Sequential([
      layers.InputLayer(input_shape=(7,7,1280)),  # base model dim before GlobalAveragePooling (ignoring batch)
      layers.GlobalAveragePooling2D(),  # quirk of code that this is included - could have been input_shape=(1280) and skipped GAP2D
      # TODO the following layers will likely need some experimentation to find a good combination for your problem
      # layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(1, activation="sigmoid", name='sigmoid_output')  # output should be one neuron w/ sigmoid for binary classification...
      # layers.Dense(3, activation="softmax", name="softmax_output")  # ...or N neurons w/ softmax for N-class classification
    ])

    # stick the new head on the pretrained base model
    model = tf.keras.Sequential([
      tf.keras.Input(shape=(requested_img_size, requested_img_size, channels)),
      base_model,
      new_head
    ])

    # Retrain the model. Only the new head will train as the rest is frozen.

    epochs = max(int(max_galaxies_to_show / train_dataset_size), 1)
    patience = min(max(10, int(epochs/6)), 30)  # between 5 and 30 epochs, sliding depending on num. epochs (TODO may just set at 30, we'll see)
    logging.info('Epochs: {}'.format(epochs))
    logging.info('Early stopping patience: {}'.format(patience))

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal learning rate is okay
        metrics=['accuracy']
    )
    model.summary()

    train_config = training_config.TrainConfig(
      log_dir=log_dir_head,
      epochs=epochs,
      patience=patience  # early stopping: if val loss does not improve for this many epochs in a row, end training
    )

    training_config.train_estimator(
      model,
      train_config,  # e.g. how to train epochs, patience
      train_dataset,
      val_dataset
    )

    # evaluate performance on test set, repeating to marginalise over any test-time augmentations or dropout:
    losses = []
    accuracies = []
    for _ in range(5):
      test_metrics = model.evaluate(test_dataset.repeat(3), verbose=0)
      losses.append(test_metrics[0])
      accuracies.append(test_metrics[0])
    logging.info('Mean test loss: {:.3f} (var {:.4f})'.format(np.mean(losses), np.var(losses)))
    logging.info('Mean test accuracy: {:.3f} (var {:.4f})'.format(np.mean(accuracies), np.var(accuracies)))
    # should train to a loss of around 0.54, equivalent to 75-80% accuracy on the (class-balanced) validation set

    results = {
      'batch_size': int(batch_size),
      # 'losses': np.array(losses).tolist(),
      'mean_loss': float(np.mean(losses)),
      'mean_acc': float(np.mean(accuracies)),
      'train_dataset_size': int(train_dataset_size),
      'log_dir': log_dir,
      'run_name': str(os.path.basename(log_dir))
    }
    with open('{}_result_timestamped_{}_{}.json'.format(run_name, train_dataset_size, np.random.randint(10000)), 'w') as f:
      json.dump(results, f)

    exit()


    """
    The head has been retrained.
    It may be possible to further improve performance by unfreezing the layers just before the head in the base model,
    and training with a very low learning rate (to avoid overfitting).

    If you want to focus on this step, you can comment the training above and instead simply load the previous model (including the finetuned head)
    """
    define_model.load_weights(model=model, checkpoint_loc=os.path.join(log_dir_head, 'checkpoint'), expect_partial=True)

    # you can unfreeze layers like so:
    utils.unfreeze_model(model, unfreeze_names=['top'])
    # or more...
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7', 'block6'])
    # note that the number of free parameters increases very quickly!

    logging.info('Recompiling with lower learning rate and trainable upper layers')
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # very low learning rate is crucial
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
    # a little odd that currently, the train loss does change but the val loss almost doesn't? TODO
    training_config.train_estimator(
      model,  # inplace
      train_config_full,
      train_dataset,
      val_dataset
    )

    losses = []
    for _ in range(5):
      losses.append(model.evaluate(test_dataset, verbose=0)[0])
    logging.info('After unfrozen finetuning: {:.3f} (var={:.4f})'.format(np.mean(losses), np.var(losses)))


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      
    parser = argparse.ArgumentParser(description='Transfer learning and finetuning from pretrained model on ring dataset')
    parser.add_argument('--dataset-size', dest='train_dataset_size', default=None, type=int,
                        help='Max labelled galaxies to use (including resampling)')
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int,
                        help='Batch size to use for train/val/test of model')
    parser.add_argument('--img-size', dest='requested_img_size', default=300, type=int,
                        help='Image size before conv layers i.e. after loading (from 424, by default) and cropping.')

    args = parser.parse_args()

    main(
      batch_size=args.batch_size,
      requested_img_size=args.requested_img_size,
      train_dataset_size=args.train_dataset_size,
      greyscale=True
    )