

import os
import logging
import glob
import random
import shutil
import argparse
import json
import time

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
from zoobot.datasets import rings


def main(train_dataset_size=None, batch_size=128, max_galaxies_to_show=5000000):
    """
    Set up your finetuning dataset
    
    Here, I'm using galaxies tagged or not tagged as "ring" by Galaxy Zoo volunteers.
    I've already saved a pandas dataframe with:
    - rows of each galaxy
    - columns of the path (path/to/img.png) and label (1 if tagged ring, 0 if not tagged ring)
    """

    log_dir = 'results/finetune_frozen/fractionsv2_{}'.format(time.time())
    train_dataset, val_dataset, test_dataset = rings.get_ring_feature_dataset(train_dataset_size=train_dataset_size)

    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    """
    Load only the new head. There's no need for the pretrained model itself, as we're using the saved output features instead
    """
    # note - this was (7, 7, 1280) before i.e. values prior to global average pooling. Swapped to do after GAP as that's the representation I've saved.
    model = tf.keras.Sequential([
      layers.InputLayer(input_shape=(1280)),  # base model dim before GlobalAveragePooling (ignoring batch)
    #   layers.GlobalAveragePooling2D(),
      # TODO the following layers will likely need some experimentation to find a good combination for your problem
    #   layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(1, activation="sigmoid", name='sigmoid_output')  # output should be one neuron w/ sigmoid for binary classification...
      # layers.Dense(3, activation="softmax", name="softmax_output")  # ...or N neurons w/ softmax for N-class classification
    ])

    """
    Retrain the model. Only the new head will train as the rest is frozen.
    """

    epochs = max(int(max_galaxies_to_show / train_dataset_size), 1)
    patience = min(max(10, int(epochs/6)), 30)  # between 5 and 30 epochs, sliding depending on num. epochs (TODO may just set at 30, we'll see)
    logging.info('Epochs: {}'.format(epochs))
    logging.info('Early stopping patience: {}'.format(patience))

    loss = tf.keras.losses.binary_crossentropy

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal learning rate is okay
        metrics=['accuracy']
    )
    model.summary()


    train_config = training_config.TrainConfig(
      log_dir=log_dir,
      epochs=epochs,
      patience=patience  # early stopping: if val loss does not improve for this many epochs in a row, end training
    )

    # acts inplace on model
    # saves best checkpoint to train_config.logdir / checkpoints
    training_config.train_estimator(
      model,
      train_config,  # e.g. how to train epochs, patience
      train_dataset,
      val_dataset
    )

    predictions = model.predict(val_dataset)
    print(predictions[:10])

    # evaluate performance on val set, repeating to marginalise over any test-time augmentations or dropout:
    losses = []
    accuracies = []
    for _ in range(5):
      test_metrics = model.evaluate(test_dataset, verbose=0)
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
    with open('finetune_frozen_result_timestamped_{}_{}.json'.format(train_dataset_size, np.random.randint(10000)), 'w') as f:
      json.dump(results, f)

    """
    Well done!
    
    You can now use the trained head to make predictions.

    Note that you will need to either stack the base model under the head (see finetune_advanced.py, recommended) or precalculate the base model output features on your new data.

    See make_predictions.py for a self-contained example predicting the base model output features.
    """



if __name__ == '__main__':

    # configure logging
    logging.basicConfig(level=logging.INFO)
    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

      
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset-size', dest='dataset_size', default=None, type=int,
                        help='Max labelled galaxies to use (including resampling)')
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int,
                        help='Batch size to use for train/val/test of model')

    args = parser.parse_args()

    main(train_dataset_size=args.dataset_size, batch_size=args.batch_size)
  