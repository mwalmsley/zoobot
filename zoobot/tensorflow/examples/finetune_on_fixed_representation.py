

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

from zoobot.shared import schemas, label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import preprocess, define_model, alexnet_baseline, small_cnn_baseline
from zoobot.tensorflow.predictions import predict_on_tfrecords, predict_on_dataset
from zoobot.tensorflow.training import training_config
from zoobot.tensorflow.transfer_learning import utils
from zoobot.tensorflow.estimators import custom_layers
from zoobot.tensorflow.datasets import rings


def main(train_dataset_size=None, batch_size=256, max_galaxies_to_show=5000000):
    """
    finetune_minimal.py shows you how to train a new head on a frozen base model.
    But the frozen base model will always give the same (or at least, very similar) representation for each galaxy (as it cannot train).
    We can therefore achieve the same goal a faster way.

    We have previously calculated and saved the representation for each galaxy (see zoobot/make_predictions.py)
    We can just load these 1280-dim vectors for each galaxy and train the head model on them.
    
    As a dataset, I am using the 'advanced' ring dataset. 
    This is larger than the one from finetune_minimal.py and involves some fancy footwork - see :meth:`zoobot.datasets.rings.advanced_ring_feature_dataset` for details.
    """

    log_dir = 'results/finetune_on_fixed_representation/example_{}'.format(time.time())
    train_dataset, val_dataset, test_dataset = rings.get_advanced_ring_feature_dataset(train_dataset_size=train_dataset_size)

    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    """
    Load only the new head. There's no need for the pretrained model itself, as we're using the saved output features instead
    """
    model = tf.keras.Sequential([
      layers.InputLayer(input_shape=(1280)),  # base model dim before GlobalAveragePooling (ignoring batch)
      # TODO the following layers will likely need some experimentation to find a good combination for your problem
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

    # evaluate performance on val set, repeating to marginalise over any test-time augmentations or dropout:
    losses = []
    accuracies = []
    for _ in range(5):
      test_metrics = model.evaluate(test_dataset, verbose=0)
      losses.append(test_metrics[0])
      accuracies.append(test_metrics[0])
    logging.info('Mean test loss: {:.3f} (var {:.4f})'.format(np.mean(losses), np.var(losses)))
    logging.info('Mean test accuracy: {:.3f} (var {:.4f})'.format(np.mean(accuracies), np.var(accuracies)))

    results = {
      'batch_size': int(batch_size),
      # 'losses': np.array(losses).tolist(),
      'mean_loss': float(np.mean(losses)),
      'mean_acc': float(np.mean(accuracies)),
      'train_dataset_size': int(train_dataset_size),
      'log_dir': log_dir,
      'run_name': str(os.path.basename(log_dir))
    }
    with open('finetune_frozen_shuffled_result_timestamped_{}_{}.json'.format(train_dataset_size, np.random.randint(10000)), 'w') as f:
      json.dump(results, f)

    """
    Well done!
    
    You can now use the trained head to make predictions.

    Note that you will need to either stack the base model under the head (see finetune_advanced.py, recommended) or precalculate the base model output features on your new data.

    See make_predictions.py for a self-contained example predicting the base model output features.
    """



if __name__ == '__main__':

    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

      
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset-size', dest='dataset_size', default=None, type=int,
                        help='Max labelled galaxies to use (including resampling)')
    parser.add_argument('--batch-size', dest='batch_size', default=256, type=int,
                        help='Batch size to use for train/val/test of model')

    args = parser.parse_args()

    main(train_dataset_size=args.dataset_size, batch_size=args.batch_size)
  