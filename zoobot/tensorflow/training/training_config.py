import logging
import os
import copy
import shutil
import time
from functools import partial
from typing import List

import tensorflow as tf
import pandas as pd
import matplotlib
import numpy as np

from zoobot.tensorflow.estimators import preprocess, efficientnet_standard, efficientnet_custom, custom_layers, define_model, custom_callbacks
from zoobot.shared import schemas

# augmentations now done via keras Input layers instead - see define_model.py
# globals shared between train and test input configurations
# MAX_SHIFT = 30  # not implemented
# MAX_SHEAR = np.pi/4.  # not implemented
# ZOOM = (1/1.65, 1/1.4)  # keras interprets zoom the other way around to normal humans, for some reason: zoom < 1 = magnification


class TrainConfig():
    def __init__(
            self,
            epochs=1500,  # rely on earlystopping callback
            min_epochs=0,
            patience=10,
            log_dir='runs/default_run_{}'.format(time.time()),
            save_freq='epoch'
    ): 
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.patience = patience
        self.log_dir = log_dir
        self.save_freq = save_freq


    # TODO move to shared utilities
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])


    # don't decorate, this is session creation point

def train_estimator(model, train_config, train_dataset, test_dataset, extra_callbacks=[], eager=False, verbose=2):
    """
    Train and evaluate a model.

    Includes tensorboard logging (to log_dir/tensorboard).
    Includes checkpointing (named log_dir/checkpoint), with the rolling best val loss checkpoint saved.
    Includes early stopping according to train_config.patience.

    Args:
        model (tf.keras.Model): model to train. Must already be compiled with model.compile(loss, optimizer)
        train_config (TrainConfig): parameters controlling training procedure e.g. epochs, early stopping
        train_dataset (tf.data.Dataset): yielding batched tuples of (galaxy images, labels)
        test_dataset (tf.data.Dataset): yielding batched tuples of (galaxy images, labels)
        extra_callbacks (list): any extra callbacks to use when training the model. See e.g. tf.keras.callbacks.
        eager (bool, optional): If True, train in eager mode - slow, but helpful for debugging. Defaults to False.
        verbose (int, optional): 1 for progress bar, useful for local training. 2 for one line per epoch, useful for scripts. Defaults to 2.

    Returns:
        None
    """

    if not os.path.isdir(train_config.log_dir):
        os.mkdir(train_config.log_dir)

    # will create a multi-file checkpoint like {checkpoint.index, checkpoint.data.00000-00001, ...}
    checkpoint_name = os.path.join(train_config.log_dir, 'checkpoint')

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(train_config.log_dir, 'tensorboard'),
            histogram_freq=0,  # don't log all the internal histograms, possibly slow
            write_images=False,  # this actually writes the weights, terrible name
            write_graph=False,
            # profile_batch='2,10' 
            profile_batch=0   # i.e. disable profiling
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_name,
            monitor='val_loss',
            mode='min',
            save_freq=train_config.save_freq,
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=train_config.patience),
        tf.keras.callbacks.TerminateOnNaN(),
        custom_callbacks.UpdateStepCallback(
            batch_size=next(iter(train_dataset))[0].shape[0]  # grab the first batch, 0th tuple element (the images), 0th dimension, to check the batch size
        )
    ] + extra_callbacks

    # attribute used by the callbacks to track the current step when writing to tensorboard.
    model.step = tf.Variable(
      0, dtype=tf.int64, name='model_step', trainable=False
    )

    # https://www.tensorflow.org/tensorboard/scalars_and_keras
    fit_summary_writer = tf.summary.create_file_writer(os.path.join(train_config.log_dir, 'manual_summaries'))
    # pylint: disable=not-context-manager
    with fit_summary_writer.as_default(): 
        # pylint: enable=not-context-manager
        # for debugging
        if eager:
            logging.warning('Running in eager mode')
            model.run_eagerly = True
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model

        model.fit(
            train_dataset,
            validation_data=test_dataset.repeat(2),  # reduce variance from dropout, augs
            epochs=train_config.epochs,
            callbacks=callbacks,
            verbose=verbose
        )

    logging.info('All epochs completed - finishing gracefully')
    # note that the BEST model is saved as the latest checkpoint, but self.model is the LAST model after training completes
    # to set self.model to the best model, load the latest checkpoint 
    logging.info('Loading and returning (best) model')
    model.load_weights(checkpoint_name)  # inplace

    return model
