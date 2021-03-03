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

from zoobot.estimators import input_utils, efficientnet_standard, efficientnet_custom, custom_layers, define_model, custom_callbacks
from zoobot import schemas

# augmentations now done via keras Input layers instead - see define_model.py
# globals shared between train and test input configurations
# MAX_SHIFT = 30  # not implemented
# MAX_SHEAR = np.pi/4.  # not implemented
# ZOOM = (1/1.65, 1/1.4)  # keras interprets zoom the other way around to normal humans, for some reason: zoom < 1 = magnification


class RunEstimatorConfig():
    def __init__(
            self,
            initial_size,
            final_size,
            crop_size,
            schema: schemas.Schema,
            channels=3,
            epochs=1500,  # rely on earlystopping callback
            train_steps=30,
            eval_steps=3,
            batch_size=10,
            min_epochs=0,
            patience=10,
            log_dir='runs/default_run_{}'.format(time.time()),
            save_freq=10,
            weights_loc=None
    ):  # TODO refactor for consistent order
        self.initial_size = initial_size
        self.final_size = final_size
        self.crop_size = crop_size
        self.channels = channels
        self.schema = schema
        self.epochs = epochs
        self.train_batches = train_steps
        self.eval_batches = eval_steps
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.weights_loc = weights_loc
        self.patience = patience
        self.min_epochs = min_epochs
        self.train_config = None
        self.eval_config = None
        self.model = None

    
    def assemble(self, train_config, eval_config, model):
        self.train_config = train_config
        self.eval_config = eval_config
        self.model = model
        assert self.is_ready_to_train()

    def is_ready_to_train(self):
        # TODO can make this check much more comprehensive
        return (self.train_config is not None) and (self.eval_config is not None)

    def log(self):
        logging.info('Parameters used: ')
        for config_object in [self, self.train_config, self.eval_config, self.model]:
            for key, value in config_object.asdict().items():
                logging.info('{}: {}'.format(key, value))

    # TODO move to shared utilities
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])


    # don't decorate, this is session creation point
    def run_estimator(self, extra_callbacks=[]):
        """
        Train and evaluate an estimator.
        `self` may well be provided by default_estimator_params.py`

        TODO save every n epochs 
        TODO enable early stopping
        TODO enable use with tf.serving
        TODO enable logging hooks?

        Args:
            self (RunEstimatorConfig): parameters controlling both estimator and train/test procedure

        Returns:
            None
        """

        logging.info('Batch {}, final size {}'.format(self.batch_size, self.final_size))
        logging.info('Train: {}'.format(self.train_config.tfrecord_loc))
        logging.info('Test: {}'.format(self.eval_config.tfrecord_loc))

        train_dataset = input_utils.get_input(config=self.train_config)
        test_dataset = input_utils.get_input(config=self.eval_config)

        checkpoint_loc = os.path.join(self.log_dir, 'in_progress')
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.log_dir, 'tensorboard'),
                histogram_freq=3,
                write_images=False,  # this actually writes the weights, terrible name
                write_graph=False,
                # profile_batch='2,10' 
                profile_batch=0   # i.e. disable profiling
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_loc,
                monitor='val_loss',
                mode='min',
                save_freq='epoch',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=self.patience),
            tf.keras.callbacks.TerminateOnNaN(),
            custom_callbacks.UpdateStepCallback(
                batch_size=self.batch_size
            )
        ] + extra_callbacks

        verbose = 2
        # https://www.tensorflow.org/tensorboard/scalars_and_keras
        fit_summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'manual_summaries'))
        # pylint: disable=not-context-manager
        with fit_summary_writer.as_default(): 
            # pylint: enable=not-context-manager
            # for debugging
            # self.model.run_eagerly = True
            # https://www.tensorflow.org/api_docs/python/tf/keras/Model

            self.model.fit(
                train_dataset,
                validation_data=test_dataset.repeat(2),  # reduce variance from dropout, augs
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=verbose
            )

        logging.info('All epochs completed - finishing gracefully')
        # note that the BEST model is saved as the latest checkpoint, but self.model is the LAST model after training completes
        # to set self.model to the best model, load the latest checkpoint 
        logging.info('Loading and returning (best) model')
        self.model.load_weights(checkpoint_loc)  # inplace

        # to debug
        # y = self.model.predict(test_dataset)
        # print(y)  # final layer output
        # print(self.model.evaluate(test_dataset))

        return self.model


# batch size changed from 256 for now, for my poor laptop
# can override less rarely specified RunEstimatorConfig defaults with **kwargs if you like
def get_run_config(initial_size, final_size, crop_size, weights_loc, log_dir, train_records, eval_records, epochs, schema, batch_size, **kwargs):


    run_config = RunEstimatorConfig(
        initial_size=initial_size,
        crop_size=crop_size,
        final_size=final_size,
        schema=schema,
        epochs=epochs,
        log_dir=log_dir,
        weights_loc=weights_loc,
        batch_size=batch_size
    )

    train_config = get_train_config(train_records, schema.label_cols, run_config.batch_size, run_config.initial_size, run_config.final_size, run_config.channels)

    eval_config = get_eval_config(eval_records, schema.label_cols, run_config.batch_size, run_config.initial_size, run_config.final_size, run_config.channels)

    model = define_model.get_model(schema, run_config.initial_size, run_config.crop_size, run_config.final_size, weights_loc=weights_loc)

    run_config.assemble(train_config, eval_config, model)
    return run_config


def get_train_config(train_records, label_cols, batch_size, initial_size, final_size, channels):
    # tiny func, refactored for easy reuse
    train_config = input_utils.InputConfig(
        name='train',
        tfrecord_loc=train_records,
        label_cols=label_cols,
        # stratify=False,
        shuffle=True,
        drop_remainder=True,
        repeat=False,  # Changed from True for keras, which understands to restart a dataset
        # stratify_probs=None,
        # geometric_augmentation=False,
        # photographic_augmentation=False,
        # zoom=ZOOM,
        # max_shift=MAX_SHIFT,
        # max_shear=MAX_SHEAR,
        # contrast_range=(0.98, 1.02),
        batch_size=batch_size,
        initial_size=initial_size,
        final_size=final_size,
        channels=channels,
        greyscale=True
    )
    return train_config


def get_eval_config(eval_records, label_cols, batch_size, initial_size, final_size, channels):
    # tiny func, refactored for easy reuse
    eval_config = input_utils.InputConfig(
        name='eval',
        tfrecord_loc=eval_records,
        label_cols=label_cols,
        # stratify=False,
        shuffle=False,  # see above
        repeat=False,
        drop_remainder=False,
        # stratify_probs=None,
        # geometric_augmentation=False,
        # photographic_augmentation=False,
        # zoom=(ZOOM), 
        # max_shift=MAX_SHIFT,
        # max_shear=MAX_SHEAR,
        # contrast_range=(0.98, 1.02),
        batch_size=batch_size,
        initial_size=initial_size,
        final_size=final_size,
        channels=channels,
        greyscale=True
    )
    return eval_config
