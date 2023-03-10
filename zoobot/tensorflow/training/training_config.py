import logging
import os
import time

import tensorflow as tf

# from zoobot.tensorflow.estimators import custom_callbacks

# similar style to PyTorch Lightning
class Trainer():
    """Lighting-esque object for executing training.

    Args:
        epochs (int, optional): Max epochs to train. Defaults to 1500.
        patience (int, optional): Max epochs to wait for loss improvement before cancelling training. Defaults to 10.
        log_dir (str, optional): Directory to save training logs. Defaults to 'runs/default_run_{}'.format(time.time()).
        save_freq (str, optional): Frequency with which to save training logs. Defaults to 'epoch'.
    """
    
    def __init__(
        # this doesn't really need to be a class, 
        # but it's kinda nice to break up the training instructions from the model/dataset
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


    def fit(self, model: tf.keras.Model, train_dataset, val_dataset, test_dataset=None, extra_callbacks=[], eager=False, verbose=2):
        """
        Train and evaluate a model.

        Includes tensorboard logging (to log_dir/tensorboard).
        Includes checkpointing (named log_dir/checkpoint), with the rolling best val loss checkpoint saved.
        Includes early stopping according to train_config.patience.

        Args:
            model (tf.keras.Model): model to train. Must already be compiled with model.compile(loss, optimizer)
            train_dataset (tf.data.Dataset): yielding batched tuples of (galaxy images, labels)
            val_dataset (tf.data.Dataset): yielding batched tuples of (galaxy images, labels)
            extra_callbacks (list): any extra callbacks to use when training the model. See e.g. tf.keras.callbacks.
            eager (bool, optional): If True, train in eager mode - slow, but helpful for debugging. Defaults to False.
            verbose (int, optional): 1 for progress bar, useful for local training. 2 for one line per epoch, useful for scripts. Defaults to 2.

        Returns:
            None
        """

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # will create a multi-file checkpoint like {checkpoint.index, checkpoint.data.00000-00001, ...}
        checkpoint_name = os.path.join(self.log_dir, 'checkpoint')

        tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_dir,
                # explicitly disable various slow logging options - enable these if you like
                # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
                histogram_freq=0,  # don't log all the internal histograms, possibly slow
                write_images=False,  # this actually writes the weights, terrible name
                write_graph=False,
                # profile_batch='2,10',
                profile_batch=0,   # i.e. disable profiling
                update_freq=100  # must be an integer, else logging within define_model.py will fail
            )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_name,
                monitor='val_loss',
                mode='min',
                save_freq=self.save_freq,
                save_best_only=True,
                save_weights_only=True)
            
        callbacks = [
            tensorboard_callback,
            checkpoint_callback,
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=self.patience),
            tf.keras.callbacks.TerminateOnNaN()
            # custom_callbacks.VisualizeImages()
        ] + extra_callbacks
        # TODO https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau

        # https://www.tensorflow.org/tensorboard/scalars_and_keras
        # automatically logs (train/validation)/epoch_loss
        # adds into tensorboard_dir, which also has train/val subfolders
        fit_summary_writer = tf.summary.create_file_writer(os.path.join(tensorboard_dir, 'writer'))
        # pylint: disable=not-context-manager
        with fit_summary_writer.as_default(): 
            # pylint: enable=not-context-manager
            # for debugging
            if eager:
                logging.warning('Running in eager mode')
                model.run_eagerly = True
            # https://www.tensorflow.org/api_docs/python/tf/keras/Model

            # import numpy as np
            # tf.summary.image(name='example', data=np.random.rand(16, 64, 64, 1), step=0)

            # print(callbacks)
            # exit()

            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=verbose
                # batch_size not needed when providing tf.data.Datasets
            )


        logging.info('All epochs completed - finishing gracefully')
        # note that the BEST model is saved as the latest checkpoint, but self.model is the LAST model after training completes
        # to set self.model to the best model, load the latest checkpoint 
        logging.info('Loading and returning (best) model')
        model.load_weights(checkpoint_name)  # inplace

        if test_dataset is not None:
            logging.info('Evaluating on test dataset - be careful not to overfit your choices')
            metrics = model.evaluate(
                test_dataset,
                callbacks=[tensorboard_callback],
                verbose=verbose
            )
            logging.info(metrics)
            logging.info(model.metrics_names)
            import wandb
            # wandb.log(dict(zip(model.metrics_names, metrics)))
            wandb.log({'test/epoch_loss': metrics})
        else:
            logging.info('Skipping test evaluation')

        return model
