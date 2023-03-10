.. _training_config:

training_config
===================

This module creates the :class:`Trainer` class for training a Zoobot model (itself a tf.keras.Model).
Implements common features training like early stopping and tensorboard logging.

Follows the same idea as the PyTorch Lightning Trainer object.

.. autoclass:: zoobot.tensorflow.training.training_config.Trainer
    :members:
