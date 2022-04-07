define_model
===================

This module contains functions for defining an EfficientNet model (:meth:`zoobot.estimators.define_model.get_model`),
with or without the GZ DECaLS head, and optionally to load the weights of a pretrained model.

Models are defined using functions in ``efficientnet_standard`` and ``efficientnet_custom``.

.. autofunction:: zoobot.tensorflow.estimators.define_model.get_model

|

.. autofunction:: zoobot.tensorflow.estimators.define_model.add_augmentation_layers

|

.. autofunction:: zoobot.tensorflow.estimators.define_model.load_weights

|

.. autofunction:: zoobot.tensorflow.estimators.define_model.load_model
