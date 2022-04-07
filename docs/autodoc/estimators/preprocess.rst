preprocess
===================

This module contains utilities to apply static (i.e. not randomised, not changing each time) preprocessing to a ``tf.data.Dataset`` of images.
This is usually the step after loading the raw images (see :ref:`image_datasets`) and before training the model (see :ref:`training_config`).

.. autofunction:: zoobot.tensorflow.estimators.preprocess.PreprocessingConfig

|

.. autofunction:: zoobot.tensorflow.estimators.preprocess.preprocess_dataset

|

.. autofunction:: zoobot.tensorflow.estimators.preprocess.preprocess_batch

|

.. autofunction:: zoobot.tensorflow.estimators.preprocess.preprocess_images

|

.. autofunction:: zoobot.tensorflow.estimators.preprocess.get_images_from_batch

|

.. autofunction:: zoobot.tensorflow.estimators.preprocess.get_labels_from_batch
