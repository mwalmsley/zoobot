.. _catalog_to_tfrecord:

catalog_to_tfrecord
===================

This module contains utilities to write galaxy catalogs (as pandas dataframes) into TFRecord files.
These TFRecord files can be read from disk very quickly, which is useful for training ML models.

.. autofunction:: zoobot.tensorflow.data_utils.catalog_to_tfrecord.write_image_df_to_tfrecord

|

.. autofunction:: zoobot.tensorflow.data_utils.catalog_to_tfrecord.row_to_serialized_example
