import os
from functools import partial
import logging
import random

import numpy as np
import tensorflow as tf


def cast_bytes_of_uint8_to_float32(some_bytes):
    return tf.cast(tf.io.decode_raw(some_bytes, out_type=tf.uint8), tf.float32)


def general_parsing_function(serialized_example, features):
    """Parse example. Decode feature 'matrix' into float32 if present"""
    example = tf.io.parse_single_example(serialized=serialized_example, features=features)
    if 'matrix' in features.keys():
        example['matrix'] = cast_bytes_of_uint8_to_float32(example['matrix'])
    return example


def load_dataset(filenames, feature_spec, num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=False):
    # TODO consider num_parallel_calls = len(list)?
    # small wrapper around loading a TFRecord as a single tensor tuples
    logging.info('tfrecord.io: Loading dataset from {}'.format(filenames))
    parse_function = partial(general_parsing_function, features=feature_spec)    
    if isinstance(filenames, str):
        logging.warning('Loading single tfrecord {} - is this expected?'.format(filenames))
        dataset = tf.data.TFRecordDataset(filenames)
        return dataset.map(parse_function, num_parallel_calls=num_parallel_calls)  # Parse the record into tensors
    else:
        # see https://github.com/tensorflow/tensorflow/issues/14857#issuecomment-365439428
        logging.warning('Loading multiple tfrecords with interleaving, shuffle={}'.format(shuffle))
        logging.info('Files to load: {} ({})'.format(filenames, len(filenames)))
        assert isinstance(filenames, list)
        assert len(filenames) > 0
        # tensorflow will NOT raise an error if a tfrecord file is missing, if the directory exists!
        assert all([os.path.isfile(loc) for loc in filenames])
        num_files = len(filenames)

        if shuffle:
            random.shuffle(filenames)  # inplace

        # simple version for robustness/debugging
        # dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel_calls)  # interleave, hopefully deterministically!
        # dataset = dataset.map(parse_function)

        # get tfrecords matching filenames, optionally shuffling order of shards to be read

        # if shuffle:
        #     dataset = dataset.shuffle(num_files)

        # dataset.flat_map(lambda filename: tf.data.TFRecordDataset(filename).map(parse_function))

        # fancy version, temporarily turned off in case it's messing with training
        # read 1 file per shard, cycling through shards
        # dataset = tf.data.TFRecordDataset(filenames)
        # if shuffle:
        #     dataset = dataset.shuffle(num_files)
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(filenames, dtype=tf.string))
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename).map(parse_function),
            cycle_length=num_files,  # concurrently processed input elements
            num_parallel_calls=num_parallel_calls,
            deterministic=True
        )
            # could add num_parallel_calls if desired, but let's leave for now 
        # for extra randomness, may shuffle those (1st in s1, 1st in s2, ...) subjects
        return dataset
