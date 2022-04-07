import os
import logging
import random
from functools import partial
from typing import Dict

import numpy as np
import tensorflow as tf



def get_tfrecord_dataset(tfrecord_locs, label_cols, batch_size, shuffle, drop_remainder=False):
    """
    Use feature_spec to load data from tfrecord_locs, and optionally shuffle/batch according to args.
    Does NOT apply any preprocessing.

    Minor differences from image_datasets.get_image_dataset:
    - Includes ``shuffle`` option, because shuffling is best done when loading the tfrecords as we can interleave from different records (rather than later, after loading)
    - Labels are expected to already be encoded in the tfrecords (keyed by ``label_cols``), hence no ``labels`` argument
    
    Args:
        tfrecord_locs (list): paths to tfrecords to load.
        label_cols (list): of features encoded in tfrecord e.g. ['smooth_votes', 'featured_votes']. 'id_str' and 'matrix' will be loaded automatically and do not need to be included.
        batch_size (int): batch size
        shuffle (bool): if True, shuffle the dataset
        drop_remainder (bool): if True, drop any galaxies that don't fit exactly into a batch e.g. galaxy 9 of a list of 9 galaxies with batch size 8. Default False.
    

    Returns:
        tf.data.Dataset: yielding batches of {'matrix': , 'id_str': , label_cols[0]: , label_cols[1], ...}, optionally shuffled and cut.
    """
    feature_spec = get_feature_spec(label_cols)


    dataset = load_tfrecords(tfrecord_locs, feature_spec, shuffle=shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # ensure that a batch is always ready to go
    return dataset


def load_tfrecords(tfrecord_locs, feature_spec, num_parallel_calls=tf.data.experimental.AUTOTUNE, shuffle=False):
    """
    Load tfrecords as tf.data.Dataset.
    Used by get_tfrecord_dataset.

    shuffle will randomise the order of the tfrecords. This is different (and complementary) to shuffling the batches (see get_tfrecord_dataset).
    If shuffle=False and multiple tfrecords, returns examples in deterministic but not equal order 

    Args:
        tfrecord_locs (list): paths to tfrecords to load.
        feature_spec (dict): like {feature: tf.io spec}. See source code.
        num_parallel_calls (int, optional): Number of parallel threads to use when loading. Defaults to tf.data.experimental.AUTOTUNE.
        shuffle (bool, optional): If True, shuffle order in which to load TFRecords. Defaults to False.

    Returns:
        tf.data.Dataset: yielding {'matrix': , 'id_str': , label_cols[0]: , label_cols[1], ...} for each galaxy in each TFRecord, optionally shuffled by TFRecord.
    """
    # TODO consider num_parallel_calls = len(list)?
    # small wrapper around loading a TFRecord as a single tensor tuples
    logging.info('tfrecord.io: Loading dataset from {}'.format(tfrecord_locs))
    parse_function = partial(general_parsing_function, features=feature_spec)    
    if isinstance(tfrecord_locs, str):
        logging.warning('Loading single tfrecord {} - is this expected?'.format(tfrecord_locs))
        dataset = tf.data.TFRecordDataset(tfrecord_locs)
        return dataset.map(parse_function, num_parallel_calls=num_parallel_calls)  # Parse the record into tensors
    else:
        # see https://github.com/tensorflow/tensorflow/issues/14857#issuecomment-365439428
        logging.warning('Loading multiple tfrecords with interleaving, shuffle={}'.format(shuffle))
        logging.info('Files to load: {} ({})'.format(tfrecord_locs, len(tfrecord_locs)))
        assert isinstance(tfrecord_locs, list)
        assert len(tfrecord_locs) > 0
        # tensorflow will NOT raise an error if a tfrecord file is missing, if the directory exists!
        assert all([os.path.isfile(loc) for loc in tfrecord_locs])
        num_files = len(tfrecord_locs)

        if shuffle:
            random.shuffle(tfrecord_locs)  # inplace

        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecord_locs, dtype=tf.string))
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename).map(parse_function),
            cycle_length=num_files,  # concurrently processed input elements
            num_parallel_calls=num_parallel_calls,
            deterministic=True
        )
        # could add num_parallel_calls if desired, but let's leave for now 
        # for extra randomness, may shuffle those (1st in s1, 1st in s2, ...) subjects
        return dataset


def get_feature_spec(label_cols):
    """
    Get dataset images and labels from tfrecord according to instructions in config
    Does NOT apply preprocessing e.g. augmentations or further brightness tweaks

    Args:
        config (PreprocessingConfig): instructions to load and preprocess the image and label data

    Returns:
        (tf.Tensor, tf.Tensor)
    """
    requested_features = {'matrix': 'string', 'id_str': 'string'}  # new - must always have id_str
    # We only support float labels!
    # add key-value pairs like (col: float) for each col in config.label_cols
    # the order of config.label_cols will be the order that labels (axis=1) is indexed
    requested_features.update(zip(label_cols, ['float' for col in label_cols]))
    return construct_feature_spec(requested_features)


def general_parsing_function(serialized_example, features):
    """Parse example. Decode feature 'matrix' into float32 if present"""
    example = tf.io.parse_single_example(serialized=serialized_example, features=features)
    if 'matrix' in features.keys():
        example['matrix'] = cast_bytes_of_uint8_to_float32(example['matrix'])
    return example


def construct_feature_spec(expected_features: Dict) -> Dict:
    """For arbitrary feature specs, to generalise active learning to multi-label
    
    Args:
        expected_features_dict ([type]): [description]
    
    Raises:
        ValueError: [description]
    
    Returns:
        [type]: [description]
    """
    features = dict()
    for key, value  in expected_features.items():
        if value == 'string':  # e.g. matrix
            features[key] = tf.io.FixedLenFeature([], tf.string)
        elif value == 'float':  # e.g. a label
            features[key] = tf.io.FixedLenFeature([], tf.float32)
        elif value == 'int':  # e.g. size of image
            features[key] = tf.io.FixedLenFeature([], tf.int64)
        else:
            raise ValueError('Data type {} (for {}) not understood'.format(value, key))
    return features


def cast_bytes_of_uint8_to_float32(some_bytes):
    # bytes are uint of range 0-255 (i.e. pixels)
    # floats are 0-1 by convention (and may be clipped if not)
    # tfrecord datasets will be saved as 0-1 floats and so do NOT need dividing again (see preprocess.py, normalise_from_uint8 should be False)
    return tf.cast(tf.io.decode_raw(some_bytes, out_type=tf.uint8), tf.float32) / 255.
