
from tqdm import tqdm
import numpy as np
import tensorflow as tf


def serialize_image_example(matrix, **extra_kwargs):
    """
    Save an image, label and any additional data to serialized byte string

    Args:
        matrix (np.array): pixel data. Floats in shape [height, width, depth]
        **extra_kwargs (dict): any further keyword args will be saved as named in the tfrecord

    Returns:
        None
    """
    matrix_feature = uint8_array_to_feature(matrix)

    # Expects TensorFlow data format convention, "Height-Width-Depth".
    if matrix.shape[1] > matrix.shape[0]:
        raise Exception('Fatal error: image not in height-width-depth convention')

    height_feature = int_to_feature(matrix.shape[0])
    width_feature = int_to_feature(matrix.shape[1])
    if len(matrix.shape) == 2:
        channels_feature = int_to_feature(1)
    else:
        channels_feature = int_to_feature(matrix.shape[2])

    features_to_save = {
        'matrix': matrix_feature,
        'channels': channels_feature,
        'height': height_feature,
        'width': width_feature
    }

    extra_data = {}
    for name, value in extra_kwargs.items():
        extra_data[name] = value_to_feature(value)
    features_to_save.update(extra_data)

    # construct the Example protocol buffer boject
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature=features_to_save))
    # use the proto object to serialize the example to a string
    return example.SerializeToString()


def value_to_feature(value):
    """
    Helper function to convert value of unknown type into proto feature

    Args:
        value (): value to be converted

    Returns:
        (tf.train.Feature) encoding of value, according to value type.
    """
    if type(value) == int or type(value) == np.int64:
        return int_to_feature(value)
    if type(value) == str:
        return str_to_feature(value)
    elif type(value) == float or type(value) == np.float64:
        return float_to_feature(value)
    elif type(value) == list or type(value) == np.ndarray:
        return float_list_to_feature(value)
    else:
        raise Exception('Fatal error: {} feature of type {} not understood'.format(value, type(value)))


def str_to_feature(str_to_save):
    bytes_to_save = bytes(str_to_save, encoding='utf-8')
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[bytes_to_save])
    )


def int_to_feature(int_to_save):
    return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int_to_save]))


def float_to_feature(float_to_save):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[float_to_save]))


def float_list_to_feature(floats_to_save):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=floats_to_save))
        # could be done with tf.train.BytesList, might save a lot of storage space
        # I think: will be much smaller for pngs (uint8), but exactly the same for fits
        # could make loading much faster, since am loading the gz thumbnails anyway
        # see https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af


def uint8_array_to_feature(matrix_to_save):
    """Useful for images. Feature 'matrix' is later decoded into float32 by read_tfrecord"""
    flat_matrix = np.reshape(matrix_to_save, matrix_to_save.size)
    bytes_to_save = flat_matrix.astype(np.uint8).tobytes()
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[bytes_to_save])
    )
