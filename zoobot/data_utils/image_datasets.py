import os
import logging

from pathlib import Path

import tensorflow as tf


# https://stackoverflow.com/questions/62544528/tensorflow-decodejpeg-expected-image-jpeg-png-or-gif-got-unknown-format-st?rq=1
def load_image_file(loc, mode='png'):
    # specify mode explicitly to avoid graph tracing issues
    image = tf.io.read_file(loc)
    if mode == 'png':
        image = tf.image.decode_png(image)
    elif mode == 'jpeg':  # rename jpg to jpeg or validation checks in decode_jpg will fail
        image = tf.image.decode_jpeg(image)
    else:
        raise ValueError(f'Image filetype mode {mode} not recognised')
    converted_image = tf.image.convert_image_dtype(image, tf.float32)

    # if loc_to_label is not None:
    #     assert callable(loc_to_label)
    #     label = loc_to_label(loc)
    #     return converted_image, label
    # else:
    return {'matrix': converted_image, 'id_str': loc}  # using the file paths as identifiers


def resize_image_batch_with_tf(batch, size):
    return tf.image.resize(batch, (size, size), method=tf.image.ResizeMethod.LANCZOS3, antialias=True)


def prepare_image_batch(batch, initial_size):
    images, id_strs = batch['matrix'], batch['id_str']  # unpack from dict
    images = resize_image_batch_with_tf(images , size=initial_size)   # initial size = after resize from 424 but before crop/zoom
    # images = tf.reduce_mean(input_tensor=images, axis=3, keepdims=True)  # greyscale NOPE, do in preprocess preprocessing
    return {'matrix': images, 'id_str': id_strs}  # pack back into dict


def get_image_dataset(image_paths, file_format, initial_size, batch_size, labels=None):
    """
    Load images in a folder as a tf.data dataset
    Supports jpeg (note the e) and png
    For labels, encode them in the loc (e.g. images/spirals/gal_1345.png, or images/gal_1345_spiral.png) and provide loc_to_label function to decode

    Args:
        folder_to_predict ([type]): [description]
        file_format ([type]): [description]
        initial_size ([type]): [description]
        batch_size ([type]): [description]
        loc_to_label ([type], optional): [description]. Defaults to None.

    Raises:
        FileNotFoundError: [description]

    Returns:
        [type]: [description]
    """
    
    assert len(image_paths) > 0
    assert isinstance(image_paths[0], str)
    logging.info('Images to predict on: {}'.format(len(image_paths)))

    # check they exist
    missing_paths = [path for path in image_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(f'Missing {len(missing_paths)} images e.g. {missing_paths[0]}')

    path_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths])


    image_ds = path_ds.map(lambda x: load_image_file(x, mode=file_format))
    image_ds = image_ds.batch(batch_size, drop_remainder=False)
    image_ds = image_ds.map(lambda x: prepare_image_batch(x, initial_size=initial_size))

    if labels is not None:
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        label_ds = label_ds.batch(batch_size, drop_remainder=False)
        image_ds = tf.data.Dataset.zip((image_ds, label_ds)).map(lambda image_dict, label: dict(image_dict, label=label))  # now yields (image, label) tuples

    # for image, label in image_ds.take(5):
    #     print(image.shape, label.shape)
    # exit()

    image_ds = image_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_ds
