import os
import logging

import pandas as pd
import tensorflow as tf


# https://stackoverflow.com/questions/62544528/tensorflow-decodejpeg-expected-image-jpeg-png-or-gif-got-unknown-format-st?rq=1
def load_image_file(loc, mode='png'):
    """
    Load an image file from disk to memory.
    *Recently changed to return 0-1 floats not 0-255 floats*

    Args:
        loc (str): Path to image on disk. Includes format e.g. .png.
        mode (str, optional): Image format. Defaults to 'png'.

    Raises:
        ValueError: mode is neither png nor jpeg.

    Returns:
        dict: like {'matrix': float32 np.ndarray from 0. to 1., 'id_str': ``loc``}
    """
    # values will be 0-255, does not normalise. Happens in preprocessing instead.
    # specify mode explicitly to avoid graph tracing issues
    image = tf.io.read_file(loc)
    if mode == 'png':
        image = tf.image.decode_png(image)
    # rename jpg to jpeg or validation checks in decode_jpg will fail
    # might fail, let's see
    elif (mode == 'jpg') or (mode == 'jpeg'):  
        # TODO also allow jpg
        image = tf.image.decode_jpeg(image)
    else:
        raise ValueError(f'Image filetype mode {mode} not recognised')

    converted_image = tf.cast(image, tf.float32) / 255.  # floats in 0-1 range

    return {'matrix': converted_image, 'id_str': loc}  # using the file paths as identifiers


def resize_image_batch_with_tf(batch, size):
    # May cause values outside 0-255 margins
    # May be slow. Ideally, resize the images beforehand on disk (or save as TFRecord, see make_shards.py and tfrecord_datasets.py)
    return tf.image.resize(batch, (size, size), method=tf.image.ResizeMethod.LANCZOS3, antialias=True)


def prepare_image_batch(batch, resize_size=None):
    # batch['matrix'] must be 0-1 floats, not 0-255 ints, or clipping will ruin
    images, id_strs = batch['matrix'], batch['id_str']  # unpack from dict
    if resize_size:
        images = resize_image_batch_with_tf(images , size=resize_size)   # initial size = after resize from image on disk (e.g. 424 for GZ pngs) but before crop/zoom
        images = tf.clip_by_value(images, 0., 1.)  # resizing can cause slight change in min/max
    return {'matrix': images, 'id_str': id_strs}  # pack back into dict


def get_image_dataset(image_paths, file_format, requested_img_size, batch_size, labels=None, check_valid_paths=True, shuffle=False, drop_remainder=False):
    """
    Load images in a folder as a tf.data dataset
    Supports jpeg (note the e) and png

    Args:
        image_paths (list): list of image paths to load
        file_format (str): image format e.g. png, jpeg
        requested_img_size (int): e.g. 256 for 256x256x3 image. Assumed square. Will resize if size on disk != this.
        batch_size (int): batch size to use when grouping images into batches
        labels (list or None): If not None, include labels in dataset (see Returns). Must be equal length to image_paths. Defaults to None.

    Raises:
        FileNotFoundError: at least one path does not match an existing file

    Returns:
        tf.data.Dataset: yielding batches with 'matrix' key for image array, 'id_str' for the image path, and 'label' if ``labels`` was provided.
    """
    
    assert len(image_paths) > 0
    assert isinstance(image_paths[0], str)
    logging.info('Image paths to load as dataset: {}'.format(len(image_paths)))

    if check_valid_paths:
        logging.info('Checking if all paths are valid')
        missing_paths = [path for path in image_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(f'Missing {len(missing_paths)} images e.g. {missing_paths[0]}')
        logging.info('All paths exist')
    else:
        logging.warning('Skipping valid path check')

    path_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths])

    image_ds = path_ds.map(lambda x: load_image_file(x, mode=file_format), num_parallel_calls=tf.data.AUTOTUNE)  # keep determinstic though

    image_ds = image_ds.batch(batch_size, drop_remainder=drop_remainder)

    # check if the image shape matches requested_img_size, and resize if not
    test_images = [batch for batch in image_ds.take(1)][0]['matrix']
    size_on_disk = test_images.numpy().shape[1]  # x dimension (BXYC convention)
    if size_on_disk == requested_img_size:
        logging.info('Image size on disk matches requested_img_size of {}, skipping resizing'.format(requested_img_size))  # x dimension of first image, first y index, first channel
    else:
        logging.warning('Resizing images from disk size {} to requested size {}'.format(size_on_disk, requested_img_size))
        image_ds = image_ds.map(lambda x: prepare_image_batch(x, resize_size=requested_img_size))

    # now returns floats from 0 to 1
    # image_batch = list(image_ds.take(1))[0]
    # print(image_batch)
    # image = image_batch['matrix'].numpy()[0]
    # print(image.min(), image.max())
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()
    # exit()

    if labels is not None:
        assert len(labels) == len(image_paths)

        if isinstance(labels[0], dict):
            # assume list of dicts, each representing one datapoint e.g. [{'feat_a': 1, 'feat_b': 2}, {'feat_a': 12, 'feat_b': 42}]
            # reshape to columnlike dict e.g. {'feat_a': [1, 12], 'feat_b: [2, 42]} because that's what tf supports
            # (could pass this directly, but inputs tend to be easier to handle row-wise for e.g. shuffling etc)
            label_ds = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(labels).to_dict(orient="list"))
        else:
            # make it a dict anyway for consistency, keyed by 'label' instead of e.g. 'feat_a', 'feat_b'
            label_ds = tf.data.Dataset.from_tensor_slices({'label': labels})

        # drop_remainder applied to labels as well, if relevant. Equal length = same drop.
        label_ds = label_ds.batch(batch_size, drop_remainder=drop_remainder)

        # print(list(label_ds.take(1)))  

        # label_dict is {'label': (256)} or {'feat_a': (256), 'feat_b': (256)}
        # image_dict is {'id_str': some_id 'matrix': (image)}
        # merge the two dicts to create {'id_str': ..., 'matrix': ..., 'feat_a': ..., 'feat_b': ...}
        image_ds = tf.data.Dataset.zip((image_ds, label_ds)).map(lambda image_dict, label_dict: {**image_dict, **label_dict})
        # now yields {'matrix': , 'id_str': , 'label': } batched dicts

    # shuffle must only happen *after* zipping in the labels
    if shuffle:
        image_ds = image_ds.shuffle(buffer_size=5)  # already batched, so buffer is *batches*
        # TODO could use interleave etc

    image_ds = image_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return image_ds
