import copy
from typing import List
import logging

import tensorflow as tf
# from skimage.transform import warp, AffineTransform, SimilarityTransform

class PreprocessingConfig():

    def __init__(
            self,
            label_cols: List,  # use [] if no labels
            input_size: int,
            make_greyscale: bool,
            normalise_from_uint8: bool,
            permute_channels=False,
            input_channels=3,  # for png, jpg etc. Might be different if e.g. fits (not supported yet).
    ):
        """
        Simple data class to define how images should be preprocessed.
        Can then be used to easily pass those parameters around e.g. preprocess.shuffle, preprocess.make_greyscale, etc.

        Args:
            label_cols (List): list of answer strings in fixed order. Useful for loading labels.
            input_size (int): length of image before preprocessing (assumed square) e.g. 300
            make_greyscale (bool): if True, average over channels (last dimension). Incompatible with ``permute_channels``
            normalise_from_uint8 (bool): if True, assume input image is 0-255 range and divide by 255.
            permute_channels (bool, optional): If True, randomly swap channels around. Defaults to False.
            input_channels (int, optional): Number of channels in input image (last dimension). Defaults to 3.

        Raises:
            ValueError: trying to permute channels when ``input_channels == 1``
        """
        self.label_cols = label_cols
        self.input_size = input_size
        self.input_channels = input_channels
        self.normalise_from_uint8 = normalise_from_uint8
        self.make_greyscale = make_greyscale
        self.permute_channels = permute_channels

        if make_greyscale and permute_channels:
            raise ValueError("Incompatible options - can't permute channels if there's only one!")

    # TODO move to shared utilities
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])

    def copy(self):
        return copy.deepcopy(self)


# Wrapping this causes weird op error - leave it be. Issue raised w/ tf.
# @tf.function
def preprocess_dataset(dataset, config):
    """
    Thin wrapper applying ``preprocess_batch`` across dataset. See ``preprocess_batch`` for more.

    Args:
        config (PreprocessingConfig): Configuration object defining how 'get_input' should function

    Returns:
        (dict) of form {'x': make_greyscale image batch}, as Tensor of shape [batch, size, size, 1]}
        (Tensor) categorical labels for each image
    """
    return dataset.map(lambda x: preprocess_batch(x, config))


def preprocess_batch(batch, config):
    """
    Apply preprocessing to batch as directed by ``config``.

    If config.normalise_from_uint8, assume images are 0-255 range and divide by 255.
    Then apply ``preprocess_images``.

    Finally, split batch into tuples of (images, labels) (if ``config.label_cols`` is not empty) or (images, id_strings) otherwise.
    
    Args:
        batch (dict): not quite a dict but a tf.data.Dataset batch, which can be keyed with batch['matrix'], batch['id_str'], and perhaps batch[col] for each col in ``config.label_cols``
        config (PreprocessingConfig): Configuration object defining how 'get_input' should function
    
    Returns:
        tuple: see above
    """
    # logging.info('Loading image size {}'.format(config.input_size))
    batch_images = get_images_from_batch(
        batch,
        size=config.input_size,
        channels=config.input_channels)

    if config.normalise_from_uint8:
        batch_images = batch_images / 255.

    # WARNING the /255 may cause issues if repeated again by accident, maybe move
    # by default, simply makes the images make_greyscale. More augmentations on loading model.
    augmented_images = preprocess_images(batch_images, config.input_size, config.make_greyscale, config.permute_channels)
    # tf.summary.image('c_augmented', augmented_images)

    if len(config.label_cols) == 0:
        logging.warning('No labels requested, returning id_str as labels')
        return augmented_images, batch['id_str']
    else:
        batch_labels = get_labels_from_batch(batch, label_cols=config.label_cols)
        return augmented_images, batch_labels # labels are unchanged


def preprocess_images(batch_images, input_size, make_greyscale, permute_channels):
    """
    Apply basic preprocessing to a batch of images.

    Args:
        batch_images (tf.Tensor): of shape (batch_size, input_size, input_size, channels)
        input_size (int): length of images before preprocessing (assumed square)
        make_greyscale (bool): if True, take an average over channels.
        permute_channels (bool): if True, randomly swap channels around.

    Returns:
        tf.Tensor: preprocessed images, with channels=1 if ``make_greyscale``.
    """
    assert len(batch_images.shape) == 4
    assert batch_images.shape[3] == 3  # should still have 3 channels at this point

    if make_greyscale:
        # new channel dimension of 1
        channel_images = tf.reduce_mean(input_tensor=batch_images, axis=3, keepdims=True)
        assert channel_images.shape[1] == input_size
        assert channel_images.shape[2] == input_size
        assert channel_images.shape[3] == 1
        # tf.summary.image('b_make_greyscale', channel_images)
    else:
        if permute_channels:
            channel_images = tf.map_fn(permute_channels, batch_images)  # map to each image in batch
        else:
            channel_images = tf.identity(batch_images)

    # augmentation is now done through tf.keras.layers.experimental.preprocessing for speed
    augmented_images = tf.identity(channel_images)

    return augmented_images


def get_images_from_batch(batch, size, channels):
    """
    Extract images from batch and ensure they are the expected size.
    Useful to then manipulate those images.

    Args:
        batch (dict): tf.data.Dataset batch with images under 'matrix' key
        size (int): length of images before preprocessing (assumed square)
        channels (int): Number of channels in input image (last dimension).

    Returns:
        tf.Tensor: images of shape ``(batch_size, size, size, channels)``
    """
    batch_data = tf.cast(batch['matrix'], tf.float32)  # may automatically read uint8 into float32, but let's be sure
    # watch out, this may reshape to the wrong size if you specified the wrong --shard-img-size by an integer factor
    batch_images = tf.reshape(
        batch_data,
        [-1, size, size, channels])  # may not get full batch at end of dataset
    assert len(batch_images.shape) == 4
    # tf.summary.image('a_original', batch_images)
    # tf.summary.scalar('batch_size', tf.shape(preprocessed_batch_images['x'])[0])
    return batch_images


def get_labels_from_batch(batch, label_cols: List):
    """
    Extract labels from batch.

    Batch will have labels keyed under batch[col] for col in ``label_cols``.
    Stack those labels into a tf.Tensor that can then be used for e.g. evaluating a model.
    Order of labels in the tf.Tensor will match that of ``label_cols``.

    Args:
        batch (dict): tf.data.Dataset batch
        label_cols (List): strings for each answer e.g. ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk', etc]

    Returns:
        tf.Tensor: labels extracted from batch, of shape (batch_size, num. answers)
    """
    return tf.stack([batch[col] for col in label_cols], axis=1)   # batch[cols] appears not to work


def permute_channels(im):
    assert tf.shape(im)[-1] == 3
    # tf.random.shuffle shuffles 0th dimension, so need to temporarily swap channel to 0th, shuffle, and swap back
    return tf.transpose(tf.random.shuffle(tf.transpose(im, perm=[2, 1, 0])), perm=[2, 1, 0])
