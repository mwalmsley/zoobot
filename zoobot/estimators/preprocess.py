import copy
from typing import List
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.transform import warp, AffineTransform, SimilarityTransform

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


# def load_batches_without_labels(config):
#     # does not fetch id - unclear if this is important
#     feature_spec = get_feature_spec({'matrix': 'string'})
#     batch = get_dataset(config.tfrecord_loc, feature_spec, config.batch_size, config.shuffle, config.repeat, config.drop_remainder)
#     return get_images_from_batch(batch, config.input_size, config.input_channels, summary=True)


# def load_batches_with_id_str(config):
#     # does not fetch id - unclear if this is important
#     feature_spec = get_feature_spec({'matrix': 'string', 'id_str': 'string'})
#     batch = get_dataset(config.tfrecord_loc, feature_spec, config.batch_size, config.shuffle, config.repeat, config.drop_remainder)
#     return get_images_from_batch(batch, config.input_size, config.input_channels, summary=True), batch['id_str']



# def augment_images(images, input_config):
#     """

#     Args:
#         images (tf.Variable):
#         input_config (PreprocessingConfig):

#     Returns:

#     """
#     if input_config.geometric_augmentation:
#         images = geometric_augmentation(
#             images,
#             max_shift=input_config.max_shift,
#             max_shear=input_config.max_shear,
#             zoom=input_config.zoom,
#             output_size=input_config.output_size)

#     if input_config.photographic_augmentation:
#         images = photographic_augmentation(
#             images,
#             max_brightness_delta=input_config.max_brightness_delta,
#             contrast_range=input_config.contrast_range)

#     if input_config.permute_channels:
#         assert not input_config.make_greyscale
#         # assert tf.shape(images)[-1] > 1
#         images = tf.map_fn(permute_channels, images)  # map to each image in batch

#     return images


def permute_channels(im):
    assert tf.shape(im)[-1] == 3
    # tf.random.shuffle shuffles 0th dimension, so need to temporarily swap channel to 0th, shuffle, and swap back
    return tf.transpose(tf.random.shuffle(tf.transpose(im, perm=[2, 1, 0])), perm=[2, 1, 0])


# def geometric_augmentation(images, max_shift, max_shear, zoom, output_size):
#     """
#     Runs best if image is originally significantly larger than final target size
#     for example: load at 256px, rotate/flip, crop to 246px, then finally resize to 64px
#     This leads to more computation, but more pixel info is preserved

#     # TODO add stretch and/or shear?
#     # TODO add cutout http://arxiv.org/abs/1708.04552 ?

#     Args:
#         images ():
#         zoom (tuple): of form {min zoom in decimals e,g, 1.0, max zoom in decimals e.g, 1.2}
#         output_size (): resize to this after crop

#     Returns:
#         (Tensor): image rotated, flipped, cropped and (perhaps) normalized, shape (target_size, target_size, channels)
#     """

#     images = ensure_images_have_batch_dimension(images)

#     assert images.shape[1] == images.shape[2]  # must be square
#     assert len(zoom) == 2
#     assert zoom[0] <= zoom[1] 
#     # assert zoom[1] > 1. and zoom[1] < 10.  # catch user accidentally putting in pixel values here

#     # flip functions support batch dimension, but it must be precisely fixed
#     # let's take the performance hit for now and use map_fn to allow variable length batches
#     images = tf.map_fn(tf.image.random_flip_left_right, images)
#     images = tf.map_fn(tf.image.random_flip_up_down, images)


#     # images = tf.map_fn(random_rotation_batch, images)  No, tfa causes segfault on many CPU...
#     images = tf.map_fn(lambda x: wrapped_augmentation(x, max_shift, max_shear, zoom, patches=4, min_size=2, max_size=5), images)



#     # resize to final desired size (may match crop size)
#     images = tf.map_fn(
#         lambda x: tf.image.resize(
#             x,
#             tf.constant([output_size, output_size], dtype=tf.int32),
#             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR  # only nearest neighbour works - otherwise gives noise
#         ),
#         images
#     )

#     return images


# def wrapped_augmentation(im, max_shift, max_shear, zoom, patches, min_size, max_size):
#     im_shape = im.shape
#     im = tf.py_function(lambda x: keras_np_augmentation(x, max_shift=max_shift, max_shear=max_shear, zoom=zoom), [im], tf.float32)
#     # im = tf.py_function(lambda x: py_augmentation(x, max_shift=max_shift, max_shear=max_shear, patches=patches, min_size=min_size, max_size=max_size), [im], tf.float32)
#     im.set_shape(im_shape)
#     return im


# def py_augmentation(im, max_shift, max_shear, patches, min_size, max_size):
#     im_np = im.numpy()
#     im_np = np_affine_augmentation(im_np, max_shift, max_shear)
#     # im_np = np_multiple_cutout(im_np, patches, min_size, max_size)
#     return im_np


# def np_affine_augmentation(im_np, max_shift, max_shear):
#     # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.AffineTransform

#     # https://stackoverflow.com/questions/25895587/python-skimage-transform-affinetransform-rotation-center
#     shift_y, shift_x = np.array(im_np.shape[:2]) / 2.
#     tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
#     tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])

#     tf_rotate = AffineTransform(
#         # scale=(1.3, 1.1),
#         rotation=np.random.uniform(low=0., high=np.pi/2.),  # 90 deg, enough when you consider flips
#         shear=np.random.uniform(low=0., high=max_shear),
#         translation=np.random.uniform(low=0, high=max_shift, size=2)
#     )
#     # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
#     return warp(im_np, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, output_shape=im_np.shape)


# def np_multiple_cutout(im_np, patches, min_size, max_size):
#     for _ in range(patches):
#         im_np = cutout_patch(im_np, sizes=np.random.randint(size=[2], low=min_size, high=max_size))
#     return im_np


# def cutout_patch(im_np, sizes):
#     x_size, y_size = sizes[0], sizes[1]
#     x0 = np.random.randint(low=0, high=im_np.shape[0]-x_size)  # will execute eagerly, should be okay to actually randomise
#     x1 = x0 + x_size
#     y0 = np.random.randint(low=0, high=im_np.shape[1]-y_size)
#     y1 = y0 + y_size
#     im_np[x0:x1, y0:y1] = 0.
#     return im_np


# # will be replaced in TF 2.2
# def keras_np_augmentation(im, max_shift, max_shear, zoom):
    
    
#     # np random is okay here in python/eager context, don't overoptimise - tf 2.2. should solve for me

#     rotated = tf.keras.preprocessing.image.random_rotation(
#         im.numpy(), 
#         rg=90.,
#         row_axis=0,
#         col_axis=1,
#         channel_axis=2,
#         fill_mode='reflect',  # might consider reflecting or wrapping?
#         # cval=0.0
#     )
    
#     # return rotated
#     return tf.keras.preprocessing.image.apply_affine_transform(
#         rotated, 
#         # theta=np.random.uniform(low=0, high=90),  # enough, when you also consider flips
#         tx=np.random.randint(low=0, high=max_shift+1),  # +1 as high is exclusive
#         ty=np.random.randint(low=0, high=max_shift+1),  # similarly
#         shear=np.random.uniform(low=0., high=max_shear),
#         # can have different zoom in each axis, a bit like shear
#         zx=np.random.uniform(low=zoom[0], high=zoom[1]),
#         zy=np.random.uniform(low=zoom[0], high=zoom[1]),
#         row_axis=0,
#         col_axis=1,
#         channel_axis=2,
#         fill_mode='reflect'
#     )



# def random_rotation_batch(images):
#     return tfa.image.rotate(
#         images,
#         tf.random.uniform(shape=[1]),
#         interpolation='BILINEAR'
#     )

# def random_rotation_py(im):
#     # see https://www.tensorflow.org/guide/data#applying_arbitrary_python_logic
#     im_shape = im.shape
#     im = tf.py_function(np_random_rotation, [im], tf.float32)
#     im.set_shape(im_shape)
#     return im


# def np_random_rotation(im):
#     # tracing may be a problem
#     return ndimage.rotate(im, np.random.uniform(-180, 180), reshape=False)


# def crop_random_size(im, zoom, central):
#     original_width = tf.cast(im.shape[1], tf.float32) # cast allows division of Dimension
#     new_width = tf.squeeze(tf.cast(original_width / tf.random.uniform(shape=[1], minval=zoom[0], maxval=zoom[1]), dtype=tf.int32))
#     # updated from np to avoid fixed value from tracing. Not yet tested!
#     if central:
#         lost_width = int((original_width - new_width) / 2)
#         cropped_im = im[lost_width:original_width-lost_width, lost_width:original_width-lost_width]
#     else:
#         n_channels = tf.constant(im.shape[2], dtype=tf.int32)
#         cropped_shape = tf.stack([new_width, new_width, n_channels], axis=0)
#         cropped_im = tf.image.random_crop(im, cropped_shape)

#     return cropped_im


# def photographic_augmentation(images, max_brightness_delta, contrast_range):
#     """
#     TODO do before or after geometric?
#     TODO add slight redshifting?

#     Args:
#         images ():
#         max_brightness_delta ():
#         contrast_range ():

#     Returns:

#     """
#     images = ensure_images_have_batch_dimension(images)

#     images = tf.map_fn(
#         lambda im: tf.image.random_brightness(im, max_delta=max_brightness_delta),
#         images)
#     images = tf.map_fn(
#         lambda im: tf.image.random_contrast(im, lower=contrast_range[0], upper=contrast_range[1]),
#         images)

#     # experimental
#     # images = tf.map_fn(
#     #     lambda im: im + tf.random.normal(tf.shape(im), mean=0., stddev=.01)  # image values are 0->1 
#     # )

#     return images


# def ensure_images_have_batch_dimension(images):
#     if len(images.shape) < 3:
#         raise ValueError
#     if len(images.shape) == 3:
#         images = tf.expand_dims(images, axis=0)  # add a batch dimension
#     return images


# def predict_input_func(tfrecord_loc, n_galaxies, input_size, mode='labels', label_cols=None):
#     """Wrapper to mimic the run_estimator.py input procedure.
#     Get subjects and labels from tfrecord, just like during training
#     Subjects must fit in memory, as they are loaded as a single batch
#     Args:
#         tfrecord_loc (str): tfrecord to read subjects from. Should be test data.
#         n_galaxies (int, optional): Defaults to 128. Num of galaxies to predict on, as single batch.

#     Returns:
#         subjects: np.array of shape (batch, x, y, channel)
#         labels: np.array of shape (batch)
#     """
#     raise NotImplementedError('Deprecated, check to see how/if this is useful')
#     config = PreprocessingConfig(
#         name='predict',
#         tfrecord_loc=tfrecord_loc,
#         label_cols=label_cols,
#         stratify=False,
#         shuffle=False,  # important - preserve the order
#         repeat=False,
#         regression=True,
#         geometric_augmentation=None,
#         photographic_augmentation=None,
#         zoom=None,
#         fill_mode=None,
#         batch_size=n_galaxies,
#         input_size=input_size,
#         output_size=None,
#         channels=3
#     )
#     # dataset = get_dataset(tfrecord_loc, feature_spec, batch_size, shuffle, repeat)
#     if mode == 'labels':
#         assert label_cols is not None
#         # batch_images = batch['matrix'] for batch for batch in dataset
#         id_strs = None
#     elif mode == 'id_str':
#         batch_images, id_strs = load_batches_with_id_str(config)
#         batch_labels = None
#     elif mode == 'matrix':
#         batch_images = load_batches_without_labels(config)
#         batch_labels = None
#         id_strs = None
#     else:
#         raise ValueError('Predict input func. mode not recognised: {}'.format(mode))

#     # don't do this! preprocessing is done at predict time, expects raw-ish images
#     # preprocessed_batch_images = preprocess_batch(batch_images, config)['x']
#     return batch_images, batch_labels, id_strs
