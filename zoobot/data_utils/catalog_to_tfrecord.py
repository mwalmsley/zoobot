import logging
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from astropy.io import fits
from tqdm import tqdm

from zoobot.tfrecord import create_tfrecord, image_utils

import matplotlib
# 
import matplotlib.pyplot as plt


def get_train_test_fraction(total_size, eval_size):
    assert eval_size < total_size
    # in memory for now, but will be serialized for later/logs
    train_test_fraction = (total_size - int(eval_size))/total_size  # always eval on random 2500 galaxies
    logging.info('Train test fraction: {}'.format(train_test_fraction))
    return train_test_fraction


# TODO refactor to make sure this aligns with downloader
def load_decals_as_pil(subject):
    try:
        loc = subject['fits_loc']
    except KeyError:
        loc = subject['file_loc']
        assert loc[-5:] == '.fits'
    img = fits.getdata(loc)

    _scales = dict(
        g=(2, 0.008),
        r=(1, 0.014),
        z=(0, 0.019))

    _mnmx = (-0.5, 300)

    rgb_img = image_utils.dr2_style_rgb(
        (img[0, :, :], img[1, :, :], img[2, :, :]),
        'grz',
        mnmx=_mnmx,
        arcsinh=1.,
        scales=_scales,
        desaturate=True)

    # plt.imshow(rgb_img)
    # plt.savefig('zoobot/test_examples/rescaled_before_pil.png')
    pil_safe_img = np.uint8(rgb_img * 255)
    assert pil_safe_img.min() >= 0. and pil_safe_img.max() <= 255
    return Image.fromarray(pil_safe_img, mode='RGB')


def load_png_as_pil(subject):
    try:
        loc = subject['png_loc']
    except KeyError:
        loc = subject['file_loc']
        assert loc[-4:] == '.png'
    return Image.open(loc)


def get_reader(paths):
    # find file format
    file_format = paths[0].split('.')[-1]
    # check for consistency
    assert all([loc.split('.')[-1] == file_format for loc in paths])
    # check that file paths resolve correctly
    if not all(os.path.isfile(loc) for loc in paths):
        raise FileNotFoundError('Check file paths: currently prefixed like {}'.format(paths[0]))
    if file_format == 'png':
        reader = load_png_as_pil
    elif file_format == 'fits':
        reader = load_decals_as_pil
    return reader


def split_df(df, train_test_fraction):
    # TODO needs test
    train_test_split = int(train_test_fraction * len(df))
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df[:train_test_split].copy()
    test_df = df[train_test_split:].copy()
    assert not train_df.empty
    assert not test_df.empty
    return train_df, test_df


def write_catalog_to_train_test_tfrecords(df, train_loc, test_loc, img_size, columns_to_save, reader, train_test_fraction=0.8):
    """[summary]
    
    Args:
        df ([type]): [description]
        train_loc ([type]): [description]
        test_loc ([type]): [description]
        img_size ([type]): [description]
        columns_to_save ([type]): [description]
        reader (function): expecting subject dictlike (i.e. row), returning PIL image
        train_test_fraction (float, optional): Defaults to 0.8. [description]
    
    Returns:
        [type]: [description]
    """
    train_df, test_df = split_df(df, train_test_fraction)
    train_df.to_csv(train_loc + '.csv')  # ugly but effective
    test_df.to_csv(test_loc + '.csv')


    write_image_df_to_tfrecord(train_df, train_loc, img_size, columns_to_save, append=False, reader=reader)
    write_image_df_to_tfrecord(test_df, test_loc, img_size, columns_to_save, append=False, reader=reader)
    return train_df, test_df


def write_image_df_to_tfrecord(df, tfrecord_loc, img_size, columns_to_save, reader, append=False):
    # tfrecord does not support appending :'(
    if append:
        raise NotImplementedError('tfrecord does not support appending')
    else:
        if os.path.exists(tfrecord_loc):
            logging.warning('{} already exists - deleting'.format(tfrecord_loc))
            os.remove(tfrecord_loc)

    writer = tf.io.TFRecordWriter(tfrecord_loc)
    # for _, subject in tqdm(df.iterrows(), total=len(df), unit=' subjects saved'):
    for _, subject in df.iterrows():
        serialized_example = row_to_serialized_example(subject, img_size, columns_to_save, reader)
        writer.write(serialized_example)
    writer.close()  # good to be explicit - will give 'DataLoss' error if writer not closed


def row_to_serialized_example(row, img_size, columns_to_save, reader):
    """
    Row should have columns that exactly match a read_tfrecord feature spec function
    Serialised example will have columns ['matrix] + columns_to_save
    e.g. ['matrix', 'smooth-or-featured_smooth', 'smooth-or-featured_featured', 'smooth-or-featured_total']
    
    Args:
        row ([type]): [description]
        img_size ([type]): [description]
        columns_to_save ([type]): [description]
        reader ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    #

    matrix, extra_data_dict = row_to_serializable_data(reader, row, img_size, columns_to_save)

    return create_tfrecord.serialize_image_example(matrix, **extra_data_dict)


def row_to_serializable_data(reader, row, img_size, columns_to_save):
    pil_img = reader(row)
    # pil_img.save('zoobot/test_examples/rescaled_after_pil.png')
    # to align with north/east 
    # TODO refactor this to make sure it matches downloader
    final_pil_img = pil_img.resize(size=(img_size, img_size), resample=Image.LANCZOS).transpose(
        Image.FLIP_TOP_BOTTOM)
    matrix = np.array(final_pil_img)

    extra_data_dict = {}
    for col in columns_to_save:
        extra_data_dict.update({col: row[col]})
    return matrix, extra_data_dict
