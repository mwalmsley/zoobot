import logging
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
# from astropy.io import fits
from tqdm import tqdm

from zoobot.data_utils import create_tfrecord

import matplotlib
# 
import matplotlib.pyplot as plt


def write_catalog_to_train_test_tfrecords(df, train_loc, test_loc, img_size, columns_to_save, reader, train_test_fraction=0.8):
    """Save a galaxy catalog as a pair of train and test TFRecords, for fast loading later.
    Makes the train/test split and returns the corresponding train/test catalogs for convenience.
    
    Args:
        df (pandas.Dataframe): galaxy catalog, including columns_to_save columns
        train_loc (str): path to save training tfrecord
        test_loc (str): path to save testing tfrecord
        img_size (int): image edge length e.g. 256 for 256x256 image. Assumed square.
        columns_to_save (list): columns to save as features in tfrecord. e.g. ['id_str', smooth_votes', 'featured_votes'].
        reader (function): expecting galaxy row (dictlike), returning loaded PIL image for that galaxy
        train_test_fraction (float, optional): Defaults to 0.8.
    
    Returns:
        pd.Dataframe: train portion of original catalog (``df``)
        pd.Dataframe: test portion of original catalog (``df``)
    """
    train_df, test_df = split_df(df, train_test_fraction)
    train_df.to_csv(train_loc + '.csv')  # ugly but effective
    test_df.to_csv(test_loc + '.csv')


    write_image_df_to_tfrecord(train_df, train_loc, img_size, columns_to_save, append=False, reader=reader)
    write_image_df_to_tfrecord(test_df, test_loc, img_size, columns_to_save, append=False, reader=reader)
    return train_df, test_df


def write_image_df_to_tfrecord(df, tfrecord_loc, img_size, columns_to_save, reader):
    """Write a galaxy catalog to TFRecord file.
    For example, the training catalog to the training TFRecord, to then be trained on by a model.

    Args:
        df (pandas.Dataframe): galaxy catalog, including columns_to_save columns
        tfrecord_loc (str): path to save tfrecord
        img_size (int): image edge length e.g. 256 for 256x256 image. Assumed square.
        columns_to_save (list): columns to save as features in tfrecord. e.g. ['id_str', smooth_votes', 'featured_votes'].
        reader (function): expecting galaxy row (dictlike), returning loaded PIL image for that galaxy
    """

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
    Convert a galaxy catalog to serialized binary format, as part of saving to TFRecord

    Row should have columns that exactly match a feature spec function
    Serialised example will have columns ['matrix] + columns_to_save
    e.g. ['matrix', 'smooth-or-featured_smooth', 'smooth-or-featured_featured', 'smooth-or-featured_total']
    
    Args:
        row (pd.Series): galaxy row to save. Must include columns_to_save keys.
        img_size (int): image edge length e.g. 256 for 256x256 image. Assumed square.
        columns_to_save (list): columns to save as features in tfrecord. e.g. ['id_str', smooth_votes', 'featured_votes'].
        reader (function): expecting galaxy row (dictlike), returning loaded PIL image for that galaxy
    
    Returns:
        str: binary serialized representation of the galaxy row, including ``columns_to_save`` features.
    """
    pil_img = reader(row)
    final_pil_img = pil_img.resize(size=(img_size, img_size), resample=Image.LANCZOS).transpose(
        Image.FLIP_TOP_BOTTOM)
    matrix = np.array(final_pil_img)

    extra_data_dict = {}
    for col in columns_to_save:
        extra_data_dict.update({col: row[col]})

    return create_tfrecord.serialize_image_example(matrix, **extra_data_dict)
