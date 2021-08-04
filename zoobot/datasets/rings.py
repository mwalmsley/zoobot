import logging
import json
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from zoobot.data_utils import image_datasets


def get_advanced_ring_image_dataset(batch_size, requested_img_size, train_dataset_size=None, seed=1):
    file_format = 'png'
    # TODO point to example catalog instead
    # uses automatic morphology predictions from gz decals cnn
    # ring_catalog = pd.read_csv('data/ring_catalog_with_morph.csv', dtype={'ring': int})
    ring_catalog = pd.read_parquet('data/rare_features_dr5_with_ml_morph.parquet')

    # some personal path fiddling TODO point to example data instead
    # ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].str.replace('/media/walml/beta1', '/Volumes/beta')  # local
    ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].str.replace('/media/walml/beta1/decals/png_native/dr5', '/share/nas/walml/galaxy_zoo/decals/dr5/png')  # galahad

    # apply selection cuts

    # cuts for 200-ish non-spiral rings: good but small selection
    # feat = ring_catalog['smooth-or-featured_featured-or-disk_fraction'] > 0.6
    # face = ring_catalog['disk-edge-on_no_fraction'] > 0.75
    # not_spiral = ring_catalog['has-spiral-arms_no_fraction'] > 0.5
    # passes_cuts = feat & face & not_spiral

    # cuts for 1000-ish rings that are not edge on and not super smooth (mostly wrong tags)
    not_very_smooth = ring_catalog['smooth-or-featured_featured-or-disk_fraction'] > 0.25
    face = ring_catalog['disk-edge-on_no_fraction'] > 0.75
    passes_cuts = not_very_smooth & face

    ring_catalog = ring_catalog[passes_cuts].reset_index(drop=True)
    # logging.info('Labels after selection cuts: \n{}'.format(pd.value_counts(ring_catalog['ring'])))

    # ring_catalog['label'] = ring_catalog['ring'] == 1

    ring_catalog['label'] = get_rough_class_from_ring_fraction(ring_catalog['rare-features_ring_fraction'])

    rings = ring_catalog.query('label == 1')
    not_rings = ring_catalog.query('label == 0')
    logging.info('Rings: {}, Not Rings: {}'.format(len(rings), len(not_rings)))

    resampling_fac = int((len(not_rings)) / (len(rings)))
    logging.info('Resampling fraction: {}'.format(resampling_fac))

    train_split_index = int(len(rings) * 0.7)  # 70% train, 30% hidden (will be further split to validation and test)
    rings_train = pd.concat([rings[:train_split_index]] * resampling_fac)  # resample (repeat) rings
    rings_hidden = rings[train_split_index:]  # do not resample validation, instead will select same num. of non-rings

    not_rings_train = not_rings[:len(rings_train)]
    not_rings_hidden = not_rings[len(rings_train):].sample(len(rings_hidden), replace=False, random_state=seed)

    ring_catalog_train = pd.concat([rings_train, not_rings_train])
    ring_catalog_hidden = pd.concat([rings_hidden, not_rings_hidden.sample(len(rings_hidden), random_state=seed)])  # not crucial to resample rings to increase, just decrease non-rings (for simplicity)

    # shuffle
    ring_catalog_train = ring_catalog_train.sample(len(ring_catalog_train), random_state=seed)
    ring_catalog_hidden = ring_catalog_hidden.sample(len(ring_catalog_hidden), random_state=seed)

    # split ring_catalog_hidden into validation and test
    val_split_index = int(len(ring_catalog_hidden) * 0.33)  # 1/3rd of 30%, 2/3rds of 30% -> 10% and 20%, respectively 
    ring_catalog_val = ring_catalog_hidden[:val_split_index]
    ring_catalog_test = ring_catalog_hidden[val_split_index:]

    logging.info('Train labels: \n {}'.format(pd.value_counts(ring_catalog_train['label'])))  # will be exactly balanced between rings and non-rings
    # logging.info('Hidden labels: \n {}'.format(pd.value_counts(ring_catalog_hidden['label'])))
    logging.info('Validation labels: \n {}'.format(pd.value_counts(ring_catalog_val['label'])))
    logging.info('Test labels: \n {}'.format(pd.value_counts(ring_catalog_test['label'])))
    # val and test will have the same total number of rings and non-rings, but a slightly different ratio within each due to random shuffle then split
    # that feels okay and realistic

    paths_train, paths_val, paths_test = list(ring_catalog_train['local_png_loc']), list(ring_catalog_val['local_png_loc']), list(ring_catalog_test['local_png_loc'])
    labels_train, labels_val, labels_test = list(ring_catalog_train['label']), list(ring_catalog_val['label']), list(ring_catalog_test['label'])

    # check that no repeated rings ended up in multiple datasets
    assert set(paths_train).intersection(set(paths_val)) == set()
    assert set(paths_train).intersection(set(paths_test)) == set()
    assert set(paths_val).intersection(set(paths_test)) == set()

    # iauname_file_name = 'advanced_ring_iaunames_fractions'
    # with open('data/{}_train.json'.format(iauname_file_name), 'w') as f:
    #     json.dump([os.path.basename(x).replace('.png', '') for x in paths_train], f)
    # with open('data/{}_val.json'.format(iauname_file_name), 'w') as f:
    #     json.dump([os.path.basename(x).replace('.png', '') for x in paths_val], f)
    # with open('data/{}_test.json'.format(iauname_file_name), 'w') as f:
    #     json.dump([os.path.basename(x).replace('.png', '') for x in paths_test], f)
    # exit()

    # shuffled, so can just take the top N

    if train_dataset_size is None:
        train_dataset_size = len(paths_train)
    raw_train_dataset = image_datasets.get_image_dataset(paths_train[:train_dataset_size], file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_train[:train_dataset_size])
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_val)
    raw_test_dataset = image_datasets.get_image_dataset(paths_test, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_test)

    return raw_train_dataset, raw_val_dataset, raw_test_dataset


def get_rough_class_from_ring_fraction(fractions):
    is_ring = fractions > 0.25
    is_not_ring = fractions < 0.05
    uncertain = (~is_ring) & (~is_not_ring)
    # ring -> 1, not ring -> 0, uncertain -> -1
    labels = np.where(is_ring, 1, 0)
    labels[uncertain] = -1
    return labels


def get_ring_feature_dataset(train_dataset_size=None, shuffle=True, iauname_file_name='advanced_ring_iaunames_fractions'):
    feature_df = pd.read_parquet('../morphology-tools/anomaly/data/cnn_features_concat.parquet')
    assert not any(feature_df.duplicated(subset=['iauname']))
    feature_cols = [col for col in feature_df.columns.values if 'pred' in col]
    logging.info('Loaded {} features e.g. {}, for {} galaxies'.format(len(feature_cols), feature_cols[0], len(feature_df)))

    label_df = pd.read_parquet('data/rare_features_dr5_with_ml_morph.parquet')
    assert not any(label_df.duplicated(subset=['iauname']))

    # label_df['ring_label'] = label_df['ring'] == 1
    label_df['label'] = get_rough_class_from_ring_fraction(label_df['rare-features_ring_fraction'])  # must match above
    label_df = label_df[label_df['label'] != -1].reset_index()

    logging.info('Loaded labels of: \n{}'.format(label_df['label'].value_counts()))

    feature_label_df = pd.merge(feature_df, label_df, on='iauname', how='inner')  # 1-1 mapping
    assert len(feature_label_df) == min(len(feature_df), len(label_df))
    del label_df
    del feature_df

    with open('data/{}_train.json'.format(iauname_file_name), 'r') as f:
        train_iaunames = json.load(f)
    with open('data/{}_val.json'.format(iauname_file_name), 'r') as f:
        val_iaunames = json.load(f)
    with open('data/{}_test.json'.format(iauname_file_name), 'r') as f:
        test_iaunames = json.load(f)

    logging.info('Loaded {} iaunames ({} unique) for train dataset'.format(len(train_iaunames), len(set(train_iaunames))))
    logging.info('Loaded {} iaunames ({} unique) for val dataset'.format(len(val_iaunames), len(set(val_iaunames))))
    logging.info('Loaded {} iaunames ({} unique) for val dataset'.format(len(test_iaunames), len(set(test_iaunames))))

    # must be no galaxies in both train and validation
    # (probably a cleaner way to do this with sets)
    assert len(set(train_iaunames).intersection(set(val_iaunames))) == 0
    assert len(set(train_iaunames).intersection(set(test_iaunames))) == 0
    assert len(set(val_iaunames).intersection(set(test_iaunames))) == 0

    train_rows = [{'iauname': iauname, 'dataset_split': 'train'} for iauname in train_iaunames]
    val_rows = [{'iauname': iauname, 'dataset_split': 'validation'} for iauname in val_iaunames]
    test_rows = [{'iauname': iauname, 'dataset_split': 'test'} for iauname in test_iaunames]
    split_df = pd.DataFrame(data=train_rows+val_rows+test_rows)
    # assumes no other splits e.g. test dataset
    logging.info('Labelled galaxies in each dataset: {}'.format(split_df.value_counts('dataset_split')))

    # merge against split_df to copy feature_label_df many times as needed
    df = pd.merge(split_df, feature_label_df, how='inner', on='iauname')  # many-one
    assert len(df) == len(split_df) 
    del feature_label_df

    if shuffle:
        logging.info('Shuffling to re-order (fixed) train and val datasets.')
        df = df.sample(len(df))
    else:
        logging.warning('Not shuffling (fixed) train and val datasets. Train may be e.g. all not-ring then all ring - be careful.')

    train_df = df.query('dataset_split == "train"')
    val_df = df.query('dataset_split == "validation"')
    test_df = df.query('dataset_split == "test"')
    logging.info('Train labels: {}'.format(train_df.value_counts('label')))
    logging.info('Validation labels: {}'.format(val_df.value_counts('label')))
    logging.info('Test labels: {}'.format(test_df.value_counts('label')))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_df[feature_cols], train_df['label']))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_df[feature_cols], val_df['label']))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df[feature_cols], test_df['label']))

    # cut off as needed for requested train_dataset_size
    initial_train_dataset_size = len([_ for _ in train_dataset])  # read into memory at once, nbd
    if dataset_size is None:
      dataset_size = initial_train_dataset_size  # may be a small rounding error up to 1 batch size
    # train_dataset = train_dataset.take(100)
    train_dataset = train_dataset.take(dataset_size)

    logging.info('Initial train dataset size: {}'.format(initial_train_dataset_size))
    logging.info('Size after cutting up: {}'.format(dataset_size))

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':  # debugging only

    logging.basicConfig(level=logging.INFO)

    # get_advanced_ring_image_dataset(batch_size=16, requested_img_size=64)
    get_ring_feature_dataset()
