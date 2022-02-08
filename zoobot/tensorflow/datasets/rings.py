import logging
import json
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from zoobot.tensorflow.data_utils import image_datasets


def get_advanced_ring_image_dataset(batch_size: int, requested_img_size: int, train_dataset_size=None, seed=1, file_format='png'):
    """
    Get train, val, and test tf.data.datasets containing images of ring galaxies and labels derived from GZ DECaLS volunteer votes.
    Does not apply any preprocessing or augmentations.

    This is a more complicated dataset than data/example_ring_tag_catalog.csv. 
    You may want to look at that file and zoobot/finetune_minimal.py first.

    Lastly, the train dataset can optionally be made smaller (restricted).
    I used this in a paper to test how the number of available labels changed the performance of the finetuned models.
    Note that, because galaxies are dropped randomly and rings are not unique (see get_random_ring_catalogs),
    the number of unique rings will drop faster than the number of (unique) non-rings).
    You should probably quote results for the number of *unique* rings, rather than the total restricted train dataset size.

    Train will have exactly balanced ring/non-rings unless restricting the train size, in which case the balance will vary slightly.
    Val and test will have the same total number of rings and non-rings, but similarly vary slightly within each due to the random shuffle then split.

    Args:
        batch_size (int): batch size for each tf.data.dataset to use (i.e. ds.batch(batch_size))
        requested_img_size (int): [description]
        train_dataset_size ([type], optional): [description]. Defaults to None.
        seed (int, optional): [description]. Defaults to 1.
        file_format (str, optional): [description]. Defaults to 'png'.

    Returns:
        tf.data.dataset: of training galaxies, yielding (image values, labels) batched pairs.
        tf.data.dataset: validation galaxies, as above.
        tf.data.dataset: test galaxies, as above.
    """
    ring_catalog_train, ring_catalog_val, ring_catalog_test = get_random_ring_catalogs(seed=seed, train_dataset_size=train_dataset_size)

    paths_train, paths_val, paths_test = list(ring_catalog_train['local_png_loc']), list(ring_catalog_val['local_png_loc']), list(ring_catalog_test['local_png_loc'])
    labels_train, labels_val, labels_test = list(ring_catalog_train['label']), list(ring_catalog_val['label']), list(ring_catalog_test['label'])

    # check that no repeated rings ended up in multiple datasets
    assert set(paths_train).intersection(set(paths_val)) == set()
    assert set(paths_train).intersection(set(paths_test)) == set()
    assert set(paths_val).intersection(set(paths_test)) == set()

    logging.info('Train labels after cut of {} galaxies (should be ~50/50): \n{}'.format(train_dataset_size, pd.value_counts(labels_train)))
    logging.info('Unique train rings: {}'.format(len(set([path for (path, label) in zip(paths_train, labels_train) if label == 1]))))

    if train_dataset_size is None:
        train_dataset_size = len(paths_train)
    raw_train_dataset = image_datasets.get_image_dataset(paths_train, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_train)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_val)
    raw_test_dataset = image_datasets.get_image_dataset(paths_test, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_test)

    return raw_train_dataset, raw_val_dataset, raw_test_dataset


def get_advanced_ring_feature_dataset(train_dataset_size=None, seed=1):
    """
    This is the equivalent of :meth:`get_advanced_ring_image_dataset` but returning datasets of (features, labels) rather than (images, labels).
    Features are the CNN internal representations saved beforehand for each galaxy.
    They are the 1280-dimensional activations of the penultimate layer of EfficientNetB0.
    See the morphology tools paper for more.

    See :meth:`get_advanced_ring_image_dataset` for notes on labels and on restricting the train dataset size.

    Args:
        train_dataset_size (int, optional): Max number of training galaxies. Defaults to None. See :meth:`get_advanced_ring_image_dataset`.
        seed (int, optional): Random seed for catalog shuffle and splits. Defaults to 1.

    Returns:
        tf.data.dataset: of training galaxies, yielding (features, label). Not batched yet (use ds.batch(batch_size)).
        tf.data.dataset: validation galaxies, as above.
        tf.data.dataset: test galaxies, as above.
    """
    # TODO move these features either into the repo or somewhere otherwise accessible (Zenodo?)
    feature_df = pd.read_parquet('../morphology-tools/anomaly/data/cnn_features_concat.parquet')
    assert not any(feature_df.duplicated(subset=['iauname']))
    feature_cols = [col for col in feature_df.columns.values if 'pred' in col]
    logging.info('Loaded {} features e.g. {}, for {} galaxies'.format(len(feature_cols), feature_cols[0], len(feature_df)))

    ring_catalog_train, ring_catalog_val, ring_catalog_test = get_random_ring_catalogs(seed=seed, train_dataset_size=train_dataset_size)

    train_df = pd.merge(ring_catalog_train, feature_df, on='iauname',how='inner')
    val_df = pd.merge(ring_catalog_val, feature_df, on='iauname', how='inner')
    test_df = pd.merge(ring_catalog_test, feature_df, on='iauname', how='inner')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_df[feature_cols], train_df['label']))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_df[feature_cols], val_df['label']))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df[feature_cols], test_df['label']))

    logging.info('Train size after cutting up: {}'.format(train_dataset_size))

    return train_dataset, val_dataset, test_dataset



def get_random_ring_catalogs(seed: int, train_dataset_size: int):
    """
    Get train/val/test catalogs of ring galaxies, each including 'label' (1 or 0) and 'local_png_loc' columns.
    Split is randomised according to `seed`.

    Galaxies are filtered to be not very smooth and roughly face-on using automatic vote fraction predictions (see GZ DECalS data release paper)
    The public repo already includes only those galaxies, so this may have no effect for you.

    Labels are calculated based on the GZ DECaLS "Are there any of these rare features?" "Ring" answer vote fraction.
    See :meth:`get_rough_class_from_ring_fraction`.

    Instead of balancing the classes by dropping most of the non-rings, I'm repeating the rings by a factor of ~5
    This includes more information but needs some footwork to make sure that no repeated ring ends up in both the train and test sets
    
    Args:
        seed (int): random seed for splitting to train/val/test

    Returns:
        pd.DataFrame: Catalog of galaxies to be used for training, including 'label' (1 or 0) and 'local_png_loc' columns
        pd.DataFrame: As above, for validation
        pd.DataFrame: As above, for testing
    """
    # TODO filter to only appropriate galaxies and columns, then include with repo
    # uses automatic morphology predictions from gz decals cnn
    ring_catalog = pd.read_parquet('data/rare_features_dr5_with_ml_morph.parquet')

    # some personal path fiddling TODO point to example data instead
    # ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].str.replace('/media/walml/beta1', '/Volumes/beta')  # local
    ring_catalog['local_png_loc'] = ring_catalog['local_png_loc'].str.replace('/media/walml/beta1/decals/png_native/dr5', '/share/nas/walml/galaxy_zoo/decals/dr5/png')  # galahad

    # apply selection cuts (in data/ring_votes_catalog_advanced.parquet, I only include galaxies that pass these cuts anyway)
    not_very_smooth = ring_catalog['smooth-or-featured_featured-or-disk_fraction'] > 0.25
    face = ring_catalog['disk-edge-on_no_fraction'] > 0.75
    passes_cuts = not_very_smooth & face

    ring_catalog = ring_catalog[passes_cuts].reset_index(drop=True)
    logging.info('Labels after selection cuts: \n{}'.format(pd.value_counts(ring_catalog['ring'])))

    ring_catalog['label'] = get_rough_class_from_ring_fraction(ring_catalog['rare-features_ring_fraction'])

    # new for review - fix previous possible error where possible positive ring examples in train set vs hidden set were not randomised?
    ring_catalog = ring_catalog.sample(len(ring_catalog), random_state=seed).reset_index(drop=True)

    rings = ring_catalog.query('label == 1')
    not_rings = ring_catalog.query('label == 0')
    logging.info('Rings: {}, Not Rings: {}'.format(len(rings), len(not_rings)))

    resampling_fac = int((len(not_rings)) / (len(rings)))
    logging.info('Resampling fraction: {}'.format(resampling_fac))

    train_split_index = int(len(rings) * 0.7)  # 70% train, 30% hidden (will be further split to validation and test)
    rings_train = pd.concat([rings[:train_split_index]] * resampling_fac)  # resample (repeat) the training rings
    rings_hidden = rings[train_split_index:]  # do not resample validation, instead will select same num. of non-rings

    not_rings_train = not_rings[:len(rings_train)]  # chose exactly that many training not-rings to match
    not_rings_hidden = not_rings[len(rings_train):].sample(len(rings_hidden), replace=False, random_state=seed)  # and chose exactly as many hidden not-rings from the remainder

    ring_catalog_train = pd.concat([rings_train, not_rings_train])
    ring_catalog_hidden = pd.concat([rings_hidden, not_rings_hidden.sample(len(rings_hidden), random_state=seed)])  # not crucial to resample rings to increase, just decrease non-rings (for simplicity)

    # shuffle
    ring_catalog_train = ring_catalog_train.sample(len(ring_catalog_train), random_state=seed)
    ring_catalog_hidden = ring_catalog_hidden.sample(len(ring_catalog_hidden), random_state=seed)

    # split ring_catalog_hidden into validation and test
    val_split_index = int(len(ring_catalog_hidden) * 0.33)  # 1/3rd of 30%, 2/3rds of 30% -> 10% and 20%, respectively 
    ring_catalog_val = ring_catalog_hidden[:val_split_index]
    ring_catalog_test = ring_catalog_hidden[val_split_index:]

    # train will initially be exactly balanced between rings and non-rings
    logging.info('Train labels: \n {}'.format(pd.value_counts(ring_catalog_train['label'])))
    unique_rings = len(ring_catalog_train.query('label == 1')['local_png_loc'].unique())
    logging.info('Unique training rings: {}'.format(unique_rings))

    # validation and test will be include exactly the same total number of rings/non-rings,
    # but the random split will make each be only close-to-balanced
    logging.info('Validation labels: \n {}'.format(pd.value_counts(ring_catalog_val['label'])))
    logging.info('Test labels: \n {}'.format(pd.value_counts(ring_catalog_test['label'])))

    # randomly drop train galaxies as requested by `train_dataset_size`
    if train_dataset_size is None:  # if not requested, don't drop any
        train_dataset_size = len(ring_catalog_train)
    assert train_dataset_size <= len(ring_catalog_train)
    rng = np.random.default_rng()
    train_indices_to_pick = rng.permutation(np.arange(len(ring_catalog_train)))[:train_dataset_size]
    ring_catalog_train_cut = ring_catalog_train.iloc[train_indices_to_pick].reset_index()

    logging.info('Train labels after restriction: \n {}'.format(pd.value_counts(ring_catalog_train_cut['label'])))

    # val and test will have the same total number of rings and non-rings, but a slightly different ratio within each due to random shuffle then split
    # that feels okay and realistic
    return ring_catalog_train_cut, ring_catalog_val, ring_catalog_test


def get_rough_class_from_ring_fraction(fractions: pd.Series):
    """
    Bin GZ DECaLS "ring" vote fractions into binary ring labels.
    If f > 0.25, label is 1
    If f < 0.05, label is 0
    If neither, label is -1

    Expects a pd.Series, np.array, or similar.

    Args:
        fractions (pd.Series): GZ DECaLS "ring" vote fractions.

    Returns:
        np.array: integer ring labels of 0, 1 or -1, according to those fractions
    """
    is_ring = fractions > 0.25
    is_not_ring = fractions < 0.05
    uncertain = (~is_ring) & (~is_not_ring)
    # ring -> 1, not ring -> 0, uncertain -> -1
    labels = np.where(is_ring, 1, 0)
    labels[uncertain] = -1
    return labels


if __name__ == '__main__':  # debugging only

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    get_advanced_ring_image_dataset(batch_size=16, requested_img_size=64, train_dataset_size=1590)
    # get_advanced_ring_feature_dataset(train_dataset_size=159)
