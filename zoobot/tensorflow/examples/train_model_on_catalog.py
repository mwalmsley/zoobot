import logging
import argparse
from multiprocessing.sharedctypes import Value
import os

import pandas as pd
import tensorflow as tf
import wandb

from zoobot.shared import label_metadata, schemas
from zoobot.tensorflow.training import train_with_keras


if __name__ == '__main__':

    """
    Convenient command-line API/example for training models on a catalog of images.
    Interfaces with Zoobot via ``train_with_keras.train``

    Note that training on TFRecords is now deprecated. 
    Train directly on the image files, as listed in a catalog.

    Example use:

    python zoobot/tensorflow/examples/train_model_on_shards.py \
        --experiment-dir /will/save/model/here \
        --resize-size 224 \
        --catalog-loc path/to/some/catalog.csv
        --wandb
    """

    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=logging.INFO
    )

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # check which GPU we're using
    physical_devices = tf.config.list_physical_devices('GPU')
    logging.info('GPUs: {}'.format(physical_devices))

    parser = argparse.ArgumentParser()
    # args re. what to train on
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--catalog',
                        dest='catalog_loc', type=str, action='append')
    # TODO note - no num_workers arg, tf does this automatically
    # how to train
    parser.add_argument('--epochs', dest='epochs', type=int, 
        help='Supports multiple space-separated paths')
    parser.add_argument('--resize-size', dest='resize_size',
                        type=int, default=224)
    parser.add_argument('--batch-size', dest='batch_size',
                        default=128, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    # TODO note - no nodes arg, not supported (yet)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--eager', default=False, action='store_true',
                        help='Use TensorFlow eager mode. Great for debugging, but significantly slower to train.'),
    parser.add_argument('--test-time-augs', dest='always_augment', default=False, action='store_true',
                        help='Zoobot includes keras.preprocessing augmentation layers. \
        These only augment (rotate/flip/etc) at train time by default. \
        They can be enabled at test time as well, which gives better uncertainties (by increasing variance between forward passes) \
        but may be unexpected and mess with e.g. GradCAM techniques.'),
    parser.add_argument('--dropout-rate', dest='dropout_rate',
                        default=0.2, type=float)
    parser.add_argument('--patience', dest='patience',
                        default=8, type=int)
    parser.add_argument('--mixed-precision', dest='mixed_precision', default=False, action='store_true',
                        help='If true, use automatic mixed precision (via PyTorch Lightning) to reduce GPU memory use (~x2). Else, use full (32 bit) precision')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                        help='If true, cut each catalog down to 5k galaxies (for quick training). Should cause overfitting.')
    args = parser.parse_args()

    question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs  
    dependencies = label_metadata.gz2_and_decals_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    # load each csv catalog file into a combined pandas data frame
    # Note: this requires the same csv column format across csv files
    if '.csv' in args.catalog_loc:
        catalog_reader = pd.read_csv
    elif '.parquet' in args.catalog_loc:
        catalog_reader = pd.read_parquet
    else:
        raise ValueError('Extension not automatically understood as csv or parquet: {}'.format(args.catalog_loc))

    catalog = pd.concat(
        map(catalog_reader, args.catalog_loc),
        ignore_index=True
    )
    # morph local file locations
    catalog['file_loc'] = catalog['file_loc'].str.replace(
        '/raid/scratch',  '/share/nas2')
    # print the first and last file loc of the loaded catalog
    logging.info('Catalog has {} rows'.format(len(catalog.index)))
    logging.info('First file_loc {}'.format(catalog['file_loc'].iloc[0]))
    logging.info('Last file_loc {}'.format(catalog['file_loc'].iloc[len(catalog.index) - 1]))

    # debug mode
    if args.debug:
        logging.warning(
            'Using debug mode: cutting catalog down to 5k galaxies')
        catalog = catalog.sample(5000).reset_index(drop=True)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir=args.save_dir)
        wandb.init(
            sync_tensorboard=True,
            project='zoobot-tf',  # TODO rename
            name=os.path.basename(args.save_dir)
        )
    #   with TensorFlow, doesn't need to be passed to trainer

    train_with_keras.train(
        save_dir=args.save_dir,
        catalog=catalog,
        schema=schema,
        # architecture_name=args.architecture_name, TODO not yet implemented
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        # augmentation parameters
        color=args.color,
        resize_size=args.resize_size,
        always_augment=args.always_augment,
        # hardware params
        mixed_precision=args.mixed_precision,
        gpus=args.gpus,
        eager=args.eager, 
        # other hparams
        dropout_rate=args.dropout_rate,
        # no wandb logger required, tensorboard patch applied above instead
    )
