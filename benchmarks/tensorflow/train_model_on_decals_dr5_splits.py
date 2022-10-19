import logging
import argparse
import os

import tensorflow as tf
import wandb
from sklearn.model_selection import train_test_split

from pytorch_galaxy_datasets.prepared_datasets import decals_dr5_setup

from zoobot.shared import label_metadata, schemas
from zoobot.tensorflow.training import train_with_keras


if __name__ == '__main__':

    """
    See zoobot/tensorflow/examples/train_model_on_catalog for a version training on a catalog without prespecifing the splits

    This will automatically download GZ DECaLS DR5, which is ~220k galaxies and ~11GB.
    I use pytorch-galaxy-datasets as convenient downloader, but am actually using tensorflow otherwise
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
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--data-dir', dest='data_dir', type=str)
    parser.add_argument('--resize-size', dest='resize_size',
                        type=int, default=224)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size',
                        default=512, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--mixed-precision', dest='mixed_precision', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--eager', default=False, action='store_true',
                        help='Use TensorFlow eager mode. Great for debugging, but significantly slower to train.'),
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    question_answer_pairs = label_metadata.decals_dr5_ortho_pairs  # dr5
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    # use the setup() methods in pytorch_galaxy_datasets.prepared_datasets to get the canonical (i.e. standard) train and test catalogs
    canonical_train_catalog, _ = decals_dr5_setup(root=args.data_dir, train=True, download=True)
    canonical_test_catalog, _ = decals_dr5_setup(root=args.data_dir, train=False, download=True)

    train_catalog, val_catalog = train_test_split(canonical_train_catalog, test_size=0.1)  # could add random_state
    test_catalog = canonical_test_catalog.copy()

    # debug mode
    if args.debug:
        logging.warning('Using debug mode: cutting catalogs down to 5k galaxies each')
        train_catalog = train_catalog.sample(5000).reset_index(drop=True)
        val_catalog = val_catalog.sample(5000).reset_index(drop=True)
        test_catalog = test_catalog.sample(5000).reset_index(drop=True)
        epochs = 2
    else:
        epochs = args.epochs

    if args.wandb:
        wandb.tensorboard.patch(root_logdir=args.save_dir)
        wandb.init(
            sync_tensorboard=True,
            project='zoobot-pytorch-dr5-presplit-replication',  # TODO rename
            name=os.path.basename(args.save_dir)
        )
    #   with TensorFlow, doesn't need to be passed as arg

    train_with_keras.train(
        save_dir=args.save_dir,
        schema=schema,
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=args.batch_size,
        eager=args.eager,
        gpus=args.gpus,
        epochs=epochs,
        dropout_rate=0.2,
        color=args.color,
        resize_size=224,
        mixed_precision=args.mixed_precision
    )
