import logging
import argparse
import os

import tensorflow as tf
tf.config.optimizer.set_jit(False)
import wandb
from sklearn.model_selection import train_test_split

from galaxy_datasets import gz_decals_5

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

    logging.info('Hello world')
    logging.warning('Hello world')
    logging.critical('Hello world')

    # import time
    # time.sleep(1)
    # print('exiting')
    # exit()

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # check which GPU we're using
    physical_devices = tf.config.list_physical_devices('GPU')
    logging.info('GPUs: {}'.format(physical_devices))

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--data-dir', dest='data_dir', type=str)
    parser.add_argument('--architecture', dest='architecture_name',
                        type=str, default='efficientnet')
    parser.add_argument('--resize-after-crop', dest='resize_after_crop',
                        type=int, default=224)
    parser.add_argument('--batch-size', dest='batch_size',
                        default=128, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--mixed-precision', dest='mixed_precision', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--seed', dest='random_state', default=42, type=int)
    parser.add_argument('--eager', default=False, action='store_true',
                        help='Use TensorFlow eager mode. Great for debugging, but significantly slower to train.'),
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    random_state = args.random_state

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    question_answer_pairs = label_metadata.decals_dr5_ortho_pairs  # dr5
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    # use the setup() methods in galaxy_datasets.prepared_datasets to get the canonical (i.e. standard) train and test catalogs
    # TODO could refactor to immediately give the canonical datasets from galaxy_datasets.tensorflow.datasets?
    if args.debug:
        download = False
    else:
        download = True
    canonical_train_catalog, _ = gz_decals_5(root=args.data_dir, train=True, download=download)
    canonical_test_catalog, _ = gz_decals_5(root=args.data_dir, train=False, download=download)

    train_catalog, val_catalog = train_test_split(canonical_train_catalog, test_size=0.1, random_state=random_state)
    test_catalog = canonical_test_catalog.copy()

    # debug mode
    if args.debug:
        logging.warning('Using debug mode: cutting catalogs down to 5k galaxies each')
        train_catalog = train_catalog.sample(5000).reset_index(drop=True)
        val_catalog = val_catalog.sample(5000).reset_index(drop=True)
        test_catalog = test_catalog.sample(5000).reset_index(drop=True)
        epochs = 2
    else:
        epochs = 1000

    if args.wandb:
    # root_logdir must match tensorboard logdir, not full logdir (aka root for /checkpoint, /tensorboard..)
    # tensorboard logdir is /tensorboard as per training_config.py
        wandb.tensorboard.patch(root_logdir=os.path.join(args.save_dir, 'tensorboard'))
        wandb.init(
            sync_tensorboard=True,
            project='zoobot-benchmarks',
            name=os.path.basename(args.save_dir)
        )
    #   with TensorFlow, doesn't need to be passed as arg
    # comment out if not desired to use wandb

    train_with_keras.train(
        save_dir=args.save_dir,
        schema=schema,
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=args.batch_size,
        architecture_name=args.architecture_name,
        eager=args.eager,
        gpus=args.gpus,
        epochs=epochs,
        dropout_rate=0.2,
        color=args.color,
        resize_after_crop=args.resize_after_crop,
        mixed_precision=args.mixed_precision,
        patience=20,
        # random state has no effect here yet as catalogs already split and tf not seeded
        random_state=random_state
    )
