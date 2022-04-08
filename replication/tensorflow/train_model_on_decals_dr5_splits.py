import logging
import argparse
import os

import tensorflow as tf
import wandb


from zoobot.shared import label_metadata, schemas
from zoobot.tensorflow.training import train_with_keras


if __name__ == '__main__':

    """
    Reproduce the EfficientNet models trained by W+22a
    Run on slurm with train_model_on_decals_dr5_splits.sh
    Requires previously-created shards - see shards.sh
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
    parser.add_argument('--train-dir', dest='train_records_dir', type=str)
    parser.add_argument('--test-dir', dest='test_records_dir', type=str)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size',
                        default=512, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--eager', default=False, action='store_true',
                        help='Use TensorFlow eager mode. Great for debugging, but significantly slower to train.'),
    args = parser.parse_args()

    train_records_dir = args.train_records_dir
    test_records_dir = args.test_records_dir

    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(
        train_records_dir) if x.endswith('.tfrecord')]
    test_records = [os.path.join(test_records_dir, x) for x in os.listdir(
        test_records_dir) if x.endswith('.tfrecord')]

    question_answer_pairs = label_metadata.decals_pairs  # dr5
    dependencies = label_metadata.gz2_and_decals_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

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
        train_records=train_records,
        test_records=test_records,
        shard_img_size=300,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        dropout_rate=0.2,
        color=args.color,
        resize_size=224,
    )
