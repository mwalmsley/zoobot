import logging
import argparse
import os

import tensorflow as tf
import wandb


from zoobot.shared import label_metadata, schemas
from zoobot.tensorflow.training import train_with_keras


if __name__ == '__main__':

    """
    Convenient command-line API/example for training models on previously-created shards
    Interfaces with Zoobot via ``train_with_keras.train``

    Example use:

    python zoobot/tensorflow/examples/train_model_on_shards.py \
        --experiment-dir /will/save/model/here \
        --shard-img-size 300 \
        --resize-size 224 \
        --train-dir /dir/with/training/tfrecords \
        --test-dir /dir/with/test/tfrecords
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
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--shard-img-size',
                        dest='shard_img_size', type=int, default=300)
    parser.add_argument('--resize-size', dest='resize_size',
                        type=int, default=224)
    parser.add_argument('--train-dir', dest='train_records_dir', type=str)
    parser.add_argument('--test-dir', dest='test_records_dir', type=str)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size',
                        default=128, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--eager', default=False, action='store_true',
                        help='Use TensorFlow eager mode. Great for debugging, but significantly slower to train.'),
    parser.add_argument('--test-time-augs', dest='always_augment', default=False, action='store_true',
                        help='Zoobot includes keras.preprocessing augmentation layers. \
        These only augment (rotate/flip/etc) at train time by default. \
        They can be enabled at test time as well, which gives better uncertainties (by increasing variance between forward passes) \
        but may be unexpected and mess with e.g. GradCAM techniques.'),
    parser.add_argument('--dropout-rate', dest='dropout_rate',
                        default=0.2, type=float)
    args = parser.parse_args()

    train_records_dir = args.train_records_dir
    test_records_dir = args.test_records_dir

    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(
        train_records_dir) if x.endswith('.tfrecord')]
    test_records = [os.path.join(test_records_dir, x) for x in os.listdir(
        test_records_dir) if x.endswith('.tfrecord')]

    question_answer_pairs = label_metadata.decals_dr5_ortho_pairs
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

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
        schema=schema,
        train_records=train_records,
        test_records=test_records,
        shard_img_size=args.shard_img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        dropout_rate=args.dropout_rate,
        color=args.color,
        resize_size=args.resize_size,
    )
