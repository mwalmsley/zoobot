  
import os
import argparse
import time
import logging
import contextlib

import tensorflow as tf
import wandb

from zoobot.data_utils import tfrecord_datasets
from zoobot.training import training_config, losses
from zoobot.estimators import preprocess, define_model
from zoobot import schemas, label_metadata

# adapted from zoobot/train_model.py
if __name__ == '__main__':
    """
    DECaLS debugging (make the shards first with create_shards.py):
      python train_model.py --experiment-dir results/decals_debug --shard-img-size 32 --resize-size 224 --train-dir data/decals/shards/decals_debug/train_shards --eval-dir data/decals/shards/decals_debug/eval_shards --epochs 2 --batch-size 8
      
    DECaLS full:
      python train_model.py --experiment-dir results/decals_debug --shard-img-size 300 --train-dir /raid/scratch/walml/galaxy_zoo/decals/tfrecords/all_2p5_unfiltered_retired/train_shards --eval-dir /raid/scratch/walml/galaxy_zoo/decals/tfrecords/all_2p5_unfiltered_retired/eval_shards --epochs 200 --batch-size 256 --resize-size 224
    New features: add --distributed for multi-gpu, --wandb for weights&biases metric tracking, --color for color (does not perform better)

    GZ2 debugging:
      python train_model.py --experiment-dir results/gz2_debug --shard-img-size 300 --train-dir data/gz2/shards/all_sim_2p5_unfiltered_300/train_shards --eval-dir data/gz2/shards/all_sim_2p5_unfiltered_300/train_shards --epochs 1 --batch-size 8 --resize-size 128


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
    parser.add_argument('--shard-img-size', dest='shard_img_size', type=int, default=300)
    parser.add_argument('--resize-size', dest='resize_size', type=int, default=224)
    parser.add_argument('--train-dir', dest='train_records_dir', type=str)
    parser.add_argument('--eval-dir', dest='eval_records_dir', type=str)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--eager', default=False, action='store_true',
        help='Use TensorFlow eager mode. Great for debugging, but significantly slower to train.'),
    parser.add_argument('--no-test=augment', dest='no_test_augment', default=False, action='store_true'),
    parser.add_argument('--dropout-rate', dest='dropout_rate', default=0.2, type=float)
    args = parser.parse_args()
    
    # a bit awkward, but I think it is better to have to specify you def. want color than that you def want greyscale
    greyscale = not args.color
    if greyscale:
      logging.info('Converting images to greyscale before training')
      channels = 1
    else:
      logging.warning('Training on color images, not converting to greyscale')
      channels = 3

    always_augment = not args.no_test_augment,  # if no_test_augment, then *don't* use test-time augmentations

    initial_size = args.shard_img_size
    resize_size = args.resize_size
    batch_size = args.batch_size

    epochs = args.epochs
    train_records_dir = args.train_records_dir
    eval_records_dir = args.eval_records_dir
    save_dir = args.save_dir

    assert save_dir is not None
    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)

    question_answer_pairs = label_metadata.decals_pairs
    dependencies = gz2_and_decals_dependencies

    # TODO this bit is the bit I changed! otherwise copy-paste
    # question_answer_pairs = {'smooth-or-featured': question_answer_pairs['smooth-or-featured']}  # keep only this dict key
    # question_answer_pairs = {'bar': question_answer_pairs['bar']}  # keep only this dict key
    # question_answer_pairs = {'bulge-size': question_answer_pairs['bulge-size']}  # keep only this dict key
    question_answer_pairs = {'has-spiral-arms': question_answer_pairs['has-spiral-arms']}  # keep only this dict key


    # dependencies doesn't need to change

    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
    eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

    if args.distributed:
      logging.info('Using distributed mirrored strategy')
      strategy = tf.distribute.MirroredStrategy()  # one machine, one or more GPUs
      # strategy = tf.distribute.MultiWorkerMirroredStrategy()  # one or more machines. Not tested - you'll need to set this up for your own cluster.
      context_manager = strategy.scope()
      logging.info('Replicas: {}'.format(strategy.num_replicas_in_sync))
    else:
      logging.info('Using single GPU, not distributed')
      context_manager = contextlib.nullcontext()  # does nothing, just a convenience for clean code

    raw_train_dataset = tfrecord_datasets.get_dataset(train_records, schema.label_cols, batch_size, shuffle=True, drop_remainder=True)
    raw_test_dataset = tfrecord_datasets.get_dataset(eval_records, schema.label_cols, batch_size, shuffle=False, drop_remainder=True)
  
    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=schema.label_cols,
        input_size=initial_size,
        make_greyscale=greyscale,
        normalise_from_uint8=False  # False for tfrecords with 0-1 floats, True for png/jpg with 0-255 uints
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    test_dataset = preprocess.preprocess_dataset(raw_test_dataset, preprocess_config)

    with context_manager:

      model = define_model.get_model(
        output_dim=len(schema.label_cols),
        input_size=initial_size, 
        crop_size=int(initial_size * 0.75),
        resize_size=resize_size,
        channels=channels,
        always_augment=always_augment,
        dropout_rate=args.dropout_rate
      )
    
      multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
      # SUM reduction over loss, cannot divide by batch size on replicas when distributed training
      # so do it here instead
      loss = lambda x, y: multiquestion_loss(x, y) / batch_size  
      # loss = multiquestion_loss



    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam()
    )
    model.summary()

    train_config = training_config.TrainConfig(
      log_dir=save_dir,
      epochs=epochs,
      patience=10
    )

    if args.wandb:
      this_script_dir = os.path.dirname(__file__)
      # For weights&biases you need to make this file yourself, with your api key and nothing else
      with open(os.path.join(this_script_dir, 'wandb_api.txt'), 'r') as f:
        api_key = f.readline()
      wandb.login(key=api_key)
      wandb.init(sync_tensorboard=True)
      config = wandb.config
      config.label_cols=schema.label_cols,
      config.initial_size=initial_size
      config.greyscale = greyscale
      config.resize_size = resize_size
      config.batch_size = batch_size
      config.train_records = train_records
      config.epochs = epochs
      config.always_augment = always_augment
      config.dropout_rate = args.dropout_rate

    # inplace on model
    training_config.train_estimator(
      model, 
      train_config,  # parameters for how to train e.g. epochs, patience
      train_dataset,
      test_dataset,
      eager=args.eager  # set this True (or use --eager) for easier debugging, but slower training
    )
