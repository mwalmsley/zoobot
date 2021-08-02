  
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


if __name__ == '__main__':
    """
    To make model for smooth/featured (also change cols below):
      # python train_model.py --experiment-dir results/smooth_or_featured_offline --shard-img-size 128 --train-dir data/decals/shards/multilabel_master_filtered_128/train --eval-dir data/decals/shards/multilabel_master_filtered_128/eval --epochs 1000 
      python train_model.py --experiment-dir results/smooth_or_featured_offline --shard-img-size 256 --train-dir data/decals/shards/multilabel_master_filtered_256/train --eval-dir data/decals/shards/multilabel_master_filtered_256/eval --epochs 1000 --batch-size 8 --resize-size 128

    To make model for predictions on all cols, for appropriate galaxies only:
      python train_model.py --experiment-dir results/latest_offline_featured --shard-img-size 128 --train-dir data/decals/shards/multilabel_master_filtered_128/train --eval-dir data/decals/shards/multilabel_master_filtered_128/eval --epochs 1000 
    
    DECALS testing:
      python train_model.py --experiment-dir ~/repos/zoobot_private/results/debug --shard-img-size 64 --train-dir ~/repos/zoobot_private/data/decals/shards/all_2p5_unfiltered_retired/train_shards --eval-dir ~/repos/zoobot_private/data/decals/shards/all_2p5_unfiltered_retired/eval_shards --epochs 2 --batch-size 8 --resize-size 64
      python train_model.py --experiment-dir /raid/scratch/walml//repos/zoobot/results/debug --shard-img-size 64 --train-dir /raid/scratch/walml/galaxy_zoo/decals/tfrecords/all_2p5_unfiltered_retired/train_shards --eval-dir /raid/scratch/walml/galaxy_zoo/decals/tfrecords/all_2p5_unfiltered_retired/eval_shards --epochs 2 --batch-size 8 --resize-size 64


    GZ2 testing:
      python train_model.py --experiment-dir results/debug --shard-img-size 300 --train-dir data/gz2/shards/all_featp5_facep5_sim_2p5_300/train_shards --eval-dir data/gz2/shards/all_featp5_facep5_sim_2p5_300/eval_shards --epochs 2 --batch-size 8 --resize-size 128
      python train_model.py --experiment-dir results/debug --shard-img-size 300 --train-dir data/gz2/shards/all_sim_2p5_unfiltered_300/train_shards --eval-dir data/gz2/shards/all_sim_2p5_unfiltered_300/train_shards --epochs 1 --batch-size 8 --resize-size 128
      python train_model.py --experiment-dir results/debug --shard-img-size 64 --train-dir data/gz2/shards/debug_sim/train_shards --eval-dir data/gz2/shards/debug_sim/eval_shards --epochs 2 --batch-size 8 --resize-size 64

    Local testing:
      python train_model.py --experiment-dir results/debug --shard-img-size 64 --resize-size 224 --train-dir data/decals/shards/all_2p5_unfiltered_retired/train_shards --eval-dir data/decals/shards/all_2p5_unfiltered_retired/eval_shards --epochs 2 --batch-size 8
      
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
    parser.add_argument('--batch-size', dest='batch_size', default=64, type=int)
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    args = parser.parse_args()

    # greyscale = True
    greyscale = False
    if greyscale:
      channels = 1
    else:
      channels = 3

    initial_size = args.shard_img_size
    resize_size = args.resize_size  # step time prop. to resolution
    batch_size = args.batch_size

    epochs = args.epochs
    train_records_dir = args.train_records_dir
    eval_records_dir = args.eval_records_dir
    save_dir = args.save_dir

    assert save_dir is not None
    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)

    question_answer_pairs = label_metadata.decals_pairs
    dependencies = label_metadata.get_gz2_and_decals_dependencies(question_answer_pairs)
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
    eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

    if args.distributed:
      logging.info('Using distributed mirrored strategy')
      strategy = tf.distribute.MultiWorkerMirroredStrategy()
      # strategy = tf.distribute.MirroredStrategy()
      context_manager = strategy.scope()
      logging.info('Replicas: {}'.format(strategy.num_replicas_in_sync))
    else:
      logging.info('Using single GPU, not distributed')
      context_manager = contextlib.nullcontext()  # does nothing, just a convenience for clean code

    raw_train_dataset = tfrecord_datasets.get_dataset(train_records, schema.label_cols, batch_size, shuffle=True)
    raw_test_dataset = tfrecord_datasets.get_dataset(eval_records, schema.label_cols, batch_size, shuffle=False)
  
    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=schema.label_cols,
        input_size=initial_size,
        make_greyscale=greyscale,
        normalise_from_uint8=False
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    test_dataset = preprocess.preprocess_dataset(raw_test_dataset, preprocess_config)

    with context_manager:

      model = define_model.get_model(
        output_dim=len(schema.label_cols),
        input_size=initial_size, 
        crop_size=int(initial_size * 0.75),
        resize_size=resize_size,
        channels=channels
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
      # you need to make this file yourself, with your api key and nothing else
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

    # inplace on model
    training_config.train_estimator(
      model, 
      train_config,  # parameters for how to train e.g. epochs, patience
      train_dataset,
      test_dataset,
      eager=False
    )
