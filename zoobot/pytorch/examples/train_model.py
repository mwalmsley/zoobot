import logging
import os
import argparse
import logging
import pandas as pd

import pytorch_lightning as pl

from zoobot import schemas, label_metadata
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.datasets import decals_dr8
from zoobot.pytorch.training import losses

# import wandb

if __name__ == '__main__':

    logging.basicConfig(
      format='%(levelname)s:%(message)s',
      level=logging.INFO
    )


    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--catalog', dest='catalog_loc', type=str)  # expects catalog, not tfrecords
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--shard-img-size', dest='shard_img_size', type=int, default=300)
    parser.add_argument('--resize-size', dest='resize_size', type=int, default=224)
    parser.add_argument('--batch-size', dest='batch_size', default=256, type=int)
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--test-time-augs', dest='always_augment', default=False, action='store_true',
        help='Zoobot includes keras.preprocessing augmentation layers. \
        These only augment (rotate/flip/etc) at train time by default. \
        They can be enabled at test time as well, which gives better uncertainties (by increasing variance between forward passes) \
        but may be unexpected and mess with e.g. GradCAM techniques.'),
    parser.add_argument('--dropout-rate', dest='dropout_rate', default=0.2, type=float)
    args = parser.parse_args()

    "shared setup"

    # a bit awkward, but I think it is better to have to specify you def. want color than that you def want greyscale
    greyscale = not args.color
    if greyscale:
      logging.info('Converting images to greyscale before training')
      channels = 1
    else:
      logging.warning('Training on color images, not converting to greyscale')
      channels = 3

    catalog_loc = args.catalog_loc
    initial_size = args.shard_img_size
    resize_size = args.resize_size
    batch_size = args.batch_size
    always_augment = not args.always_augment

    epochs = args.epochs
    save_dir = args.save_dir

    assert save_dir is not None
    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)

    question_answer_pairs = label_metadata.decals_pairs
    dependencies = label_metadata.get_gz2_and_decals_dependencies(question_answer_pairs)
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    "shared setup ends"

    loss_func = losses.calculate_multiquestion_loss

    model = define_model.ZoobotModel(schema=schema, loss=loss_func)

    catalog = pd.read_csv(catalog_loc).sample(1000)  # debugging
    catalog['file_loc'] = catalog['file_loc'].str.replace('/share/nas',  '/share/nas2')

    datamodule = decals_dr8.DECALSDR8DataModule(catalog, schema, greyscale=greyscale)

    trainer = pl.Trainer(accelerator="gpu", epochs=epochs)
    trainer.fit(model, datamodule, enable_checkpointing=True, default_root_dir=save_dir)
