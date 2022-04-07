import logging
import os
import argparse

import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from zoobot.shared import label_metadata, schemas
from zoobot.pytorch.training import train_with_pytorch_lightning

if __name__ == '__main__':

    """
    See zoobot/pytorch/examples/train_model_on_catalog for a version training on a catalog without prespecifing the splits
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--architecture', dest='model_architecture', default='efficientnet', type=str)
    parser.add_argument('--resize-size', dest='resize_size',
                        type=int, default=224)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--batch-size', dest='batch_size',
                        default=256, type=int)
    parser.add_argument('--gpus', default=1, type=int)

    parser.add_argument('--mixed-precision', dest='mixed_precision',
                        default=False, action='store_true')
    parser.add_argument('--debug', dest='debug',
                        default=False, action='store_true')
    args = parser.parse_args()

    question_answer_pairs = label_metadata.decals_pairs  # decals dr5
    dependencies = label_metadata.gz2_and_decals_dependencies  # decals dr5
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    # explicit splits provided (instead of just a single catalog)
    train_catalog_locs = [
        # '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr12/train_shards/train_df.csv',
        '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/train_shards/train_df.csv',
        # '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr8/train_shards/train_df.csv'
    ]
    val_catalog_locs = [
        # '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr12/val_shards/val_df.csv',
        '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/val_shards/val_df.csv',
        # '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr8/val_shards/val_df.csv'
    ]
    test_catalog_locs = [
        # '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr12/test_shards/test_df.csv',
        '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/test_shards/test_df.csv',
        # '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr8/test_shards/test_df.csv'
    ]
    answer_columns = [a.text for a in schema.answers]
    useful_columns = answer_columns + ['file_loc']

    train_catalog = pd.concat(
        [pd.read_csv(loc, usecols=useful_columns) for loc in train_catalog_locs])
    val_catalog = pd.concat(
        [pd.read_csv(loc, usecols=useful_columns) for loc in val_catalog_locs])
    test_catalog = pd.concat(
        [pd.read_csv(loc, usecols=useful_columns) for loc in test_catalog_locs])

    for catalog in (train_catalog, val_catalog, test_catalog):
        # tweak file paths
        catalog['file_loc'] = catalog['file_loc'].str.replace(
            '/raid/scratch',  '/share/nas2')
        catalog['file_loc'] = catalog['file_loc'].str.replace(
            '/dr8_downloader/',  '/dr8/')
        catalog['file_loc'] = catalog['file_loc'].str.replace(
            r'/png/', r'/jpeg/')
        catalog['file_loc'] = catalog['file_loc'].str.replace('.png', '.jpeg')

        # enforce datatypes
        for answer_col in answer_columns:
            catalog[answer_col] = catalog[answer_col].astype(int)
            catalog['file_loc'] = catalog['file_loc'].astype(str)

        logging.info(catalog['file_loc'].iloc[0])

    # debug mode
    if args.debug:
        logging.warning(
            'Using debug mode: cutting catalogs down to 5k galaxies each')
        train_catalog = train_catalog.sample(5000).reset_index(drop=True)
        val_catalog = val_catalog.sample(5000).reset_index(drop=True)
        test_catalog = test_catalog.sample(5000).reset_index(drop=True)

    if args.wandb:
        wandb_logger = WandbLogger(
            project='zoobot-pytorch-dr5-presplit-replication',
            name=os.path.basename(args.save_dir),
            log_model="all")
        # only rank 0 process gets access to the wandb.run object, and for non-zero rank processes: wandb.run = None
        # https://docs.wandb.ai/guides/integrations/lightning#how-to-use-multiple-gpus-with-lightning-and-w-and-b
    else:
        wandb_logger = None

    train_with_pytorch_lightning.train(
        save_dir=args.save_dir,
        catalog=catalog,
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        schema=schema,
        model_architecture=args.model_architecture,
        batch_size=args.batch_size,
        epochs=args.epochs,
        # augmentation parameters
        color=args.color,
        resize_size=args.resize_size,
        # hardware parameters
        nodes=1,
        gpus=args.gpus,
        mixed_precision=args.mixed_precision,
        wandb_logger=wandb_logger
    )
