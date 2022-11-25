import logging
import os
import argparse

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb

from galaxy_datasets import gz_decals_5

from zoobot.shared import label_metadata, schemas
from zoobot.pytorch.training import train_with_pytorch_lightning


if __name__ == '__main__':

    """
    Used to create the PyTorch pretrained weights checkpoints
    See .sh file of the same name for args used.

    See zoobot/pytorch/examples/minimal_examples.py for a friendlier example
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--data-dir', dest='data_dir', type=str)
    parser.add_argument('--architecture', dest='architecture_name', default='efficientnet', type=str)
    parser.add_argument('--resize-after-crop', dest='resize_after_crop',
                        type=int, default=224)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--batch-size', dest='batch_size',
                        default=256, type=int)
    parser.add_argument('--gpus', dest='gpus', default=1, type=int)
    parser.add_argument('--mixed-precision', dest='mixed_precision',
                        default=False, action='store_true')
    parser.add_argument('--debug', dest='debug',
                        default=False, action='store_true')
    parser.add_argument('--wandb', dest='wandb',
                        default=False, action='store_true')
    parser.add_argument('--seed', dest='random_state', default=42, type=int)
    args = parser.parse_args()

    # random_state = args.random_state
    import numpy as np
    random_state = np.random.randint(0, 10000)

    # already manually seeding the random bits below, but just in case
    pl.seed_everything(random_state)

    question_answer_pairs = label_metadata.decals_dr5_ortho_pairs  # decals dr5 only
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    # use the gz_decals() methods in galaxy_datasets.prepared_datasets to get the canonical (i.e. standard) train and test catalogs
    canonical_train_catalog, _ = gz_decals_5(root=args.data_dir, train=True, download=True)
    canonical_test_catalog, _ = gz_decals_5(root=args.data_dir, train=False, download=True)

    train_catalog, val_catalog = train_test_split(canonical_train_catalog, test_size=0.1, random_state=random_state)
    test_catalog = canonical_test_catalog.copy()

    logging.warning(val_catalog.iloc[0]['id_str'])
    # exit()

    # debug mode
    if args.debug:
        logging.warning(
            'Using debug mode: cutting catalogs down to 5k galaxies each')
        train_catalog = train_catalog.sample(5000).reset_index(drop=True)
        val_catalog = val_catalog.sample(5000).reset_index(drop=True)
        test_catalog = test_catalog.sample(5000).reset_index(drop=True)
        epochs = 2
    else:
        epochs = 1000

    if args.wandb:
        wandb_logger = WandbLogger(
            project='zoobot-benchmarks',
            name=os.path.basename(args.save_dir),
            log_model=True
        )
        wandb_logger.log_text(key="train_catalog", dataframe=train_catalog.sample(10))
        wandb_logger.log_text(key="val_catalog", dataframe=train_catalog.sample(10))
        wandb_logger.log_text(key="test_catalog", dataframe=train_catalog.sample(10))
    else:
        wandb_logger = None

    train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        save_dir=args.save_dir,
        schema=schema,
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        architecture_name=args.architecture_name,
        batch_size=args.batch_size,
        epochs=epochs,  # rely on early stopping
        patience=20, # increased as 8 seemed to stop too early (~300 epochs)
        # augmentation parameters
        color=args.color,
        resize_after_crop=args.resize_after_crop,
        # hardware parameters
        nodes=1,
        gpus=args.gpus,
        mixed_precision=args.mixed_precision,
        wandb_logger=wandb_logger,
        prefetch_factor=4,
        num_workers=11,  # system has 24 cpu, 12 cpu per gpu, leave a little wiggle room
        random_state=random_state,
        learning_rate=1e-3,
    )

    wandb.finish()