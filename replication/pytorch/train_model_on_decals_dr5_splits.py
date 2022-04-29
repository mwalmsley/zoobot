import logging
import os
import argparse

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger

from pytorch_galaxy_datasets.prepared_datasets import decals_dr5_setup

from zoobot.shared import label_metadata, schemas
from zoobot.pytorch.training import train_with_pytorch_lightning


if __name__ == '__main__':

    """
    See zoobot/pytorch/examples/train_model_on_catalog for a version training on a catalog without prespecifing the splits

    This will automatically download GZ DECaLS DR5, which is ~220k galaxies and ~11GB.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--data-dir', dest='data_dir', type=str)
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

    question_answer_pairs = label_metadata.decals_dr5_ortho_pairs  # decals dr5 only
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    # use the setup() methods in pytorch_galaxy_datasets.prepared_datasets to get the canonical (i.e. standard) train and test catalogs
    canonical_train_catalog, _ = decals_dr5_setup(data_dir=args.data_dir, train=True, download=True)
    canonical_test_catalog, _ = decals_dr5_setup(data_dir=args.data_dir, train=False, download=True)

    train_catalog, val_catalog = train_test_split(canonical_train_catalog, test_size=0.1)
    test_catalog = canonical_test_catalog.copy()

    # debug mode
    if args.debug:
        logging.warning(
            'Using debug mode: cutting catalogs down to 5k galaxies each')
        train_catalog = train_catalog.sample(5000).reset_index(drop=True)
        val_catalog = val_catalog.sample(5000).reset_index(drop=True)
        test_catalog = test_catalog.sample(5000).reset_index(drop=True)

    wandb_logger = WandbLogger(
        project='zoobot-pytorch-dr5-presplit-replication',
        name=os.path.basename(args.save_dir),
        log_model="all")
    # only rank 0 process gets access to the wandb.run object, and for non-zero rank processes: wandb.run = None
    # https://docs.wandb.ai/guides/integrations/lightning#how-to-use-multiple-gpus-with-lightning-and-w-and-b


    train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        save_dir=args.save_dir,
        schema=schema,
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        model_architecture=args.model_architecture,
        batch_size=args.batch_size,
        epochs=1000,  # rely on early stopping
        # augmentation parameters
        color=args.color,
        resize_size=args.resize_size,
        # hardware parameters
        nodes=1,
        gpus=args.gpus,
        mixed_precision=args.mixed_precision,
        wandb_logger=wandb_logger,
        prefetch_factor=4,
        num_workers=11  # system has 24 cpu, 12 cpu per gpu, leave a little wiggle room
    )
