import logging
import os
import argparse

import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from zoobot.shared import label_metadata, schemas
from zoobot.pytorch.training import train_with_pytorch_lightning


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    logging.info('Begin training on catalog example script')

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    # expects catalog, not tfrecords
    parser.add_argument('--catalog', dest='catalog_loc', type=str)
    parser.add_argument('--num_workers',
                        dest='num_workers', type=int, default=int((os.cpu_count() / 2)))
    parser.add_argument('--architecture',
                        dest='model_architecture', type=str, default='efficientnet')
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--resize-size', dest='resize_size',
                        type=int, default=224)
    parser.add_argument('--batch-size', dest='batch_size',
                        default=256, type=int)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--test-time-augs', dest='always_augment', default=False, action='store_true',
                        help='Zoobot includes keras.preprocessing augmentation layers. \
        These only augment (rotate/flip/etc) at train time by default. \
        They can be enabled at test time as well, which gives better uncertainties (by increasing variance between forward passes) \
        but may be unexpected and mess with e.g. GradCAM techniques.'),
    parser.add_argument('--dropout-rate', dest='dropout_rate',
                        default=0.2, type=float)
    parser.add_argument('--mixed-precision', dest='mixed_precision', default=False, action='store_true',
                        help='If true, use automatic mixed precision (via PyTorch Lightning) to reduce GPU memory use (~x2). Else, use full (32 bit) precision')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                        help='If true, cut each catalog down to 5k galaxies (for quick training). Should cause overfitting.')
    args = parser.parse_args()

    catalog_loc = args.catalog_loc

    question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    # catalog provided
    catalog = pd.read_csv(catalog_loc)
    catalog['file_loc'] = catalog['file_loc'].str.replace(
        '/raid/scratch',  '/share/nas2')
    logging.info(catalog['file_loc'].iloc[0]) 

    # debug mode
    if args.debug:
        logging.warning(
            'Using debug mode: cutting catalog down to 5k galaxies')
        catalog = catalog.sample(5000).reset_index(drop=True)

    if args.wandb:
        wandb_logger = WandbLogger(
            project='zoobot-pytorch-catalog-example',
            name=os.path.basename(args.save_dir),
            log_model="all")
        # only rank 0 process gets access to the wandb.run object, and for non-zero rank processes: wandb.run = None
        # https://docs.wandb.ai/guides/integrations/lightning#how-to-use-multiple-gpus-with-lightning-and-w-and-b
    else:
        wandb_logger = None

    train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        save_dir=args.save_dir,
        catalog=catalog,
        schema=schema,
        model_architecture=args.model_architecture,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        # augmentation parameters
        color=args.color,
        resize_size=args.resize_size,
        # hardware parameters
        accelerator=args.accelerator,
        nodes=args.nodes,
        gpus=args.gpus,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        wandb_logger=wandb_logger
    )
