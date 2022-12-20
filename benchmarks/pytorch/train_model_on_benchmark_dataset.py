import logging
import os
import argparse

from pytorch_lightning.loggers import WandbLogger
import wandb

from zoobot.pytorch.training import train_with_pytorch_lightning
from zoobot.shared import benchmark_datasets


if __name__ == '__main__':

    """
    Used to create the PyTorch pretrained weights checkpoints
    See .sh file of the same name for args used.

    See zoobot/pytorch/examples/minimal_examples.py for a friendlier example
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--data-dir', dest='data_dir', type=str)
    parser.add_argument('--dataset', dest='dataset', type=str, help='dataset to use, either "gz_decals_dr5" or "gz_evo"')
    parser.add_argument('--architecture', dest='architecture_name', default='efficientnet', type=str)
    parser.add_argument('--resize-after-crop', dest='resize_after_crop',
                        type=int, default=224)  # 380 from sweep
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--batch-size', dest='batch_size',
                        default=256, type=int)  # 64 from sweep
    parser.add_argument('--gpus', dest='gpus', default=1, type=int)
    parser.add_argument('--nodes', dest='nodes', default=1, type=int)
    parser.add_argument('--mixed-precision', dest='mixed_precision',
                        default=False, action='store_true')
    parser.add_argument('--debug', dest='debug',
                        default=False, action='store_true')
    parser.add_argument('--wandb', dest='wandb',
                        default=False, action='store_true')
    parser.add_argument('--seed', dest='random_state', default=42, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    random_state = args.random_state

    if args.nodes > 1:
        # at Manchester, our slurm cluster sets TASKS not NTASKS, which then confuses lightning
        if 'SLURM_NTASKS_PER_NODE' not in os.environ.keys():
            os.environ['SLURM_NTASKS_PER_NODE'] = os.environ['SLURM_TASKS_PER_NODE']
        # log the rest to help debug
        logging.info([(x, y) for (x, y) in os.environ.items() if 'SLURM' in x])

    if args.debug:
        download = False
    else:
        # download = True  # for first use
        download = False  # for speed afterwards


    if args.dataset == 'gz_decals_dr5':
        schema, (train_catalog, val_catalog, test_catalog) = benchmark_datasets.get_gz_decals_dr5_benchmark_dataset(args.data_dir, random_state, download=download)
    elif args.dataset == 'gz_evo':
        schema, (train_catalog, val_catalog, test_catalog) = benchmark_datasets.get_gz_evo_benchmark_dataset(args.data_dir, random_state, download=download)
    else:
        raise ValueError(f'Dataset {args.dataset} not recognised: should be "gz_decals_dr5" or "gz_evo"')


    logging.info('First val galaxy: {}'.format(val_catalog.iloc[0]['id_str']))

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
            project=f'zoobot-benchmarks-{args.dataset}',
            name=os.path.basename(args.save_dir),
            log_model=False
        )
        wandb_logger.log_text(key="train_catalog", dataframe=train_catalog.sample(5))
        wandb_logger.log_text(key="val_catalog", dataframe=train_catalog.sample(5))
        wandb_logger.log_text(key="test_catalog", dataframe=train_catalog.sample(5))
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
        patience=10,
        # augmentation parameters
        color=args.color,
        resize_after_crop=args.resize_after_crop,
        # hardware parameters
        gpus=args.gpus,
        nodes=args.nodes,
        mixed_precision=args.mixed_precision,
        wandb_logger=wandb_logger,
        prefetch_factor=4,
        num_workers=11,  # system has 24 cpu, 12 cpu per gpu, leave a little wiggle room
        random_state=random_state,
        learning_rate=3e-4,  # new default, from sweep
    )

    wandb.finish()