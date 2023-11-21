import logging
import os
import argparse
import glob

from pytorch_lightning.loggers import WandbLogger
import wandb

from zoobot.pytorch.training import train_with_pytorch_lightning
from zoobot.shared import benchmark_datasets, schemas

import pytorch_lightning as pl


if __name__ == '__main__':

    """
    Used to create the PyTorch pretrained weights checkpoints
    See .sh file of the same name for args used.

    See zoobot/pytorch/examples/minimal_examples.py for a friendlier example
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    # parser.add_argument('--data-dir', dest='data_dir', type=str)
    # parser.add_argument('--dataset', dest='dataset', type=str, help='dataset to use, either "gz_decals_dr5" or "gz_evo"')
    parser.add_argument('--architecture', dest='architecture_name', default='efficientnet_b0', type=str)
    parser.add_argument('--resize-after-crop', dest='resize_after_crop',
                        type=int, default=224)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--batch-size', dest='batch_size',
                        default=256, type=int)
    parser.add_argument('--gpus', dest='gpus', default=1, type=int)
    parser.add_argument('--nodes', dest='nodes', default=1, type=int)
    parser.add_argument('--num-workers', dest='num_workers', default=1, type=int)
    parser.add_argument('--mixed-precision', dest='mixed_precision',
                        default=False, action='store_true')
    parser.add_argument('--debug', dest='debug',
                        default=False, action='store_true')
    parser.add_argument('--wandb', dest='wandb',
                        default=False, action='store_true')
    parser.add_argument('--seed', dest='random_state', default=1, type=int)
    args = parser.parse_args()

    """
    debug
    python only_for_me/narval/train.py --save-dir only_for_me/narval/debug_models --batch-size 32 --color
    """

    logging.basicConfig(level=logging.INFO)

    random_state = args.random_state
    pl.seed_everything(random_state)

    # if args.nodes > 1:
    #     # at Manchester, our slurm cluster sets TASKS not NTASKS, which then confuses lightning
    #     if 'SLURM_NTASKS_PER_NODE' not in os.environ.keys():
    #         os.environ['SLURM_NTASKS_PER_NODE'] = os.environ['SLURM_TASKS_PER_NODE']
    #     # log the rest to help debug
    #     logging.info([(x, y) for (x, y) in os.environ.items() if 'SLURM' in x])

    if os.path.isdir('/home/walml/repos/zoobot'):
        search_str = '/home/walml/repos/zoobot/gz_decals_5_train_*.tar'

    else:
        search_str = '/home/walml/projects/def-bovy/walml/data/webdatasets/desi_labelled/desi_labelled_train_*.tar'

    all_urls = glob.glob(search_str)
    assert len(all_urls) > 0, search_str
    # train_urls, val_urls = all_urls[:70], all_urls[70:]
    train_urls, val_urls = all_urls[:60], all_urls[60:70]
    schema = schemas.decals_all_campaigns_ortho_schema

    # debug mode
    if args.debug:
        logging.warning(
            'Using debug mode: cutting urls down to 2')
        train_urls = train_urls[:2]
        val_urls = val_urls[:2]
        epochs = 1
    else:
        epochs = 1000

    if args.wandb:
        wandb_logger = WandbLogger(
            project='narval',
            # name=os.path.basename(args.save_dir),
            log_model=False
        )
    else:
        wandb_logger = None

    train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        save_dir=args.save_dir,
        schema=schema,
        train_urls = train_urls,
        val_urls = val_urls,
        test_urls = None,
        architecture_name=args.architecture_name,
        batch_size=args.batch_size,
        epochs=epochs,  # rely on early stopping
        patience=10,
        # augmentation parameters
        # color=args.color,
        color=args.color,
        resize_after_crop=args.resize_after_crop,
        # hardware parameters
        gpus=args.gpus,
        nodes=args.nodes,
        mixed_precision=args.mixed_precision,
        wandb_logger=wandb_logger,
        prefetch_factor=1, # TODO
        num_workers=args.num_workers,
        compile_model=True,  # NEW
        random_state=random_state,
        learning_rate=1e-3,
        cache_dir=os.environ['SLURM_TMPDIR'] + '/cache'
        # cache_dir='/tmp/cache'
        # /tmp for ramdisk (400GB total, vs 4TB total for nvme)
    )

    # https://discuss.pytorch.org/t/torch-dynamo-hit-config-cache-size-limit-64/183886
    # https://pytorch.org/docs/stable/torch.compiler_faq.html#why-is-compilation-slow

    wandb.finish()