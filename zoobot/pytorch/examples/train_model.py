import logging
import os
import argparse
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
# from pytorch_lightning.strategies.ddp import DDPStrategy
# from pytorch_lightning.strategies import DDPStrategy

# from pytorch_lightning.strategies import DDPStrategy  # not sure why not importing?
from pytorch_lightning.plugins.training_type import DDPPlugin
# https://github.com/PyTorchLightning/pytorch-lightning/blob/1.1.6/pytorch_lightning/plugins/ddp_plugin.py
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from yaml import parse


from zoobot.shared import schemas
from zoobot.pytorch.estimators import define_model, resnet_detectron2_custom, efficientnet_standard, resnet_torchvision_custom
from zoobot.pytorch.datasets import decals_dr8
from zoobot.pytorch.training import losses
from zoobot.shared import label_metadata


if __name__ == '__main__':

    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(levelname)s: %(message)s'
    )

    logging.info('Begin training example script')

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--catalog', dest='catalog_loc', type=str)  # expects catalog, not tfrecords
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--shard-img-size', dest='shard_img_size', type=int, default=300)
    parser.add_argument('--resize-size', dest='resize_size', type=int, default=224)
    parser.add_argument('--batch-size', dest='batch_size', default=256, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--test-time-augs', dest='always_augment', default=False, action='store_true',
        help='Zoobot includes keras.preprocessing augmentation layers. \
        These only augment (rotate/flip/etc) at train time by default. \
        They can be enabled at test time as well, which gives better uncertainties (by increasing variance between forward passes) \
        but may be unexpected and mess with e.g. GradCAM techniques.'),
    parser.add_argument('--dropout-rate', dest='dropout_rate', default=0.2, type=float)
    parser.add_argument('--mixed-precision', dest='mixed_precision', default=False, action='store_true',
      help='If true, use automatic mixed precision (via PyTorch Lightning) to reduce GPU memory use (~x2). Else, use full (32 bit) precision')
    args = parser.parse_args()

    "shared setup"

    # a bit awkward, but I think it is better to have to specify you def. want color than that you def want greyscale
    greyscale = not args.color
    if greyscale:
      logging.info('Converting images to greyscale before training')
      channels = 1  # albumentations actually keeps the third dim - need to work out a custom transform to change dim, maybe
    else:
        logging.warning('Training on color images, not converting to greyscale')
        channels = 3

    catalog_loc = args.catalog_loc
    initial_size = args.shard_img_size
    resize_size = args.resize_size  # currently does nothing, hardcoded into decals_dr8.py
    batch_size = args.batch_size
    always_augment = not args.always_augment

    epochs = args.epochs
    save_dir = args.save_dir

    assert save_dir is not None
    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)

    pl.seed_everything(42)

    question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    "shared setup ends"

    # catalog provided
    # catalog = pd.read_csv(catalog_loc)
    # # catalog = pd.read_csv(catalog_loc).sample(1000)  # debugging
    # catalog['file_loc'] = catalog['file_loc'].str.replace('/raid/scratch',  '/share/nas2')
    # logging.info(catalog['file_loc'].iloc[0])

    # num_workers = int(os.cpu_count()/args.gpus)  # if ddp mode, each gpu has own dataloaders, if 1 gpu, all cpus
    # logging.info('num workers: {}'.format(num_workers))
    # datamodule = decals_dr8.DECALSDR8DataModule(
    #   schema=schema,
    #   catalog=catalog,
    #   greyscale=greyscale,
    #   batch_size=batch_size,  # 256 with DDP, 512 with distributed (i.e. split batch)
    #   num_workers=num_workers
    # )

    # or, explicit splits provided
    train_catalog_locs = [
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr12/train_shards/train_df.csv',
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/train_shards/train_df.csv',
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr8/train_shards/train_df.csv'
    ]
    val_catalog_locs = [
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr12/val_shards/val_df.csv',
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/val_shards/val_df.csv',
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr8/val_shards/val_df.csv'
    ]
    test_catalog_locs = [
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr12/test_shards/test_df.csv',
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr5/test_shards/test_df.csv',
      '/share/nas2/walml/repos/gz-decals-classifiers/data/decals/shards/all_campaigns_ortho_v2/dr8/test_shards/test_df.csv'
    ]
    answer_columns = [a.text for a in schema.answers]
    useful_columns = answer_columns + ['file_loc']

    train_catalog = pd.concat([pd.read_csv(loc, usecols=useful_columns) for loc in train_catalog_locs])
    val_catalog = pd.concat([pd.read_csv(loc, usecols=useful_columns) for loc in val_catalog_locs])
    test_catalog = pd.concat([pd.read_csv(loc, usecols=useful_columns) for loc in test_catalog_locs])
    for catalog in (train_catalog, val_catalog, test_catalog):

      # tweak file paths
      catalog['file_loc'] = catalog['file_loc'].str.replace('/raid/scratch',  '/share/nas2')
      catalog['file_loc'] = catalog['file_loc'].str.replace('/dr8_downloader/',  '/dr8/')
      # catalog['file_loc'] = catalog['file_loc'].str.replace('.jpeg', '.png')
      catalog['file_loc'] = catalog['file_loc'].str.replace(r'/png/', r'/jpeg/')
      catalog['file_loc'] = catalog['file_loc'].str.replace('.png', '.jpeg')
      # catalog['file_loc'] = catalog['file_loc'].str.replace('/share/nas2', '/state/partition1')  # load local copy

      # enforce datatypes
      for answer_col in answer_columns:
        catalog[answer_col] = catalog[answer_col].astype(int)
        catalog['file_loc'] = catalog['file_loc'].astype(str)

      logging.info(catalog['file_loc'].iloc[0])

    # # debug mode
    # train_catalog = train_catalog.sample(5000).reset_index(drop=True)
    # val_catalog = val_catalog.sample(5000).reset_index(drop=True)
    # test_catalog = test_catalog.sample(5000).reset_index(drop=True)


    num_workers = int((os.cpu_count() - 2)/args.gpus)  # if ddp mode, each gpu has own dataloaders, if 1 gpu, all cpus. Save 2 cpu per gpu just to have some breathing room.
    assert num_workers > 0
    # num_workers = 1

    prefetch_factor = 4
    # prefetch_factor = max(1, int(20000 / (num_workers * batch_size * args.gpus)))  # may need to tweak this if your dataloaders timeout

    datamodule = decals_dr8.DECALSDR8DataModule(
      schema=schema,
      album=False,
      train_catalog=train_catalog,
      val_catalog=val_catalog,
      test_catalog=test_catalog,
      greyscale=greyscale,
      use_memory=False,
      batch_size=batch_size,  # 256 with DDP, 512 with distributed (i.e. split batch)
      num_workers=num_workers,
      prefetch_factor=prefetch_factor
    )
    datamodule.setup()

    if args.wandb:
        wandb_logger = WandbLogger(
          project='zoobot-pytorch-dr8',
          # project='zoobot-pytorch',
          name=os.path.basename(save_dir),
          log_model="all")
        # only rank 0 process gets access to the wandb.run object, and for non-zero rank processes: wandb.run = None
        # https://docs.wandb.ai/guides/integrations/lightning#how-to-use-multiple-gpus-with-lightning-and-w-and-b
    else:
      wandb_logger = None

    # # you can do this to see images, but if you do, wandb will cause training to silently hang before starting
    # if wandb_logger is not None:
    #   for (dataloader_name, dataloader) in [('train', datamodule.train_dataloader()), ('val', datamodule.val_dataloader()), ('test', datamodule.test_dataloader())]:
    #     for images, labels in dataloader:
    #       logging.info(images.shape)
    #       images_np = np.transpose(images[:5].numpy(), axes=[0, 2, 3, 1])  # BCHW to BHWC
    #       # images_np = images.numpy()
    #       logging.info((dataloader_name, images_np.shape, images[0].min(), images[0].max()))
    #       wandb_logger.log_image(key="example_{}_images".format(dataloader_name), images=[im for im in images_np[:5]]) 
    #       break  # only inner loop aka don't log the whole dataloader

    loss_func = losses.calculate_multiquestion_loss

    model = define_model.ZoobotModel(
      schema=schema,
      loss=loss_func,
      channels=channels,
      # you can use efficientnet...
      get_architecture=efficientnet_standard.efficientnet_b0,
      representation_dim=1280
      # or resnet via detectron2 definition...
      # get_architecture=resnet_detectron2_custom.get_resnet,
      # representation_dim=2048
      # or resnet via torchvision definition...
      # get_architecture=resnet_torchvision_custom.get_resnet,  # only supports color
      # representation_dim=2048
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, 'checkpoints'),
            monitor="val_loss",
            save_weights_only=True,
            mode='min',
            save_top_k=3
        ),
        EarlyStopping(monitor='val_loss', patience=8, check_finite=True)
    ]

    
    # https://hpcc.umd.edu/hpcc/help/slurmenv.html
    # logging.info(os.environ)
    logging.info(os.getenv("SLURM_JOB_ID", 'No SLURM_JOB_ID'))
    logging.info(os.getenv("SLURM_JOB_NAME", 'No SLURM_JOB_NAME'))
    logging.info(os.getenv("SLURM_NTASKS", 'No SLURM_NTASKS'))
  # https://github.com/PyTorchLightning/pytorch-lightning/blob/d5fa02e7985c3920e72e268ece1366a1de96281b/pytorch_lightning/trainer/connectors/slurm_connector.py#L29
    # disable slurm detection by pl
    # this is not necessary for single machine, but might be for multi-node
    # may help stop tasks getting left on gpu after slurm exit?
    # del os.environ["SLURM_NTASKS"]  # only exists if --ntasks specified

    logging.info(os.getenv("NODE_RANK", 'No NODE_RANK'))
    logging.info(os.getenv("LOCAL_RANK", 'No LOCAL_RANK'))
    logging.info(os.getenv("WORLD_SIZE", 'No WORLD_SIZE'))

    strategy = None
    if args.gpus > 1:
      # plugins = [DDPPlugin(find_unused_parameters=False)],  # only works as plugins, not strategy
      # strategy = 'ddp'
      strategy = DDPPlugin(find_unused_parameters=False)
      logging.info('Using multi-gpu training')
  
    if args.nodes > 1:
      assert args.gpus == 2
      logging.info('Using multi-node training')
      # this hangs silently on Manchester's slurm cluster - perhaps you will have more success?
  
    precision = None
    if args.mixed_precision:
      logging.info('Training with automatic mixed precision. Will reduce memory footprint but may cause training instability for e.g. resnet')
      precision=16

    trainer = pl.Trainer(
        accelerator="gpu", gpus=args.gpus,  # per node
        num_nodes=args.nodes,
        strategy=strategy,
        precision=precision,
        logger = wandb_logger,
        callbacks=callbacks,
        max_epochs=epochs,
        default_root_dir=save_dir
    )

    logging.info((trainer.training_type_plugin, trainer.world_size, trainer.local_rank, trainer.global_rank, trainer.node_rank))

    trainer.fit(model, datamodule)

    trainer.test(
      model=model,
      datamodule=datamodule,
      ckpt_path='best'  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
    )
