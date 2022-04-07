from cgi import test
import logging
import os

# from pytorch_lightning.strategies.ddp import DDPStrategy
# from pytorch_lightning.strategies import DDPStrategy  # not sure why not importing?
from pytorch_lightning.plugins.training_type import DDPPlugin
# https://github.com/PyTorchLightning/pytorch-lightning/blob/1.1.6/pytorch_lightning/plugins/ddp_plugin.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from zoobot.pytorch.datasets import decals_dr8
from zoobot.pytorch.training import losses
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.estimators import resnet_detectron2_custom, efficientnet_standard, resnet_torchvision_custom


def train(
    # absolutely crucial arguments
    save_dir,  # save model here
    schema,  # answer these questions
    # input data - specify *either* catalog (to be split) or the splits themselves
    catalog=None,
    train_catalog=None,
    val_catalog=None,
    test_catalog=None,
    # model training parameters
    model_architecture='efficientnet',
    batch_size=256,
    epochs=1000,
    patience=8,
    # augmentation parameters
    color=False,
    resize_size=224,
    crop_scale_bounds=(0.7, 0.8),
    crop_ratio_bounds=(0.9, 1.1),
    # hardware parameters
    nodes=1,
    gpus=2,
    num_workers=4,
    prefetch_factor=4,
    mixed_precision=False,
    # replication parameters
    random_state=42,
    wandb_logger=None
):

    # https://hpcc.umd.edu/hpcc/help/slurmenv.html
    # logging.info(os.environ)
    logging.debug(os.getenv("SLURM_JOB_ID", 'No SLURM_JOB_ID'))
    logging.debug(os.getenv("SLURM_JOB_NAME", 'No SLURM_JOB_NAME'))
    logging.debug(os.getenv("SLURM_NTASKS", 'No SLURM_NTASKS'))
  # https://github.com/PyTorchLightning/pytorch-lightning/blob/d5fa02e7985c3920e72e268ece1366a1de96281b/pytorch_lightning/trainer/connectors/slurm_connector.py#L29
    # disable slurm detection by pl
    # this is not necessary for single machine, but might be for multi-node
    # may help stop tasks getting left on gpu after slurm exit?
    # del os.environ["SLURM_NTASKS"]  # only exists if --ntasks specified

    logging.debug(os.getenv("NODE_RANK", 'No NODE_RANK'))
    logging.debug(os.getenv("LOCAL_RANK", 'No LOCAL_RANK'))
    logging.debug(os.getenv("WORLD_SIZE", 'No WORLD_SIZE'))

    pl.seed_everything(random_state)

    assert save_dir is not None
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if color:
        logging.warning(
            'Training on color images, not converting to greyscale')
        channels = 3
    else:
        logging.info('Converting images to greyscale before training')
        channels = 1

    if model_architecture == 'efficientnet':
        get_architecture = efficientnet_standard.efficientnet_b0
        representation_dim = 1280
    elif model_architecture == 'resnet_detectron':
        get_architecture = resnet_detectron2_custom.get_resnet
        representation_dim = 2048
    elif model_architecture == 'resnet_torchvision':
        get_architecture = resnet_torchvision_custom.get_resnet  # only supports color
        representation_dim = 2048
    else:
        raise ValueError(
            'Model architecture not recognised: got model={}, expected one of [efficientnet, resnet_detectron, resnet_torchvision]'.format(model_architecture))

    strategy = None
    if gpus > 1:
        # plugins = [DDPPlugin(find_unused_parameters=False)],  # only works as plugins, not strategy
        # strategy = 'ddp'
        strategy = DDPPlugin(find_unused_parameters=False)
        logging.info('Using multi-gpu training')

    if nodes > 1:
        assert gpus == 2
        logging.info('Using multi-node training')
        # this hangs silently on Manchester's slurm cluster - perhaps you will have more success?

    precision = 32
    if mixed_precision:
        logging.info(
            'Training with automatic mixed precision. Will reduce memory footprint but may cause training instability for e.g. resnet')
        precision = 16

    assert num_workers > 0

    if catalog is not None:
        assert train_catalog is None
        assert val_catalog is None
        assert test_catalog is None
        catalogs_to_use = {
            'catalog': catalog
        }
    else:
        assert catalog is None
        catalogs_to_use = {
            'train_catalog': train_catalog,
            'val_catalog': val_catalog,
            'test_catalog': test_catalog
        }

    datamodule = decals_dr8.DECALSDR8DataModule(
        schema=schema,
        # can take either a catalog (and split it), or a pre-split catalog
        **catalogs_to_use,
        #   augmentations parameters
        album=False,
        greyscale=not color,
        resize_size=resize_size,
        crop_scale_bounds=crop_scale_bounds,
        crop_ratio_bounds=crop_ratio_bounds,
        #   hardware parameters
        batch_size=batch_size, # on 2xA100s, 256 with DDP, 512 with distributed (i.e. split batch)
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    datamodule.setup()

    loss_func = losses.calculate_multiquestion_loss

    model = define_model.ZoobotModel(
        schema=schema,
        loss=loss_func,
        channels=channels,
        get_architecture=get_architecture,
        representation_dim=representation_dim
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, 'checkpoints'),
            monitor="val_loss",
            save_weights_only=True,
            mode='min',
            save_top_k=3
        ),
        EarlyStopping(monitor='val_loss', patience=patience, check_finite=True)
    ]

    trainer = pl.Trainer(
        accelerator="gpu", gpus=gpus,  # per node
        num_nodes=nodes,
        strategy=strategy,
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=epochs,
        default_root_dir=save_dir
    )

    logging.info((trainer.training_type_plugin, trainer.world_size,
                 trainer.local_rank, trainer.global_rank, trainer.node_rank))

    trainer.fit(model, datamodule)

    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path='best'  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
    )




    # # you can do this to see images, but if you do, wandb will cause training to silently hang before starting if you do this on multi-GPU runs
    # TODO refactor into datamodule setup hook that's only called on main process
    # if wandb_logger is not None:
    #   for (dataloader_name, dataloader) in [('train', datamodule.train_dataloader()), ('val', datamodule.val_dataloader()), ('test', datamodule.test_dataloader())]:
    #     for images, labels in dataloader:
    #       logging.info(images.shape)
    #       images_np = np.transpose(images[:5].numpy(), axes=[0, 2, 3, 1])  # BCHW to BHWC
    #       # images_np = images.numpy()
    #       logging.info((dataloader_name, images_np.shape, images[0].min(), images[0].max()))
    #       wandb_logger.log_image(key="example_{}_images".format(dataloader_name), images=[im for im in images_np[:5]])
    #       break  # only inner loop aka don't log the whole dataloader

  