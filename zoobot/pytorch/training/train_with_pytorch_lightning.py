import logging
import os

from pytorch_lightning.plugins.training_type import DDPPlugin
# https://github.com/PyTorchLightning/pytorch-lightning/blob/1.1.6/pytorch_lightning/plugins/ddp_plugin.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_galaxy_datasets.galaxy_datamodule import GalaxyDataModule

from zoobot.pytorch.training import losses
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.estimators import efficientnet_standard, resnet_torchvision_custom  # also resnet_detectron2_custom, imported below


# convenient API for training Zoobot (aka a base cnn model + dirichlet head) from scratch on a big galaxy catalog using sensible augmentations
# does not do finetuning, does not do more complicated architectures (e.g. custom head), does not support custom losses (uses dirichlet loss)
def train_default_zoobot_from_scratch(
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
    # data and augmentation parameters
    # datamodule_class=GalaxyDataModule,  # generic catalog of galaxies, will not download itself. Can replace with any datamodules from pytorch_galaxy_datasets
    color=False,
    resize_size=224,
    crop_scale_bounds=(0.7, 0.8),
    crop_ratio_bounds=(0.9, 1.1),
    # hardware parameters
    accelerator='auto',
    nodes=1,
    gpus=2,
    num_workers=4,
    prefetch_factor=4,
    mixed_precision=False,
    # replication parameters
    random_state=42,
    wandb_logger=None
):

    slurm_debugging_logs()

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

    strategy = None
    if (gpus is not None) and (gpus > 1):
        # only works as plugins, not strategy
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

    if (gpus is not None) and (num_workers * gpus > os.cpu_count()):
        logging.warning(
            """num_workers * gpu > num cpu.
            You may be spawning more dataloader workers than you have cpus, causing bottlenecks. 
            Suggest reducing num_workers."""
        )
    if num_workers > os.cpu_count():
        logging.warning(
            """num_workers > num cpu.
            You may be spawning more dataloader workers than you have cpus, causing bottlenecks. 
            Suggest reducing num_workers."""
        )

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

    datamodule = GalaxyDataModule(
        label_cols=schema.label_cols,
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

    get_architecture, representation_dim = select_base_architecture_func_from_name(model_architecture)

    model = define_model.get_plain_pytorch_zoobot_model(
        output_dim=len(schema.answers),
        include_top=True,
        channels=channels,
        get_architecture=get_architecture,
        representation_dim=representation_dim
    )

    # This just adds schema.question_index_groups as an arg to the usual (labels, preds) loss arg format
    # Would use lambda but multi-gpu doesn't support as lambda can't be pickled
    def loss_func(preds, labels):  # pytorch convention is preds, labels
        return losses.calculate_multiquestion_loss(labels, preds, schema.question_index_groups)  # my and sklearn convention is labels, preds

    lightning_model = define_model.GenericLightningModule(
        model, loss_func
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
        log_every_n_steps=3,
        accelerator=accelerator,
        gpus=gpus,  # per node
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

    trainer.fit(lightning_model, datamodule)

    trainer.test(
        model=lightning_model,
        datamodule=datamodule,
        ckpt_path='best'  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
    )


def select_base_architecture_func_from_name(base_architecture):
    if base_architecture == 'efficientnet':
        get_architecture = efficientnet_standard.efficientnet_b0
        representation_dim = 1280
    elif base_architecture == 'resnet_detectron':
        # only import if needed, as requires gpu version of pytorch or throws cuda errors e.g.
        # from detectron2 import _C -> ImportError: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
        from zoobot.pytorch.estimators import resnet_detectron2_custom
        get_architecture = resnet_detectron2_custom.get_resnet
        representation_dim = 2048
    elif base_architecture == 'resnet_torchvision':
        get_architecture = resnet_torchvision_custom.get_resnet  # only supports color
        representation_dim = 2048
    else:
        raise ValueError(
            'Model architecture not recognised: got model={}, expected one of [efficientnet, resnet_detectron, resnet_torchvision]'.format(base_architecture))
            
    return get_architecture,representation_dim



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


def slurm_debugging_logs():
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
