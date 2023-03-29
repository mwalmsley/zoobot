import logging
import os
from typing import Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

from zoobot.pytorch.estimators import define_model


def train_default_zoobot_from_scratch(    
    # absolutely crucial arguments
    save_dir: str,  # save model here
    schema,  # answer these questions
    # input data - specify *either* catalog (to be split) or the splits themselves
    catalog=None,
    train_catalog=None,
    val_catalog=None,
    test_catalog=None,
    # training time parameters
    epochs=1000,
    patience=8,
    # model hparams
    architecture_name='efficientnet_b0',  # recently changed
    batch_size=128,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    scheduler_params={},
    # data and augmentation parameters
    color=False,
    resize_after_crop=224,
    crop_scale_bounds=(0.7, 0.8),
    crop_ratio_bounds=(0.9, 1.1),
    # hardware parameters
    nodes=1,
    gpus=2,
    num_workers=4,
    prefetch_factor=4,
    mixed_precision=False,
    # checkpointing / logging
    wandb_logger=None,
    checkpoint_file_template=None,
    auto_insert_metric_name=True,
    save_top_k=3,
    extra_callbacks=None,
    # replication parameters
    random_state=42
) -> Tuple[define_model.ZoobotTree, pl.Trainer]:
    """
    Train Zoobot from scratch on a big galaxy catalog.
    Zoobot is a base deep learning model (anything from timm, typically a CNN) plus a dirichlet head.
    Images are augmented using the default transforms (flips, rotations, zooms)
    from `the galaxy-datasets repo <https://github.com/mwalmsley/galaxy-datasets/blob/main/galaxy_datasets/transforms.py>`_.

    Once trained, Zoobot can be finetuned to new data.
    For finetuning, see zoobot/pytorch/training/finetune.py.
    Many pretrained models are already available - see :ref:`datanotes`.

    Args:
        save_dir (str): folder to save training logs and trained model checkpoints
        catalog (pd.DataFrame, optional): Galaxy catalog with columns `id_str` and `file_loc`. Will be automatically split to train and val (no test). Defaults to None. 
        train_catalog (pd.DataFrame, optional): As above, but already split by you for training. Defaults to None.
        val_catalog (pd.DataFrame, optional): As above, for validation. Defaults to None.
        test_catalog (pd.DataFrame, optional): As above, for testing. Defaults to None.
        epochs (int, optional): Max. number of epochs to train for. Defaults to 1000.
        patience (int, optional): Max. number of epochs to wait for any loss improvement before ending training. Defaults to 8.
        architecture_name (str, optional): Architecture to use. Passed to timm. Must be in timm.list_models(). Defaults to 'efficientnet_b0'.
        dropout_rate (float, optional): Randomly drop activations prior to the output layer, with this probability. Defaults to 0.2.
        drop_connect_rate (float, optional): Randomly drop blocks with this probability, for regularisation. For supported timm models only. Defaults to 0.2.
        learning_rate (float, optional): Base learning rate for AdamW. Defaults to 1e-3.
        betas (tuple, optional): Beta args (i.e. momentum) for adamW. Defaults to (0.9, 0.999).
        weight_decay (float, optional): Weight decay arg (i.e. L2 penalty) for AdamW. Defaults to 0.01.
        scheduler_params (dict, optional): Specify a learning rate scheduler. See code. Not recommended with AdamW. Defaults to {}.
        color (bool, optional): Train on RGB images rather than channel-averaged. Defaults to False.
        resize_after_crop (int, optional): Input image size. After all transforms, images will be resized to this size. Defaults to 224.
        crop_scale_bounds (tuple, optional): Off-center crop fraction (<1 means zoom in). Defaults to (0.7, 0.8).
        crop_ratio_bounds (tuple, optional): Aspect ratio of crop above. Defaults to (0.9, 1.1).
        nodes (int, optional): Multi-node training Unlikely to work on your cluster without tinkering. Defaults to 1 (i.e. one node).
        gpus (int, optional): Multi-GPU training. Uses distributed data parallel - essentially, full dataset is split by GPU. See torch docs. Defaults to 2.
        num_workers (int, optional): Processes for loading data. See torch dataloader docs. Should be < num cpus. Defaults to 4.
        prefetch_factor (int, optional): Num. batches to queue in memory per dataloader. See torch dataloader docs. Defaults to 4.
        mixed_precision (bool, optional): Use (mostly) half-precision to halve memory requirements. May cause instability. See Lightning Trainer docs. Defaults to False.
        wandb_logger (pl.loggers.wandb.WandbLogger, optional): Logger to track experiments on Weights and Biases. Defaults to None.
        checkpoint_file_template (str, optional): formatting for checkpoint filename. See Lightning docs. Defaults to None.
        auto_insert_metric_name (bool, optional): escape "/" in metric names when naming checkpoints. See Lightning docs. Defaults to True.
        save_top_k (int, optional): Keep the k best checkpoints. See Lightning docs. Defaults to 3.
        random_state (int, optional): Seed. Defaults to 42.

    Returns:
        Tuple[define_model.ZoobotTree, pl.Trainer]: Trained ZoobotTree model, and Trainer with which it was trained.
    """

    # some optional logging.debug calls recording cluster environment
    slurm_debugging_logs()

    # redundant when already called before this, but no harm doing twice
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

    strategy = 'auto'
    plugins = None
    if (gpus is not None) and (gpus > 1):
        strategy = DDPStrategy(find_unused_parameters=False)  # static_graph=True TODO
        logging.info('Using multi-gpu training')
        if nodes > 1:  # I assume nobody is doing multi-node cpu training...
            logging.info('Using multi-node training')  # purely for your info
            # this is only needed for multi-node training
            # our cluster sets TASKS_PER_NODE not NTASKS_PER_NODE
            # (with srun, SLURM_STEP_TASKS_PER_NODE)
            # https://slurm.schedmd.com/srun.html#OPT_SLURM_STEP_TASKS_PER_NODE
            if 'SLURM_NTASKS_PER_NODE' not in os.environ.keys():
                os.environ['SLURM_NTASKS_PER_NODE'] = os.environ['SLURM_TASKS_PER_NODE']
                # from lightning_lite.plugins.environments import SLURMEnvironment
                from zoobot.pytorch import manchester
                logging.warning('Using custom slurm environment')
                # https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html#enable-auto-wall-time-resubmitions
                plugins = [manchester.ManchesterEnvironment(auto_requeue=False)]

    if gpus > 0:
        accelerator = 'gpu'
        devices = gpus
    else:
        accelerator = 'cpu'
        devices = 'auto'  # all

    
    if mixed_precision:
        logging.info(
            'Training with automatic mixed precision. Will reduce memory footprint but may cause training instability for e.g. resnet')
        precision = '16-mixed'
        torch.set_float32_matmul_precision('medium')
    else:
        precision = '32'
        torch.set_float32_matmul_precision('high')

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
            'test_catalog': test_catalog  # may be None
        }

    if wandb_logger is not None:
        wandb_logger.log_hyperparams({
            'random_state': random_state,
            'epochs': epochs,
            'accelerator': accelerator,
            'gpus': gpus,
            'nodes': nodes,
            'precision': precision,
            'batch_size': batch_size,
            'greyscale': not color,
            'crop_scale_bounds': crop_scale_bounds,
            'crop_ratio_bounds': crop_ratio_bounds,
            'resize_after_crop': resize_after_crop,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'framework': 'pytorch'
        })

    datamodule = GalaxyDataModule(
        label_cols=schema.label_cols,
        # can take either a catalog (and split it), or a pre-split catalog
        **catalogs_to_use,
        # augmentations parameters
        greyscale=not color,
        crop_scale_bounds=crop_scale_bounds,
        crop_ratio_bounds=crop_ratio_bounds,
        resize_after_crop=resize_after_crop,
        # hardware parameters
        batch_size=batch_size, # on 2xA100s, 256 with DDP, 512 with distributed (i.e. split batch)
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    datamodule.setup(stage='fit')

    # these args are automatically logged
    lightning_model = define_model.ZoobotTree(
        output_dim=len(schema.label_cols),
        question_index_groups=schema.question_index_groups,
        architecture_name=architecture_name,
        channels=channels,
        use_imagenet_weights=False,
        test_time_dropout=True,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        timm_kwargs={'drop_path_rate': drop_connect_rate},
        betas=betas,
        weight_decay=weight_decay,
        scheduler_params=scheduler_params
    )
    
    extra_callbacks = extra_callbacks if extra_callbacks else []

    # used later for checkpoint_callback.best_model_path
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(save_dir, 'checkpoints'),
            monitor="validation/epoch_loss",
            save_weights_only=True,
            mode='min',
            # custom filename for checkpointing due to / in metric
            filename=checkpoint_file_template,
            # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint.params.auto_insert_metric_name
            # avoids extra folders from the checkpoint name
            auto_insert_metric_name=auto_insert_metric_name,
            save_top_k=save_top_k
    )

    early_stopping_callback = EarlyStopping(monitor='validation/epoch_loss', patience=patience, check_finite=True)

    callbacks = [checkpoint_callback, early_stopping_callback] + extra_callbacks

    trainer = pl.Trainer(
        log_every_n_steps=150,  # at batch 512 (A100 MP max), DR5 has ~161 train steps
        accelerator=accelerator,
        devices=devices,  # per node
        num_nodes=nodes,
        strategy=strategy,
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=epochs,
        default_root_dir=save_dir,
        plugins=plugins
    )

    logging.info((trainer.strategy, trainer.world_size,
                 trainer.local_rank, trainer.global_rank, trainer.node_rank))

    trainer.fit(lightning_model, datamodule)  # uses batch size of datamodule

    test_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=wandb_logger,
        default_root_dir=save_dir
    )

    best_model_path = trainer.checkpoint_callback.best_model_path

    # can test as per the below, but note that datamodule must have a test dataset attribute as per pytorch lightning docs.
    # also be careful not to test regularly, as this breaks train/val/test conceptual separation and may cause hparam overfitting
    if test_catalog is not None:
        logging.info(f'Testing on {checkpoint_callback.best_model_path} with single GPU. Be careful not to overfit your choices to the test data...')
        test_trainer.validate(
            model=lightning_model,
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
        )
        test_trainer.test(
            model=lightning_model,
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
        )

    # explicitly update the model weights to the best checkpoint before returning
    # (assumes only one checkpoint callback, very likely in practice)
    # additional kwargs are passed to re-init the lighting_model automatically
    # more broadly, this allows for tracking hparams
    # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#initialize-with-other-parameters
    # to make this work, ZoobotLightningModule can only take "normal" parameters (e.g. not custom objects) so has quite a few args
    logging.info('Returning model from checkpoint: {}'.format(best_model_path))
    define_model.ZoobotTree.load_from_checkpoint(best_model_path)  # or .best_model_path, eventually

    return lightning_model, trainer




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
