import logging
import os
from typing import Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import TorchSyncBatchNorm
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from galaxy_datasets.pytorch.webdatamodule import WebDataModule

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
    train_urls=None,
    val_urls=None,
    test_urls=None,
    cache_dir=None,  # only works with webdataset urls
    # training time parameters
    epochs=1000,
    patience=8,
    # model hparams
    architecture_name='efficientnet_b0',
    timm_kwargs = {}, # e.g. {'drop_path_rate': 0.2, 'num_features': 1280}. Passed to timm model init method, depends on arch.
    batch_size=128,
    dropout_rate=0.2,
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
    accumulate_gradients=1,
    sync_batchnorm=False,
    num_workers=4,
    prefetch_factor=4,
    mixed_precision=False,
    compile_encoder=False,
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

    **You don't need to use this**. 
    Training from scratch is becoming increasingly complicated (as you can see from the arguments) due to ongoing research on the best methods.
    This will be refactored to a dedicated "foundation" repo.

    Args:
        save_dir (str): folder to save training logs and trained model checkpoints
        schema (shared.schemas.Schema): Schema object with label_cols, question_answer_pairs, and dependencies
        catalog (pd.DataFrame, optional): Galaxy catalog with columns `id_str` and `file_loc`. Will be automatically split to train and val (no test). Defaults to None. 
        train_catalog (pd.DataFrame, optional): As above, but already split by you for training. Defaults to None.
        val_catalog (pd.DataFrame, optional): As above, for validation. Defaults to None.
        test_catalog (pd.DataFrame, optional): As above, for testing. Defaults to None.
        train_urls (list, optional): List of URLs to webdatasets for training. Defaults to None.
        val_urls (list, optional): List of URLs to webdatasets for validation. Defaults to None.
        test_urls (list, optional): List of URLs to webdatasets for testing. Defaults to None.
        cache_dir (str, optional): Directory to cache webdatasets. Defaults to None.
        epochs (int, optional): Max. number of epochs to train for. Defaults to 1000.
        patience (int, optional): Max. number of epochs to wait for any loss improvement before ending training. Defaults to 8.
        architecture_name (str, optional): Architecture to use. Passed to timm. Must be in timm.list_models(). Defaults to 'efficientnet_b0'.
        timm_kwargs (dict, optional): Additional kwargs to pass to timm model init method, for example {'drop_connect_rate': 0.2}. Defaults to {}.
        batch_size (int, optional): Batch size. Defaults to 128.
        dropout_rate (float, optional): Randomly drop activations prior to the output layer, with this probability. Defaults to 0.2.
        learning_rate (float, optional): Base learning rate for AdamW. Defaults to 1e-3.
        betas (tuple, optional): Beta args (i.e. momentum) for adamW. Defaults to (0.9, 0.999).
        weight_decay (float, optional): Weight decay arg (i.e. L2 penalty) for AdamW. Defaults to 0.01.
        scheduler_params (dict, optional): Specify a learning rate scheduler. See code below. Defaults to {}.
        color (bool, optional): Train on RGB images rather than channel-averaged. Defaults to False.
        resize_after_crop (int, optional): Input image size. After all transforms, images will be resized to this size. Defaults to 224.
        crop_scale_bounds (tuple, optional): Off-center crop fraction (<1 means zoom in). Defaults to (0.7, 0.8).
        crop_ratio_bounds (tuple, optional): Aspect ratio of crop above. Defaults to (0.9, 1.1).
        nodes (int, optional): Multi-node training Unlikely to work on your cluster without tinkering. Defaults to 1 (i.e. one node).
        gpus (int, optional): Multi-GPU training. Uses distributed data parallel - essentially, full dataset is split by GPU. See torch docs. Defaults to 2.
        sync_batchnorm (bool, optional): Use synchronized batchnorm. Defaults to False.
        num_workers (int, optional): Processes for loading data. See torch dataloader docs. Should be < num cpus. Defaults to 4.
        prefetch_factor (int, optional): Num. batches to queue in memory per dataloader. See torch dataloader docs. Defaults to 4.
        mixed_precision (bool, optional): Use (mostly) half-precision to halve memory requirements. May cause instability. See Lightning Trainer docs. Defaults to False.
        compile_encoder (bool, optional): Compile the encoder with torch.compile (new in torch v2). Defaults to False.
        wandb_logger (pl.loggers.wandb.WandbLogger, optional): Logger to track experiments on Weights and Biases. Defaults to None.
        checkpoint_file_template (str, optional): formatting for checkpoint filename. See Lightning docs. Defaults to None.
        auto_insert_metric_name (bool, optional): escape "/" in metric names when naming checkpoints. See Lightning docs. Defaults to True.
        save_top_k (int, optional): Keep the k best checkpoints. See Lightning docs. Defaults to 3.
        extra_callbacks (list, optional): Additional callbacks to pass to the Trainer. Defaults to None.
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
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass # another gpu process may have just made it
    logging.info(f'Saving to {save_dir}')

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
        # if nodes > 1:  # I assume nobody is doing multi-node cpu training...
            # logging.info('Using multi-node training')  # purely for your info
            # this is only needed for multi-node training
            # our cluster sets TASKS_PER_NODE not NTASKS_PER_NODE
            # (with srun, SLURM_STEP_TASKS_PER_NODE)
            # https://slurm.schedmd.com/srun.html#OPT_SLURM_STEP_TASKS_PER_NODE
        if 'SLURM_NTASKS_PER_NODE' not in os.environ.keys():
            os.environ['SLURM_NTASKS_PER_NODE'] = os.environ['SLURM_TASKS_PER_NODE']
            from zoobot.pytorch import manchester
            logging.warning(f'Using custom slurm environment, --n-tasks-per-node={os.environ["SLURM_NTASKS_PER_NODE"]}')
            # https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html#enable-auto-wall-time-resubmitions
            plugins = [manchester.GalahadEnvironment(auto_requeue=False)]

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
    else:
        logging.warning('No wandb_logger passed. Using CSV logging only')
        wandb_logger = CSVLogger(save_dir=save_dir)

    # work out what dataset the user has passed
    single_catalog = catalog is not None
    split_catalogs = train_catalog is not None
    webdatasets = train_urls is not None

    if single_catalog or split_catalogs:
        # this branch will use GalaxyDataModule to load catalogs
        assert not webdatasets
        if single_catalog:
            assert not split_catalogs
            data_to_use = {
                'catalog': catalog
            }
        else:
            data_to_use = {
                'train_catalog': train_catalog,
                'val_catalog': val_catalog,
                'test_catalog': test_catalog  # may be None
            }
        assert crop_scale_bounds[1] < 1  # zoom in for albumentations
        datamodule = GalaxyDataModule(
            label_cols=schema.label_cols,
            # can take either a catalog (and split it), or a pre-split catalog
            **data_to_use,
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
    else:
        # this branch will use WebDataModule to load premade webdatasets

        # temporary: use SSL-like transform
        # ADDED BACK FOR EUCLID
        from foundation.models import transforms

        train_transform_cfg = transforms.default_view_config()
        train_transform_cfg.greyscale = not color
        assert crop_scale_bounds[1] > 1
        train_transform_cfg.random_affine['scale'] = crop_scale_bounds  # no, just use 1.2-1.4 default
        # train_transform_cfg.random_affine['scale'] = (1.1, 1.2)
        train_transform_cfg.random_affine['shear'] = None  # disable
        train_transform_cfg.random_affine['translate'] = None  # disable
        train_transform_cfg.erase_iterations = 0  # disable

        # train_transform_cfg = transforms.minimal_view_config()
        
        inference_transform_cfg = transforms.minimal_view_config()
        inference_transform_cfg.greyscale = not color

        train_transform_cfg.output_size = resize_after_crop
        inference_transform_cfg.output_size = resize_after_crop

        datamodule = WebDataModule(
            train_urls=train_urls,
            val_urls=val_urls,
            test_urls=test_urls,
            predict_urls=test_urls,  # will make test predictions
            label_cols=schema.label_cols,
            # hardware
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            cache_dir=cache_dir,
            # augmentation args
            greyscale=not color,
            crop_scale_bounds=crop_scale_bounds,
            crop_ratio_bounds=crop_ratio_bounds,
            resize_after_crop=resize_after_crop,
            # temporary: use SSL-like transform
            train_transform=transforms.GalaxyViewTransform(train_transform_cfg),
            inference_transform=transforms.GalaxyViewTransform(inference_transform_cfg),
        )

    # debug - check range of loaded images, should be 0-1
    datamodule.setup(stage='fit')
    for (images, _) in datamodule.train_dataloader():
        logging.info(f'Using batches of {images.shape[0]} images for training')
        logging.info('First batch image min/max: {}/{}'.format(images.min(), images.max()))
        assert images.max() <= 1.0001
        assert images.min() >= -0.0001
        break
    # exit()

    # these args are automatically logged
    lightning_model = define_model.ZoobotTree(
        output_dim=len(schema.label_cols),
        # NEW - pass these from schema, for better logging
        question_answer_pairs=schema.question_answer_pairs,
        dependencies=schema.dependencies,
        architecture_name=architecture_name,
        channels=channels,
        test_time_dropout=True,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py#L75C9-L75C17
        timm_kwargs=timm_kwargs,
        compile_encoder=compile_encoder,
        betas=betas,
        weight_decay=weight_decay,
        scheduler_params=scheduler_params
    )

    if sync_batchnorm:
        logging.info('Using sync batchnorm')
        lightning_model = TorchSyncBatchNorm().apply(lightning_model)
    
    

    # used later for checkpoint_callback.best_model_path
    checkpoint_callback, callbacks = get_default_callbacks(save_dir, patience, checkpoint_file_template, auto_insert_metric_name, save_top_k)
    if extra_callbacks:
        callbacks += extra_callbacks 

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        log_every_n_steps=150,
        accelerator=accelerator,
        devices=devices,  # per node
        accumulate_grad_batches=accumulate_gradients,
        num_nodes=nodes,
        strategy=strategy,
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=epochs,
        default_root_dir=save_dir,
        plugins=plugins,
        gradient_clip_val=.3  # reduced from 1 to .3, having some nan issues
    )

    trainer.fit(lightning_model, datamodule)  # uses batch size of datamodule

    best_model_path = trainer.checkpoint_callback.best_model_path

    # can test as per the below, but note that datamodule must have a test dataset attribute as per pytorch lightning docs.
    # also be careful not to test regularly, as this breaks train/val/test conceptual separation and may cause hparam overfitting
    if datamodule.test_dataloader is not None:
        logging.info(f'Testing on {checkpoint_callback.best_model_path} with single GPU. Be careful not to overfit your choices to the test data...')
        datamodule.setup(stage='test')
        # TODO with webdataset, no need for new trainer/datamodule (actually it breaks), but might still be needed with normal dataset?
        trainer.test(
            model=lightning_model,
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
        )

        # TODO this will ONLY work with webdatasets
        if isinstance('datamodule', WebDataModule):
            predictions = trainer.predict(
                model=lightning_model,
                datamodule=datamodule,
                ckpt_path=checkpoint_callback.best_model_path
            )  # list of batches, each shaped like [batch_size, model_head]
            predictions = torch.concatenate(predictions, dim=-1).numpy()
            logging.info(predictions.shape)

            datamodule.label_cols = ['id_str']  # triggers webdataset to return only id_str
            datamodule.setup(stage='predict')
            id_strs = [id_str[0] for batch in datamodule.predict_dataloader() for id_str in batch]  # [0] because each id_str is within a vector of length 1
            # logging.info(id_strs[0])
            # logging.info(len(id_strs))

            from zoobot.shared import save_predictions
            save_predictions.predictions_to_csv(predictions, id_strs, schema.label_cols, save_loc=save_dir + '/test_predictions.csv')

        

    # explicitly update the model weights to the best checkpoint before returning
    # (assumes only one checkpoint callback, very likely in practice)
    # additional kwargs are passed to re-init the lighting_model automatically
    # more broadly, this allows for tracking hparams
    # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#initialize-with-other-parameters
    # to make this work, ZoobotLightningModule can only take "normal" parameters (e.g. not custom objects) so has quite a few args
    logging.info('Returning model from checkpoint: {}'.format(best_model_path))
    define_model.ZoobotTree.load_from_checkpoint(best_model_path)  # or .best_model_path, eventually

    return lightning_model, trainer

def get_default_callbacks(save_dir, patience=8, checkpoint_file_template=None, auto_insert_metric_name=True, save_top_k=3):
    
    monitor_metric = 'validation/supervised_loss'
    
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(save_dir, 'checkpoints'),
            monitor=monitor_metric,
            save_weights_only=True,
            mode='min',
            # custom filename for checkpointing due to / in metric
            filename=checkpoint_file_template,
            # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint.params.auto_insert_metric_name
            # avoids extra folders from the checkpoint name
            auto_insert_metric_name=auto_insert_metric_name,
            save_top_k=save_top_k
    )

    early_stopping_callback = EarlyStopping(monitor=monitor_metric, patience=patience, check_finite=True)
    callbacks = [checkpoint_callback, early_stopping_callback]
    return checkpoint_callback,callbacks




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
