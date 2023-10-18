import logging
import os
import shutil

from pytorch_lightning.loggers import WandbLogger

from zoobot.pytorch.training import finetune
from galaxy_datasets import galaxy_mnist
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logging.info('Begin')

    logging.info(os.environ['SLURM_TMPDIR'])

    # import glob
    # logging.info(glob.glob(os.path.join(os.environ['SLURM_TMPDIR'], 'walml/finetune/data')))
    # logging.info(glob.glob(os.path.join(os.environ['SLURM_TMPDIR'], 'walml/finetune/data/galaxy_mnist')))

    import torch
    torch.set_float32_matmul_precision('medium')
    assert torch.cuda.is_available()

    batch_size = 256
    num_workers= 4
    n_blocks = 1  # EffnetB0 is divided into 7 blocks. set 0 to only fit the head weights. Set 1, 2, etc to finetune deeper. 
    max_epochs = 6  #  6 epochs should get you ~93% accuracy. Set much higher (e.g. 1000) for harder problems, to use Zoobot's default early stopping. \

    train_catalog, _ = galaxy_mnist(root=os.path.join(os.environ['SLURM_TMPDIR'], 'walml/finetune/data/galaxy_mnist'), download=False, train=True)
    test_catalog, _ = galaxy_mnist(root=os.path.join(os.environ['SLURM_TMPDIR'], 'walml/finetune/data/galaxy_mnist'), download=False, train=False)
    logging.info('Data ready')

    label_cols = ['label']
    num_classes = 4
  
    # load a pretrained checkpoint saved here
    # rsync -avz --no-g --no-p /home/walml/repos/zoobot/data/pretrained_models/pytorch/effnetb0_greyscale_224px.ckpt walml@narval.alliancecan.ca:/project/def-bovy/walml/zoobot/data/pretrained_models/pytorch
    checkpoint_loc = '/project/def-bovy/walml/zoobot/data/pretrained_models/pytorch/effnetb0_greyscale_224px.ckpt'

    logger = WandbLogger(name='debug', save_dir='/project/def-bovy/walml/wandb/debug', project='narval', log_model=False, offline=True)
    
    datamodule = GalaxyDataModule(
      label_cols=label_cols,
      catalog=train_catalog,  # very small, as a demo
      batch_size=batch_size,  # increase for faster training, decrease to avoid out-of-memory errors
      num_workers=num_workers  # TODO set to a little less than num. CPUs
    )
    datamodule.setup()
    model = finetune.FinetuneableZoobotClassifier(
      checkpoint_loc=checkpoint_loc,
      num_classes=num_classes,
      n_blocks=n_blocks
    )
    trainer = finetune.get_trainer(
        os.path.join(os.environ['SLURM_TMPDIR'], 'walml/finetune/checkpoints'),
        accelerator='gpu',
        devices=2,
        nodes=1,
        strategy='ddp',
        precision='16-mixed',
        max_epochs=max_epochs,
        enable_progress_bar=False,
        logger=logger
      )
    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule)
