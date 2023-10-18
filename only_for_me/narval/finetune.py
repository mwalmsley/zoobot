import logging
import os
import shutil

from zoobot.pytorch.training import finetune
from galaxy_datasets import galaxy_mnist
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logging.info('Begin')

    data_dir = '/tmp/walml/finetune'
    os.mkdirs(data_dir)

    remote_image_dir = '/project/def-bovy/walml/repos/galaxy-imagesets/roots/galaxy_mnist'
    image_dir = data_dir + '/images'
    shutil.copytree(remote_image_dir, image_dir)
    logging.info('Copied')

    remote_zoobot_dir = '/project/def-bovy/walml/repos/zoobot'

    batch_size = 32
    num_workers= 8
    n_blocks = 1  # EffnetB0 is divided into 7 blocks. set 0 to only fit the head weights. Set 1, 2, etc to finetune deeper. 
    max_epochs = 6  #  6 epochs should get you ~93% accuracy. Set much higher (e.g. 1000) for harder problems, to use Zoobot's default early stopping. \

    train_catalog, _ = galaxy_mnist(root=data_dir, download=False, train=True)
    test_catalog, _ = galaxy_mnist(root=data_dir, download=False, train=False)
    logging.info('Data ready')

    label_cols = ['label']
    num_classes = 4
  
    # load a pretrained checkpoint saved here
    # rsync -avz --no-g --no-p /home/walml/repos/zoobot/data/pretrained_models/pytorch/effnetb0_greyscale_224px.ckpt walml@narval.alliancecan.ca:/project/def-bovy/walml/zoobot/data/pretrained_models/pytorch
    checkpoint_loc = os.path.join(remote_zoobot_dir, 'data/pretrained_models/pytorch/effnetb0_greyscale_224px.ckpt')
    
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
    trainer = finetune.get_trainer(data_dir, accelerator='auto', max_epochs=max_epochs)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
