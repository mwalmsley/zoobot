import logging
import os

import pandas as pd

from zoobot.pytorch.training import finetune
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from zoobot.pytorch.estimators import define_model

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    zoobot_dir = '/home/walml/repos/zoobot'  # TODO set to directory where you cloned Zoobot

    # TODO you can update these to suit own data
    label_col = 'ring'  # name of column in catalog with binary (0 or 1) labels for your classes
    label_cols = [label_col]  # To support more complicated labels, Zoobot expects a list of columns. A list with one element works fine.
    catalog_loc = os.path.join(zoobot_dir, 'data/example_ring_catalog_basic.csv')  # includes label_col column (here, 'ring') with labels
    checkpoint_loc = os.path.join(zoobot_dir, 'data/pretrained_models/temp/dr5_py_gr_2270/checkpoints/epoch=360-step=231762.ckpt')
    save_dir = os.path.join(zoobot_dir, 'results/pytorch/finetune/finetune_binary_classification')

    df = pd.read_csv(catalog_loc)

    datamodule = GalaxyDataModule(
      label_cols=label_cols,
      catalog=df,
      batch_size=32
    )

    datamodule.setup()
    # for images, labels in datamodule.train_dataloader():
    #   print(images.shape)
    #   print(labels.shape)
    #   exit()

    config = {
        'trainer': {
            'devices': 1,
            'accelerator': 'cpu'
        },
        'finetune': {
            'encoder_dim': 1280,
            'label_dim': 2,
            'n_epochs': 100,
            'n_layers': 2,
            'label_mode': 'classification',
            'prog_bar': True
        }
    }

    model = define_model.ZoobotLightningModule.load_from_checkpoint(checkpoint_loc)  # or .best_model_path, eventually

    """
    Model:  ZoobotLightningModule(
    (train_accuracy): Accuracy()
    (val_accuracy): Accuracy()
    (model): Sequential(
      (0): EfficientNet(
    """
    encoder = model.get_submodule('model.0')  # includes avgpool and head

    best_checkpoint_path, _model = finetune.run_finetuning(config, encoder, datamodule, save_dir, logger=None)

    # can now use this model or saved checkpoint to make predictions on new data. Well done!

    # pretending we want to load from scratch:
    finetuned_model = finetune.FinetunedZoobotLightningModule.load_from_checkpoint(best_checkpoint_path)

    # or using the convenient zoobot function
    from zoobot.pytorch.predictions import predict_on_catalog

    predict_on_catalog.predict(
      df.sample(128),
      finetuned_model,
      n_samples=1,
      label_cols=label_cols,
      save_loc=os.path.join(save_dir, 'finetuned_predictions.csv')
      # trainer_kwargs={'accelerator': 'gpu'}
    )



    """
    Under the hood, this is essentially doing:

    import pytorch_lightning as pl
    predict_trainer = pl.Trainer(devices=1, max_epochs=-1)
    predict_datamodule = GalaxyDataModule(
      label_cols=None,  # important, else you will get "conv2d() received an invalid combination of arguments"
      predict_catalog=df.sample(64),
      batch_size=32
    )
    preds = predict_trainer.predict(finetuned_model, predict_datamodule)
    print(preds)
    """