import logging
import os

import pandas as pd
import numpy as np

from galaxy_datasets import gz_rings
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

from zoobot.pytorch.training import finetune
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.shared.schemas import gz_rings_schema

"""
Example for finetuning Zoobot on counts of volunteer responses to a single question.
Useful for finetuning on GZ Mobile responses.

For a simpler example doing binary classification (i.e. on labels which are 0 or 1),
see finetune_counts_binary_classification.py

This currently uses unpublished (hence private, for now) GZ Rings data (collected with GZ Mobile)
"""


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)



    schema = gz_rings_schema

    if os.path.isdir('/share/nas2'):  # run on cluster
        repo_dir = '/share/nas2/walml/repos'
        accelerator = 'gpu'
        devices = 1
        batch_size = 128
        prog_bar = False
        max_galaxies = None
    else:  # test locally
        repo_dir = '/home/walml/repos'
        accelerator = 'cpu'
        devices = None
        batch_size = 64
        prog_bar = True
        max_galaxies = 256

    checkpoint_loc = os.path.join(repo_dir, 'zoobot/data/pretrained_models/temp/dr5_py_gr_2270/checkpoints/epoch=360-step=231762.ckpt')

    save_dir = os.path.join(
        repo_dir, f'zoobot/results/pytorch/finetune/finetune_counts_single_question')
    

    # TODO not yet made public
    # df includes columns 'id_str' (unique id), 'file_loc' (path to image),
    # and label_cols with count responses (here 'ring_yes' and 'ring_no')
    # (we already have the label_cols via the schema, as schema.label_cols)
    df, _ = gz_rings(
        root=os.path.join(repo_dir, 'pytorch-galaxy-datasets/roots/gz_rings'),
        train=True,
        download=False
    )

    # make any modifications to the catalog here 
    if max_galaxies is not None:
        df = df.sample(max_galaxies)

    datamodule = GalaxyDataModule(
        label_cols=schema.label_cols,
        catalog=df,
        batch_size=batch_size
        # uses default_augs
    )
    datamodule.setup()

    config = {
        'trainer': {
            'devices': devices,
            'accelerator': accelerator
        },
        'finetune': {
            'encoder_dim': 1280,
            'n_epochs': 50,
            'n_layers': 2,  # min 0 (i.e. just train the output layer). max 4 (i.e. all layers)
            'label_dim': len(schema.label_cols),
            'label_mode': 'count',
            'schema': schema,
            'prog_bar': prog_bar
        }
    }

    model = define_model.ZoobotLightningModule.load_from_checkpoint(
        checkpoint_loc)  # or .best_model_path, eventually

    """
    Model:  ZoobotLightningModule(
    (train_accuracy): Accuracy()
    (val_accuracy): Accuracy()
    (model): Sequential(
      (0): EfficientNet(
    """
    encoder = model.get_submodule('model.0')  # includes avgpool and head

    # key method
    _, model = finetune.run_finetuning(
        config, encoder, datamodule, save_dir, logger=None)

    # demonstrate saving predictions using test set
  
    # auto-split within datamodule. pull out again.
    test_catalog = datamodule.test_catalog
    assert len(test_catalog) > 0
    datamodule_kwargs = {'batch_size': batch_size}
    trainer_kwargs = {'devices': 1, 'accelerator': accelerator}
    predict_on_catalog.predict(
        test_catalog,
        model,
        n_samples=1,
        label_cols=schema.label_cols,
        save_loc=os.path.join(save_dir, 'test_predictions.csv'),
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs
    )
