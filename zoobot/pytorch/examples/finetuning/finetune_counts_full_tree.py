import logging
import os

import pandas as pd
import numpy as np

from zoobot.pytorch.training import finetune
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.shared.schemas import cosmic_dawn_ortho_schema

"""
Example for finetuning Zoobot on counts of volunteer responses throughout a complex decision tree.
Useful if you are running a Galaxy Zoo campaign with many questions and answers.
Probably you are in the GZ collaboration if so!

For simpler examples, see:
- finetune_binary_classification.py to finetune on class (0 or 1) labels
- finetune_counts_single_question.py to finetune on answer counts (e.g. 12 volunteers said Yes, 4 said No) for a single question

This currently uses unpublished (hence private, for now) GZ Cosmic Dawn data
"""


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    schema = cosmic_dawn_ortho_schema

    if os.path.isdir('/share/nas2'):  # run on cluster
        repo_dir = '/share/nas2/walml/repos'
        accelerator = 'gpu'
        devices = 1
        batch_size = 128
        prog_bar = False
        max_galaxies = None
    else:  # test locally
        repo_dir = '/home/walml/repos'
        accelerator = 'gpu'
        devices = None
        batch_size = 32
        prog_bar = True
        # max_galaxies = 256
        max_galaxies = None

    # TODO not yet made public
    # pd.DataFrame with columns 'id_str' (unique id), 'file_loc' (path to image),
    # and label_cols (e.g. smooth-or-featured-cd_smooth) with count responses
    df = pd.read_parquet(os.path.join(
        repo_dir, 'zoobot/data/gz_cosmic_dawn_early_aggregation_ortho_with_file_locs.parquet'))
    # sometimes auto-cast to float, which causes issue when saving hdf5
    df['id_str'] = df['id_str'].astype(str)

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
        'checkpoint': {
            'file_template': "{epoch}",
            'save_top_k': 1
        },
        'early_stopping': {
            'patience': 15
        },
        'trainer': {
            'devices': devices,
            'accelerator': accelerator
        },
        'finetune': {
            'n_epochs': 100,
            'n_layers': 2,
            'label_dim': len(schema.label_cols),
            'label_mode': 'count',
            'schema': schema,
            'prog_bar': prog_bar
        }
    }

    # TODO not yet made public
    ckpt_loc = os.path.join(
        repo_dir, 'gz-decals-classifiers/results/benchmarks/pytorch/dr5/dr5_py_gr_2270/checkpoints/epoch=360-step=231762.ckpt')
    encoder = finetune.load_encoder(ckpt_loc)

    save_dir = os.path.join(
        repo_dir, f'gz-decals-classifiers/results/finetune_{np.random.randint(1e8)}')

    # can do logger=None or, to use wandb:
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project='finetune', name='full_tree_example')

    # key method
    model, _ = finetune.run_finetuning(
        config, encoder, datamodule, save_dir=save_dir, logger=logger)

    # now save predictions on test set to evaluate performance

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
