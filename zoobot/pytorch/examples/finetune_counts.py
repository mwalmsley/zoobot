import logging
import os
from trace import Trace
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from zoobot.pytorch.training import finetune
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.shared.schemas import cosmic_dawn_ortho_schema

if __name__ == '__main__':


    logging.basicConfig(level=logging.INFO)

    schema = cosmic_dawn_ortho_schema

    # temp - tweak catalog to include file_loc and renamed columns

    # slightly updated from the first version sent to Cam
    # df = pd.read_parquet('data/gz_cosmic_dawn_early_aggregation.parquet')
    # renamer = {}
    # for question in schema.questions:
    #   renamer[question.text.replace('-cd', '') + '_total-votes'] = question.text + '_total-votes'
    #   for answer in question.answers:
    #     renamer[answer.text.replace('-cd', '')] = answer.text
    #     renamer[answer.text.replace('-cd', '') + '_fraction'] = answer.text + '_fraction'
    # print(renamer)
    # df = df.rename(columns=renamer)

    # for label_col in schema.label_cols:
    #   assert label_col in df.columns.values, 'Missing {}'.format(label_col)

    # file_key = pd.read_csv('data/cosmic_dawn_file_key.csv')
    # file_key['filename'] = file_key['locations'].apply(lambda x: f'{os.path.basename(urlparse(x).path)}')
    # # file_key['file_loc'] =  file_key['filename'].apply(lambda x: os.path.join(repo_dir, 'zoobot/data/example_images/cosmic_dawn', x))  # absolute path, annoying to share to cluster
    # file_key['file_loc'] =  file_key['filename'].apply(lambda x: os.path.join('data/example_images/cosmic_dawn', x))  # relative path

    # df = pd.merge(df, file_key[['id_str', 'file_loc']], on='id_str', how='inner')

    # df.to_parquet('data/gz_cosmic_dawn_early_aggregation_with_file_locs.parquet', index=False)
    # exit()

    if os.path.isdir('/share/nas2'):  # cluster running
      repo_dir = '/share/nas2/walml/repos'
      accelerator = 'gpu'
      devices = 2
      batch_size = 128
      prog_bar = False
      max_galaxies = None
    else:  # local testing
      repo_dir = '/home/walml/repos'
      accelerator = 'cpu'
      devices = None
      batch_size = 64 
      prog_bar = True
      max_galaxies = 256

    df = pd.read_parquet(os.path.join(repo_dir, 'zoobot/data/gz_cosmic_dawn_early_aggregation_with_file_locs.parquet'))
    df['id_str'] = df['id_str'].astype(str)  # sometimes auto-cast to float, which causes issue when saving hdf5
  
    if max_galaxies is not None:
        df = df.sample(max_galaxies)

    datamodule = GalaxyDataModule(
      label_cols=schema.label_cols,
      catalog=df,
      batch_size=batch_size
      # default_augs
    )

    # datamodule.setup()
    # for images, labels in datamodule.train_dataloader():
    #   print(images.shape)
    #   print(labels.shape)
    #   exit()

    # exit()


    config = {
        'trainer': {
          'devices': devices,
          'accelerator': accelerator
        },
        'finetune': {
            'encoder_dim': 1280,
            'n_epochs': 1,
            'n_layers': 0,
            'label_dim': len(schema.label_cols),
            'label_mode': 'count',
            'schema': schema,
            'prog_bar': prog_bar
        }
    }

    ckpt_loc = os.path.join(repo_dir, 'gz-decals-classifiers/results/benchmarks/pytorch/dr5/dr5_py_gr_2270/checkpoints/epoch=360-step=231762.ckpt')
    model = define_model.ZoobotLightningModule.load_from_checkpoint(ckpt_loc)  # or .best_model_path, eventually

    """
    Model:  ZoobotLightningModule(
    (train_accuracy): Accuracy()
    (val_accuracy): Accuracy()
    (model): Sequential(
      (0): EfficientNet(
    """
    encoder = model.get_submodule('model.0')  # includes avgpool and head

    save_dir = os.path.join(repo_dir, f'gz-decals-classifiers/results/finetune_{np.random.randint(1e8)}')
    _, model = finetune.run_finetuning(config, encoder, datamodule, logger=None, save_dir=save_dir)


    test_catalog = datamodule.test_catalog  # auto-split within datamodule. pull out again.
    assert len(test_catalog) > 0
    datamodule_kwargs = {'batch_size': batch_size}
    trainer_kwargs = {'devices': 1, 'accelerator': accelerator}
    predict_on_catalog.predict(test_catalog, model, n_samples=1, label_cols=schema.label_cols, save_loc=os.path.join(save_dir, 'test_predictions.csv'), datamodule_kwargs=datamodule_kwargs, trainer_kwargs=trainer_kwargs)
