import logging

import pandas as pd

from zoobot.pytorch.training import finetune
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from zoobot.pytorch.estimators import define_model

if __name__ == '__main__':


    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv('data/example_ring_catalog_basic.csv')
    # # paths = list(df['local_png_loc'])
    # # labels = list(df['ring'].astype(int))
    # # logging.info('Labels: \n{}'.format(pd.value_counts(labels))) 
    # df['file_loc'] = df['local_png_loc'].str.replace('.png', '.jpg')
    # del df['local_png_loc']
    # df.to_csv('/home/walml/repos/zoobot/data/temp.csv', index=False)



    datamodule = GalaxyDataModule(
      label_cols=['ring'],
      catalog=df,
      batch_size=32
    )

    # datamodule.setup()
    # for images, labels in datamodule.train_dataloader():
    #   print(images.shape)
    #   print(labels.shape)
    #   exit()

    config = {
        'trainer': {
        #   'devices': 1,
          'accelerator': 'cpu'
        },
        'finetune': {
            'dim': 1280,  # TODO rename
            'n_epochs': 100,
            'n_layers': 2,
            'n_classes': 2
        }
    }

    ckpt_loc = '/home/walml/repos/gz-decals-classifiers/results/benchmarks/pytorch/dr5/dr5_py_gr_2270/checkpoints/epoch=360-step=231762.ckpt'
    model = define_model.ZoobotLightningModule.load_from_checkpoint(ckpt_loc)  # or .best_model_path, eventually

    """
    Model:  ZoobotLightningModule(
    (train_accuracy): Accuracy()
    (val_accuracy): Accuracy()
    (model): Sequential(
      (0): EfficientNet(
    """
    # TODO self properties needed
    # 0 and 1 are self.Accuracy
    # print('Model: ', list(model.modules())[0])
    # zoobot = list(model.modules())[0]
    # print('Model: ', list(zoobot.modules())[0])

    # for name, _ in model.named_modules():
    #   print(name)

    encoder = model.get_submodule('model.0')  # includes avgpool and head
    # print(encoder)


    # encoder = define_model.get_plain_pytorch_zoobot_model(output_dim=1280, include_top=False)
    # TODO remove top?

    finetune.run_finetuning(config, encoder, datamodule, logger=None)

