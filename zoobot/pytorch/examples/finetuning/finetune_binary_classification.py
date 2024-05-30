import logging
import os

from zoobot.pytorch.training import finetune
from galaxy_datasets import demo_rings
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    zoobot_dir = '/Users/user/repos/zoobot'  # TODO set to directory where you cloned Zoobot

    # load in catalogs of images and labels to finetune on
    # each catalog should be a dataframe with columns of "id_str", "file_loc", and any labels
    # here I'm using galaxy-datasets to download some premade data - check it out for examples
    data_dir = '/Users/user/repos/galaxy-datasets/roots/demo_rings'  # TODO set to any directory. rings dataset will be downloaded here
    train_catalog, _ = demo_rings(root=data_dir, download=True, train=True)
    test_catalog, _ = demo_rings(root=data_dir, download=True, train=False)

    # wondering about "label_cols"? 
    # This is a list of catalog columns which should be used as labels
    # Here:
    label_cols = ['ring']
    # For binary classification, the label column should have binary (0 or 1) labels for your classes
    # To support more complicated labels, Zoobot expects a list of columns. A list with one element works fine.
   
    # save the finetuning results here
    save_dir = os.path.join(zoobot_dir, 'results/pytorch/finetune/finetune_binary_classification')

    datamodule = GalaxyDataModule(
      label_cols=label_cols,
      catalog=train_catalog,  # very small, as a demo
      batch_size=32
    )
    # datamodule.setup()
    # for images, labels in datamodule.train_dataloader():
    #   print(images.shape)
    #   print(labels.shape)
    #   exit()

  
    model = finetune.FinetuneableZoobotClassifier(
      name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
      num_classes=2,
      n_blocks=0  # only updating the head weights. Set e.g. 1, 2 to finetune deeper. 
    )
    # under the hood, this does:
    # encoder = finetune.load_pretrained_encoder(checkpoint_loc)
    # model = finetune.FinetuneableZoobotClassifier(encoder=encoder, ...)

    # retrain to find rings
    trainer = finetune.get_trainer(save_dir, accelerator='cpu', max_epochs=1)
    trainer.fit(model, datamodule)
    # can now use this model or saved checkpoint to make predictions on new data. Well done!

    # pretending we want to load from scratch:
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    finetuned_model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(best_checkpoint)

    from zoobot.pytorch.predictions import predict_on_catalog

    predict_on_catalog.predict(
      test_catalog,
      finetuned_model,
      n_samples=1,
      label_cols=label_cols,
      save_loc=os.path.join(save_dir, 'finetuned_predictions.csv'),
      datamodule_kwargs={'batch_size': 32},  # we also need to set batch size here, or you may run out of memory
      trainer_kwargs={'accelerator': 'gpu'}  
    )
    """
    Under the hood, this is essentially doing:

    import pytorch_lightning as pl
    predict_trainer = pl.Trainer(devices=1, max_epochs=-1)
    predict_datamodule = GalaxyDataModule(
      label_cols=None,  # important, else you will get "conv2d() received an invalid combination of arguments"
      predict_catalog=test_catalog,
      batch_size=32
    )
    preds = predict_trainer.predict(finetuned_model, predict_datamodule)
    print(preds)
    """