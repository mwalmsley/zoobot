import logging
import os

import pandas as pd

from zoobot.pytorch.training import finetune
from galaxy_datasets import galaxy_mnist
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    zoobot_dir = '/home/walml/repos/zoobot'  # TODO set to directory where you cloned Zoobot
    data_dir = '/home/walml/repos/galaxy-datasets/roots/galaxy_mnist'  # TODO set to any directory. rings dataset will be downloaded here
    batch_size = 32
    num_workers= 8
    n_blocks = 1  # EffnetB0 is divided into 7 blocks. set 0 to only fit the head weights. Set 1, 2, etc to finetune deeper. 
    max_epochs = 1  #  6 epochs should get you ~93% accuracy. Set much higher (e.g. 1000) for harder problems, to use Zoobot's default early stopping. 
    # the remaining key parameters for high accuracy are weight_decay, learning_rate, and lr_decay. You might like to tinker with these.

    # load in catalogs of images and labels to finetune on
    # each catalog should be a dataframe with columns of "id_str", "file_loc", and any labels
    # here I'm using galaxy-datasets to download some premade data - check it out for examples

    train_catalog, _ = galaxy_mnist(root=data_dir, download=True, train=True)
    test_catalog, _ = galaxy_mnist(root=data_dir, download=True, train=False)

    # wondering about "label_cols"? 
    # This is a list of catalog columns which should be used as labels
    # Here, it's a single column, 'label', with values 0-3 (for each of the 4 classes)
    label_cols = ['label']
    num_classes = 4
  
    # load a pretrained checkpoint saved here
    checkpoint_loc = os.path.join(zoobot_dir, 'data/pretrained_models/pytorch/effnetb0_greyscale_224px.ckpt')
    
    # save the finetuning results here
    save_dir = os.path.join(zoobot_dir, 'results/pytorch/finetune/finetune_multiclass_classification')

    datamodule = GalaxyDataModule(
      label_cols=label_cols,
      catalog=train_catalog,  # very small, as a demo
      batch_size=batch_size,  # increase for faster training, decrease to avoid out-of-memory errors
      num_workers=num_workers  # TODO set to a little less than num. CPUs
    )
    datamodule.setup()
    # optionally, check the data loads and looks okay
    # for images, labels in datamodule.train_dataloader():
    #   print(images.shape)
    #   print(labels.shape)
    #   exit()

  
    model = finetune.FinetuneableZoobotClassifier(
      checkpoint_loc=checkpoint_loc,
      num_classes=num_classes,
      n_blocks=n_blocks
    )
    # under the hood, this does:
    # encoder = finetune.load_pretrained_encoder(checkpoint_loc)
    # model = finetune.FinetuneableZoobotClassifier(encoder=encoder, ...)

    # retrain to find rings
    trainer = finetune.get_trainer(save_dir, accelerator='auto', max_epochs=max_epochs)
    trainer.fit(model, datamodule)
    # can now use this model or saved checkpoint to make predictions on new data. Well done!

    # see how well the model performs
    # (don't do this all the time)
    trainer.test(model, datamodule)

    # we can load the model later any time
    # pretending we want to load from scratch:
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    finetuned_model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(best_checkpoint)

    from zoobot.pytorch.predictions import predict_on_catalog

    predictions_save_loc = os.path.join(save_dir, 'finetuned_predictions.csv')
    predict_on_catalog.predict(
      test_catalog,
      finetuned_model,
      n_samples=1,
      label_cols=['class_{}'.format(n) for n in range(num_classes)],  # TODO feel free to rename, it's just for the csv header
      save_loc=predictions_save_loc,
      trainer_kwargs={'accelerator': 'auto'},
      datamodule_kwargs={'batch_size': batch_size, 'num_workers': num_workers}
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

    predictions = pd.read_csv(predictions_save_loc)
    print(predictions)

    exit()  # now over to you!
