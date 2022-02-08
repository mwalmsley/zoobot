

import os
import logging
import glob
import random
import shutil

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split

from zoobot.shared import schemas, label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import preprocess, define_model, alexnet_baseline, small_cnn_baseline
from zoobot.tensorflow.predictions import predict_on_tfrecords, predict_on_dataset
from zoobot.tensorflow.training import training_config
from zoobot.tensorflow.transfer_learning import utils
from zoobot.tensorflow.estimators import custom_layers


if __name__ == '__main__':

    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    """
    Set up your finetuning dataset
    
    Here, I'm using galaxies tagged or not tagged as "ring" by Galaxy Zoo volunteers.
    I've already saved a pandas dataframe with:
    - rows of each galaxy
    - columns of the path (path/to/img.png) and label (1 if tagged ring, 0 if not tagged ring)
    """

    requested_img_size = 300 # images will be resized from disk to this before preprocessing
    batch_size = 64  # 128 for paper, you'll need a good GPU. 64 for 2070 RTX, not sure if this will mess up batchnorm tho.
    file_format = 'png'

    # TODO you'll want to replace this with your own data
    df = pd.read_csv('data/example_ring_catalog_basic.csv')
    paths = list(df['local_png_loc'])
    labels = list(df['ring'].astype(int))
    logging.info('Labels: \n{}'.format(pd.value_counts(labels)))

    # randomly divide into train and validation sets using sklearn
    paths_train, paths_val, labels_train, labels_val= train_test_split(paths, labels, test_size=0.2, random_state=42)
    assert set(paths_train).intersection(set(paths_val)) == set()  # check there's no train/val overlap

    """
    Load the dataset into memory using tensorflow
    - raw_train_dataset and raw_val_dataset are the original images. If requested_img_size = int(requested_img_size is different to the image size on disk, they will be resized (which is slow). 
    - train_dataset and val_dataset are preprocessed according to your choices. Here, I normalise to [0, 1] interval and make greyscale. Augmentations are applied later.
    """

    raw_train_dataset = image_datasets.get_image_dataset(paths_train, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_train)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_val)
  
    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=['label'],  # image_datasets.get_image_dataset will put the labels arg under the 'label' key for each batch
        input_size=requested_img_size,
        normalise_from_uint8=True,  # divide by 255
        make_greyscale=True,  # take the mean over RGB channels
        permute_channels=False  # swap channels around randomly (no need when making greyscale anwyay)
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, preprocess_config)

    """
    Load the pretrained model (without the "head" output layer), freeze it, and add a new head
    """

    pretrained_checkpoint = 'data/pretrained_models/gz_decals_full_m0/in_progress'
    # should match how the model was trained
    crop_size = int(requested_img_size * 0.75)
    resize_size = 224  # 224 for paper

    log_dir = 'results/finetune_minimal'  # TODO you'll want to replace this with your own path

    # get headless model (inc. augmentations)
    logging.info('Loading pretrained model from {}'.format(pretrained_checkpoint))
    base_model = define_model.load_model(
      pretrained_checkpoint,
      include_top=False,  # do not include the head used for GZ DECaLS - we will add our own head
      input_size=requested_img_size,  # the preprocessing above did not change size
      crop_size=crop_size,  # model augmentation layers apply a crop...
      resize_size=resize_size,  # ...and then apply a resize
      output_dim=None  # headless so no effect
    )

    base_model.trainable = False  # freeze the headless model (no training allowed)

    new_head = tf.keras.Sequential([
      layers.InputLayer(input_shape=(7,7,1280)),  # base model dim before GlobalAveragePooling (ignoring batch)
      layers.GlobalAveragePooling2D(),
      # TODO the following layers will likely need some experimentation to find a good combination for your problem
      layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(1, activation="sigmoid", name='sigmoid_output')  # output should be one neuron w/ sigmoid for binary classification...
      # layers.Dense(3, activation="softmax", name="softmax_output")  # ...or N neurons w/ softmax for N-class classification
    ])

    # stick the new head on the pretrained base model
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(shape=(requested_img_size, requested_img_size, 1)),
      base_model,
      new_head
    ])

    """
    Retrain the model. Only the new head will train as the rest is frozen.
    """

    epochs = 80
    loss = tf.keras.losses.binary_crossentropy

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal learning rate is okay
        metrics=['accuracy']
    )
    model.summary()


    train_config = training_config.TrainConfig(
      log_dir=log_dir,
      epochs=epochs,
      patience=int(epochs/6)  # early stopping: if val loss does not improve for this many epochs in a row, end training
    )

    # acts inplace on model
    # saves best checkpoint to train_config.logdir / checkpoints
    training_config.train_estimator(
      model,
      train_config,  # e.g. how to train epochs, patience
      train_dataset,
      val_dataset
    )

    # evaluate performance on val set, repeating to marginalise over any test-time augmentations or dropout:
    losses = []
    for _ in range(5):
      losses.append(model.evaluate(val_dataset)[0])
    logging.info('Mean validation loss: {:.3f} (var {:.4f})'.format(np.mean(losses), np.var(losses)))
    # should train to a loss of around 0.54, equivalent to 75-80% accuracy on the (class-balanced) validation set

    """
    Well done!
    
    You can now use your finetuned models to make predictions on new data..
    See make_predictions.py for a self-contained example.
    """
    paths_pred = paths_val  # TODO for simplicitly I'll just make more predictions on the validation images, but you'll want to change this
    raw_pred_dataset = image_datasets.get_image_dataset(paths_pred, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size)

    ordered_paths = [x.numpy().decode('utf8') for batch in raw_pred_dataset for x in batch['id_str']]

    # must exactly match the preprocessing you used for training
    pred_config = preprocess.PreprocessingConfig(
      label_cols=[],  # image_datasets.get_image_dataset will put the labels arg under 'label' key for each batch
      input_size=requested_img_size,
      make_greyscale=True,
      normalise_from_uint8=True,
      permute_channels=False
    )
    pred_dataset = preprocess.preprocess_dataset(raw_pred_dataset, pred_config)

    predictions = model.predict(pred_dataset)

    data = [{'prediction': float(prediction), 'image_loc': local_png_loc} for prediction, local_png_loc in zip(predictions, ordered_paths)]
    pred_df = pd.DataFrame(data=data)
  
    example_predictions_loc = 'results/finetune_minimal/example_predictions.csv'
    pred_df.to_csv(example_predictions_loc, index=False)
    logging.info(f'Example predictions saved to {example_predictions_loc}')
