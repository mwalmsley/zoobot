import os
import logging
import glob
import pandas as pd

import tensorflow as tf

from zoobot.shared import label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.predictions import predict_on_tfrecords, predict_on_dataset


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    """
    List the images to make predictions on.
    """
    file_format = 'png'

    # utility function to easily list the images in a folder.
    unordered_image_paths = predict_on_dataset.paths_in_folder('data/example_images/basic', file_format=file_format, recursive=False)

    ## or maybe you already have a list from a catalog?
    # unordered_image_paths = df['paths']

    assert len(unordered_image_paths) > 0
    assert os.path.isfile(unordered_image_paths[0])

    """
    Load the images as a tf.dataset, just as for training
    """
    initial_size = 300  # 300 for paper, from tfrecord or from png (png will be resized when loaded, before preprocessing)
    batch_size = 256  # 128 for paper, you'll need a very good GPU. 8 for debugging, 64 for RTX 2070, 256 for A100
    raw_image_ds = image_datasets.get_image_dataset([str(x) for x in unordered_image_paths], file_format, initial_size, batch_size)

    preprocessing_config = preprocess.PreprocessingConfig(
        label_cols=[],  # no labels are needed, we're only doing predictions
        input_size=initial_size,
        make_greyscale=True,
        normalise_from_uint8=True  # False for tfrecords with 0-1 floats, True for png/jpg with 0-255 uints
    )
    image_ds = preprocess.preprocess_dataset(raw_image_ds, preprocessing_config)
    # image_ds will give batches of (images, paths) when label_cols=[]

    
    """
    Define the model and load the weights.
    You must define the model exactly the same way as when you trained it.
    """
    crop_size = int(initial_size * 0.75)
    resize_size = 224  # 224 for paper
    channels = 3

    checkpoint_loc = 'data/pretrained_models/decals_dr_trained_on_all_labelled_m0/in_progress'

    """
    If you're just using the full pretrained Galaxy Zoo model, without finetuning, you can just use include_top=True.
    """

    model = define_model.load_model(
        checkpoint_loc=checkpoint_loc,
        include_top=True,
        input_size=initial_size,
        crop_size=crop_size,
        resize_size=resize_size,
        expect_partial=True  # optimiser state will not load as we're not using it for predictions
    )

    label_cols = label_metadata.decals_label_cols  

    """
    If you have done finetuning, use include_top=False and replace the output layers exactly as you did when training.
    For example, below is how to load the model in finetune_minimal.py.
    """

    # finetuned_dir = 'results/finetune_advanced/full/checkpoint'
    # base_model = define_model.load_model(
    #   checkpoint_dir,
    #   include_top=False,
    #   input_size=initial_size,
    #   crop_size=crop_size,
    #   resize_size=resize_size,
    #   output_dim=None 
    # )
    # new_head = tf.keras.Sequential([
    #   tf.keras.layers.InputLayer(input_shape=(7,7,1280)),
    #   tf.keras.layers.GlobalAveragePooling2D(),
    #   tf.keras.layers.Dropout(0.75),
    #   tf.keras.layers.Dense(64, activation='relu'),
    #   tf.keras.layers.Dropout(0.75),
    #   tf.keras.layers.Dense(64, activation='relu'),
    #   tf.keras.layers.Dropout(0.75),
    #   tf.keras.layers.Dense(1, activation="sigmoid", name='sigmoid_output')
    # ])
    # model = tf.keras.Sequential([
    #   tf.keras.layers.InputLayer(input_shape=(initial_size, initial_size, 1)),
    #   base_model,
    #   new_head
    # ])
    # define_model.load_weights(model, finetuned_dir, expect_partial=True)

    # label_cols = ['ring']

    # save_loc = 'data/results/make_predictions_example.csv'  # supported, but not recommended - especially with n_samples > 1
    save_loc = 'data/results/make_predictions_example.hdf5'
    n_samples = 5
    predict_on_dataset.predict(image_ds, model, n_samples, label_cols, save_loc)
