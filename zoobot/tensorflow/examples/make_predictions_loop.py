import os
import logging
import glob
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from zoobot.shared import schemas
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.predictions import predict_on_tfrecords, predict_on_dataset
from zoobot.shared import label_metadata

"""
This script is the more advanced (yet more useful) version of make_predictions.py
Use it to make predictions on large datasets.
Predictions might be GZ answers, finetuned problems, or galaxy representations (see representations/README.md)
"""


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    run_name = 'example_rings'
    overwrite = True

    """Dataframe with list of images on which to make predictions"""
    df = pd.read_parquet('data/example_ring_catalog_basic.csv')  # TODO customise your catalog here
    df['png_loc'] = df['local_png_loc'].apply(lambda x: x)  # TODO customise file your paths here, if needed (e.g. catalog made on desktop but predictions running on cluster)
    logging.info('Loaded {} example galaxies for predictions'.format(len(df)))

    png_locs = list(df['png_loc'])

    initial_size = 300  # 300 for paper, from tfrecord or from png (png will be resized when loaded, before preprocessing)
    batch_size = 256  # 128 for paper, you'll need a very good GPU. 8 for debugging, 64 for RTX 2070, 256 for A100, 512 for 2xA100
    crop_size = int(initial_size * 0.75)
    resize_size = 224  # 224 for paper

    greyscale = True
    if greyscale:
        channels = 1
    else:
        channels = 3

    """For predicting GZ answers - the model as trained"""
    # checkpoint_dir = 'data/pretrained_models/decals_dr_train_set_only_m0/in_progress'  # m0 in the paper, test set not included in training

    # model = define_model.load_model(
    #     checkpoint_dir,
    #     include_top=True,
    #     input_size=initial_size,
    #     crop_size=crop_size,
    #     resize_size=resize_size,
    #     channels=channels,
    #     output_dim=34 
    # )
    # question_answer_pairs = label_metadata.decals_pairs
    # dependencies = gz2_and_decals_dependencies
    # schema = schemas.Schema(question_answer_pairs, dependencies)
    # label_cols = schema.label_cols

    """For saving the activations (representations) - use the model with no head, only GlobalAveragePooling2D"""
    # base_model = define_model.load_model(
    #     checkpoint_dir,
    #     include_top=False,
    #     input_size=initial_size,
    #     crop_size=crop_size,
    #     resize_size=resize_size,
    #     output_dim=None,
    #     channels=channels
    # )
    # new_head = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(7,7,1280)),
    #     tf.keras.layers.GlobalAveragePooling2D()
    # ])
    # model = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(initial_size, initial_size, channels)),
    #     base_model,
    #     new_head
    # ])
    # label_cols = [f'feat_{x}' for x in range(1280)]

    """
    For making predictions on a new problem with n classes
    Be sure to use the same model architecture you specified during finetuning
    """
    checkpoint_dir = 'data/pretrained_models/decals_dr_trained_on_all_labelled_m0/in_progress'
    finetuned_dir = 'results/finetune_advanced/full/checkpoint'
    base_model = define_model.load_model(
        checkpoint_dir,
        include_top=False,
        input_size=initial_size,
        crop_size=crop_size,
        resize_size=resize_size,
        channels=channels,
        output_dim=34 
    )
    new_head = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(7,7,1280)),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dropout(0.75),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.75),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.75),
      tf.keras.layers.Dense(1, activation="sigmoid", name='sigmoid_output')
    ])
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(initial_size, initial_size, 1)),
      base_model,
      new_head
    ])
    define_model.load_weights(model, finetuned_dir, expect_partial=True)

    label_cols = ['ring']


    """
    Actually do the predictions!
    """
    png_batch_size = 10000
    png_start_index = 0
    n_samples = 1
    while png_start_index < len(png_locs):
        
        # TODO update this path as needed
        save_loc = 'data/results/make_predictions_loop/{}_{}.hdf5'.format(run_name, png_start_index)
        if not os.path.isfile(save_loc) or overwrite:

            unordered_image_paths = png_locs[png_start_index:png_start_index+png_batch_size]
            file_format = 'png'

            assert len(unordered_image_paths) > 0
            assert os.path.isfile(unordered_image_paths[0])

            # will only save with name of last folder, but that's okay, it has paths

            """
            Load the images as a tf.dataset, just as for training
            """

            raw_image_ds = image_datasets.get_image_dataset([str(x) for x in unordered_image_paths], file_format, initial_size, batch_size)

            preprocessing_config = preprocess.PreprocessingConfig(
                label_cols=[],  # no labels are needed, we're only doing predictions
                input_size=initial_size,
                make_greyscale=greyscale,
                normalise_from_uint8=True
            )
            image_ds = preprocess.preprocess_dataset(raw_image_ds, preprocessing_config)
            # image_ds will give batches of (images, paths) when label_cols=[]

            
            """
            Define the model and load the weights.
            You must define the model exactly the same way as when you trained it.
            If you have done finetuning, use include_top=False and replace the output layers exactly as you did when training.
            For example, below is how to load the model in finetune_minimal.py.
            """
            predict_on_dataset.predict(image_ds, model, n_samples, label_cols, save_loc)

        png_start_index += png_batch_size
