import logging
import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from galaxy_datasets.tensorflow.datasets import get_image_dataset, add_transforms_to_dataset
from galaxy_datasets.transforms import default_transforms

from zoobot.tensorflow.training import finetune
from zoobot.tensorflow.estimators import define_model


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    zoobot_dir = '/home/walml/repos/zoobot'  # TODO set to directory where you cloned Zoobot

    # TODO you can update these to suit own data
    label_cols = ['ring', 'not_ring']  # name of column in catalog with binary (0 or 1) labels for your classes
    catalog_loc = os.path.join(zoobot_dir, 'data/example_ring_catalog_basic.csv')  # includes label_cols column (here, 'ring') with labels
    checkpoint_loc = os.path.join(zoobot_dir, 'data/pretrained_models/temp/dr5_tf_gr_18845/checkpoint')
    save_dir = os.path.join(zoobot_dir, 'results/tensorflow/finetune/finetune_binary_classification')

    img_size = 224
    batch_size = 16

    df = pd.read_csv(catalog_loc)
    df['not_ring'] = (~df['ring'].astype(bool)).astype(int)

    # standard data setup - split into train/val/test
    train_catalog, hidden_catalog = train_test_split(df, train_size=0.7)
    val_catalog, test_catalog = train_test_split(hidden_catalog, train_size=1./3.)

    train_labels, train_image_paths = train_catalog[label_cols], train_catalog['file_loc']
    val_labels, val_image_paths = val_catalog[label_cols], val_catalog['file_loc']
    test_labels, test_image_paths = test_catalog[label_cols], test_catalog['file_loc']

    # print(train_labels)
    # exit()

    # load as tf datasets
    train_dataset = get_image_dataset(
        train_image_paths, labels=train_labels, requested_img_size=img_size, greyscale=True
    )
    val_dataset = get_image_dataset(
        val_image_paths, labels=val_labels, requested_img_size=img_size, greyscale=True
    )
    test_dataset = get_image_dataset(
        test_image_paths, labels=test_labels, requested_img_size=img_size, greyscale=True
    )

    # specify augmentations to use
    train_transforms = default_transforms()
    inference_transforms = default_transforms(
        # do not zoom at inference time
        crop_scale_bounds=(1., 1.),
        crop_ratio_bounds=(1., 1.)
    )
    train_dataset = add_transforms_to_dataset(train_dataset, train_transforms)
    val_dataset = add_transforms_to_dataset(val_dataset, inference_transforms)
    test_dataset = add_transforms_to_dataset(test_dataset, inference_transforms)

    # batch, shuffle, prefetch
    train_dataset = train_dataset.shuffle(5000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)

    config = {
        'trainer': {
            'devices': 1,
            'accelerator': 'cpu'
        },
        'finetune': {
            'img_size': 224,
            'encoder_dim': 1280,
            'label_dim': 2,  # 1 column, containing 0 or 1
            'n_epochs': 1,
            'n_layers': 2,
            'patience': 5,
            'batch_size': batch_size,
            'label_mode': 'classification',
            'prog_bar': True
        }
    }

    # load pretrained model
    encoder = define_model.load_model(
      checkpoint_loc,
      expect_partial=True,  # the optimizer state will be loaded. expect_partial silences this warning.
      include_top=False,  # do not include the head used for GZ DECaLS - we will add our own head
      input_size=img_size,  # the preprocessing above did not change size
      output_dim=None,  # headless so no effect
      channels=1
    )

    # print(encoder.get_layer('headless_efficientnet').summary())
    # print(encoder.get_layer('headless_efficientnet').get_layer('efficientnet-b0').summary())

    
    finetune.run_finetuning(config, encoder, train_dataset, val_dataset, test_dataset, save_dir)
    # can now use this saved checkpoint to make predictions on new data. Well done!
