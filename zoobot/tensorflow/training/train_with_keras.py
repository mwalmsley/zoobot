import os
import logging
import contextlib
from random import random
from sklearn.model_selection import train_test_split

import tensorflow as tf

from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.training import training_config, losses
from zoobot.tensorflow.estimators import preprocess, define_model


def train(
    # absolutely crucial arguments
    save_dir,  # save model here
    schema,  # answer these questions
    # input data - specify *either* catalog (to be split) or the splits themselves
    catalog=None,
    train_catalog=None,
    val_catalog=None,
    test_catalog=None,
    # model training parameters
    # TODO architecture_name=, only EfficientNet is currenty implemented
    batch_size=256,
    dropout_rate=0.2,
    # TODO drop_connect_rate not implemented
    epochs=1000,
    patience=8,
    # augmentation parameters
    color=False,
    # TODO I dislike this confusion/duplication - adjust when refactoring augmentations
    img_size_to_load=300,  # resizing on load *before* augmentations, will skip if given same size as on disk
    resize_size=224,  # resizing *during* augmentations, will skip if given appropriate crop
    # ideally, set shard_img_size * crop_factor ~= resize_size to skip resizing
    crop_factor=0.75,
    always_augment=False,
    # hardware parameters
    mixed_precision=True,
    gpus=2,
    eager=False,  # tf-specific. Enable eager mode. Set True for easier debugging but slower training
    # replication parameters
    random_state=42,  # TODO not yet implemented
):

    # get the image paths, divide into train/val/test if not explicitly passed above
    if catalog is not None:
        assert train_catalog is None
        assert val_catalog is None
        assert test_catalog is None
        train_catalog, hidden_catalog = train_test_split(catalog, train_size=0.7, random_state=random_state)
        val_catalog, test_catalog = train_test_split(hidden_catalog, train_size=1./3., random_state=random_state)
    else:
        assert catalog is None
        assert len(train_catalog) > 0
        assert len(val_catalog) > 0
        assert len(test_catalog) > 0

    # a bit awkward, but I think it is better to have to specify you def. want color than that you def want greyscale
    greyscale = not color
    if greyscale:
        logging.info('Converting images to greyscale before training')
        channels = 1
    else:
        logging.warning(
            'Training on color images, not converting to greyscale')
        channels = 3

    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=schema.label_cols,
        input_size=img_size_to_load,
        make_greyscale=greyscale,
        # False for tfrecords with 0-1 floats, True for png/jpg with 0-255 uints
        # normalise_from_uint8=False
    )

    assert save_dir is not None
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if gpus > 1:
        logging.info('Using distributed mirrored strategy')
        strategy = tf.distribute.MirroredStrategy()  # one machine, one or more GPUs
        # strategy = tf.distribute.MultiWorkerMirroredStrategy()  # one or more machines. Not tested - you'll need to set this up for your own cluster.
        context_manager = strategy.scope()
        logging.info('Replicas: {}'.format(strategy.num_replicas_in_sync))
    else:
        logging.info('Using single or no GPU, not distributed')
        # does nothing, just a convenience for clean code
        context_manager = contextlib.nullcontext()

    train_image_paths = list(train_catalog['file_loc'])
    val_image_paths = list(val_catalog['file_loc'])
    test_image_paths = list(test_catalog['file_loc'])

    example_image_loc = train_image_paths[0]
    file_format = example_image_loc.split('.')[-1]

    # format is [{label_col: 0, label_col: 12}, {label_col: 3, label_col: 14}, ...]
    train_labels = train_catalog[schema.label_cols].to_dict(orient='records')
    val_labels = val_catalog[schema.label_cols].to_dict(orient='records')
    test_labels = test_catalog[schema.label_cols].to_dict(orient='records')

    logging.info('Example path: {}'.format(train_image_paths[0]))
    logging.info('Example labels: {}'.format(train_labels[0]))

    raw_train_dataset = image_datasets.get_image_dataset(
        train_image_paths, file_format, img_size_to_load, batch_size, labels=train_labels, check_valid_paths=True, shuffle=True, drop_remainder=True
    )
    raw_val_dataset = image_datasets.get_image_dataset(
        val_image_paths, file_format, img_size_to_load, batch_size, labels=val_labels, check_valid_paths=True, shuffle=True, drop_remainder=False
    )
    raw_test_dataset = image_datasets.get_image_dataset(
        test_image_paths, file_format, img_size_to_load, batch_size, labels=test_labels, check_valid_paths=True, shuffle=False, drop_remainder=False
    )

    train_dataset = preprocess.preprocess_dataset(
        raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(
        raw_val_dataset, preprocess_config)
    test_dataset = preprocess.preprocess_dataset(
        raw_test_dataset, preprocess_config)

    with context_manager:

        if mixed_precision:
            logging.info(
                'Using mixed precision. \
                Note that this is a global setting (why, tensorflow, why...) \
                and so may affect any future code'
            )
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        model = define_model.get_model(
            output_dim=len(schema.label_cols),
            input_size=img_size_to_load,
            crop_size=int(img_size_to_load * crop_factor),
            resize_size=resize_size,  # ideally, matches crop size
            channels=channels,
            always_augment=always_augment,
            dropout_rate=dropout_rate
        )

        multiquestion_loss = losses.get_multiquestion_loss(
            schema.question_index_groups)
        # SUM reduction over loss, cannot divide by batch size on replicas when distributed training
        # so do it here instead
        def loss(x, y): return multiquestion_loss(x, y) / batch_size

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam()
    )
    model.summary()

    train_config = training_config.TrainConfig(
        log_dir=save_dir,
        epochs=epochs,
        patience=patience
    )

    best_trained_model = training_config.train_estimator(
        model,
        train_config,  # parameters for how to train e.g. epochs, patience
        train_dataset,
        val_dataset,
        eager=eager
    )

    # unsure if this will work
    best_trained_model.evaluate(test_dataset)

    return best_trained_model
