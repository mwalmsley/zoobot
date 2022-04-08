
import os
import logging
import contextlib

import tensorflow as tf


from zoobot.tensorflow.data_utils import tfrecord_datasets
from zoobot.tensorflow.training import training_config, losses
from zoobot.tensorflow.estimators import preprocess, define_model


def train(
    # absolutely crucial arguments
    save_dir,  # save model here
    schema,  # answer these questions
    # input data as TFRecords - TODO will be replaced by catalogs
    train_records,
    test_records,
    shard_img_size=300,
    # model training parameters
    # only EfficientNet is currenty implemented
    batch_size=256,
    epochs=1000,
    patience=8,
    dropout_rate=0.2,
    # augmentation parameters
    color=False,
    resize_size=224,
    # ideally, set shard_img_size * crop_factor ~= resize_size to skip resizing
    crop_factor=0.75,
    always_augment=False,
    # hardware parameters
    gpus=2,
    eager=False,  # set True for easier debugging but slower training
    # replication parameters
    random_state=42,  # TODO not yet implemented
):

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
        input_size=shard_img_size,
        make_greyscale=greyscale,
        # False for tfrecords with 0-1 floats, True for png/jpg with 0-255 uints
        normalise_from_uint8=False
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
        logging.info('Using single GPU, not distributed')
        # does nothing, just a convenience for clean code
        context_manager = contextlib.nullcontext()

    raw_train_dataset = tfrecord_datasets.get_tfrecord_dataset(
        train_records, schema.label_cols, batch_size, shuffle=True, drop_remainder=True)
    raw_test_dataset = tfrecord_datasets.get_tfrecord_dataset(
        test_records, schema.label_cols, batch_size, shuffle=False, drop_remainder=True)

    train_dataset = preprocess.preprocess_dataset(
        raw_train_dataset, preprocess_config)
    test_dataset = preprocess.preprocess_dataset(
        raw_test_dataset, preprocess_config)

    with context_manager:

        model = define_model.get_model(
            output_dim=len(schema.label_cols),
            input_size=shard_img_size,
            crop_size=int(shard_img_size * crop_factor),
            resize_size=resize_size,
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

    # inplace on model
    training_config.train_estimator(
        model,
        train_config,  # parameters for how to train e.g. epochs, patience
        train_dataset,
        test_dataset,
        eager=eager
    )
