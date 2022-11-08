import os
import logging
import contextlib
from random import random
from sklearn.model_selection import train_test_split

import tensorflow as tf

from zoobot.tensorflow.training import training_config, losses, custom_metrics
from zoobot.tensorflow.estimators import define_model
from galaxy_datasets.tensorflow import galaxy_dataset, augmentations

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
    architecture_name='efficientnet',  # only EfficientNet is currenty implemented
    batch_size=128,
    dropout_rate=0.2,
    # TODO drop_connect_rate not implemented
    epochs=1000,
    patience=8,
    # augmentation parameters
    color=False,
    requested_img_size=None,
    crop_scale_bounds=(0.7, 0.8),
    crop_ratio_bounds=(0.9, 1.1),
    resize_after_crop=224,
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

    assert save_dir is not None
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if gpus > 1:
        logging.info('Using distributed mirrored strategy')
        strategy = tf.distribute.MirroredStrategy()  # one machine, one or more GPUs
        # strategy = tf.distribute.MultiWorkerMirroredStrategy()  # one or more machines. Not tested - you'll need to set this up for your own cluster.
        context_manager = strategy.scope()
        logging.info('Replicas: {}'.format(strategy.num_replicas_in_sync))
          # MirroredStrategy causes loss to decrease by factor of num_gpus.
          # Multiply by gpu_loss_factor to keep loss consistent.
        gpu_loss_factor = gpus
    else:
        logging.info('Using single or no GPU, not distributed')
        # does nothing, just a convenience for clean code
        context_manager = contextlib.nullcontext()
        gpu_loss_factor = 1  # do nothing

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

    train_dataset = galaxy_dataset.get_image_dataset(
        train_image_paths, file_format, labels=train_labels, requested_img_size=requested_img_size, check_valid_paths=True, greyscale=greyscale
    )
    val_dataset = galaxy_dataset.get_image_dataset(
        val_image_paths, file_format, labels=val_labels, requested_img_size=requested_img_size, check_valid_paths=True, greyscale=greyscale
    )
    test_dataset = galaxy_dataset.get_image_dataset(
        test_image_paths, file_format, labels=test_labels, requested_img_size=requested_img_size, check_valid_paths=True, greyscale=greyscale
    )

    # specify augmentations
    transforms = augmentations.default_albumentation_transforms(
        # no need to specify greyscale here, may refactor 
        crop_scale_bounds=crop_scale_bounds,
        crop_ratio_bounds=crop_ratio_bounds,
        resize_after_crop=resize_after_crop
    )
    # apply augmentations
    train_dataset = augmentations.add_augmentations_to_dataset(train_dataset, transforms)
    if always_augment:
        logging.warning('always_augment=True, applying augmentations to val and test datasets')
        val_dataset = augmentations.add_augmentations_to_dataset(val_dataset, transforms)
        test_dataset = augmentations.add_augmentations_to_dataset(test_dataset, transforms)

    with context_manager:

        if mixed_precision:
            logging.info(
                'Using mixed precision. \
                Note that this is a global setting (why, tensorflow, why...) \
                and so may affect any future code'
            )
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # TODO use architecture_name here

        model = define_model.get_model(
            output_dim=len(schema.label_cols),
            input_size=resize_after_crop,
            channels=channels,
            dropout_rate=dropout_rate
        )

        multiquestion_loss = losses.get_multiquestion_loss(
            schema.question_index_groups)
        # SUM reduction over loss, cannot divide by batch size on replicas when distributed training
        # so do it here instead
        def loss(x, y): return gpu_loss_factor * multiquestion_loss(x, y) / batch_size

        # be careful to define this within the context_manager, so it is also mirrored if on multi-gpu
        extra_metrics = [
            # custom_metrics.LossPerQuestion(
            #     name='loss_per_question',
            #     question_index_groups=schema.question_index_groups
            # )
        ]

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=extra_metrics
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
