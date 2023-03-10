import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
import logging
import contextlib
from sklearn.model_selection import train_test_split

import tensorflow as tf
import wandb  # for direct manual hparam logging
from zoobot.tensorflow.training import training_config, losses, custom_metrics
from zoobot.tensorflow.estimators import define_model
from galaxy_datasets.tensorflow import get_image_dataset, add_transforms_to_dataset
from galaxy_datasets.transforms import default_transforms


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
    always_augment=False,  # TODO deprecated for now
    # TODO specify test_time_dropout
    # hardware parameters
    mixed_precision=True,
    gpus=2,
    eager=False,  # tf-specific. Enable eager mode. Set True for easier debugging but slower training
    check_valid_paths=True,  # checks all images exist. Can disable for start speed on large datasets (100k+)
    # replication parameters
    random_state=42  # TODO not yet implemented except for catalog split (not used in benchmarks)
) -> tf.keras.Model:
    """
    Train a Zoobot (EfficientNetB0) model from scratch.
    Note that the TensorFlow Zoobot version is not actively developed.
    We suggest the PyTorch version instead.

    Args:
        save_dir (str): directory to save the trained model and logs
        train_catalog (pd.DataFrame, optional): Catalog of galaxies for training. Must include `id_str`, `file_loc`, and label_cols columns. Defaults to None.
        val_catalog (pd.DataFrame, optional): As above, for validation. Defaults to None.
        test_catalog (pd.DataFrame, optional): As above, for testing. Defaults to None.
        architecture_name (str, optional): Specifies architecture to use. B0 by default. Defaults to 'efficientnet'.
        dropout_rate (float, optional): Prob. of dropout prior to output layer. Defaults to 0.2.
        epochs (int, optional): Max epochs to train. Defaults to 1000.
        patience (int, optional): Max epochs to wait for loss improvement before cancelling training. Defaults to 8.
        color (bool, optional): Train with RGB images. Defaults to False.
        requested_img_size (int, optional): Size to load images from disk (i.e. to resize before transforms). Defaults to None.
        crop_scale_bounds (tuple, optional): Zoom fraction for random crops. Defaults to (0.7, 0.8).
        crop_ratio_bounds (tuple, optional): Aspect ratio for random crops. Defaults to (0.9, 1.1).
        resize_after_crop (int, optional): Size to input images to network (i.e. to resize after transforms). Defaults to 224.
        always_augment (bool, optional): Use augmentations at test time. Defaults to False.
        gpus (int, optional): Num. gpus to use. Defaults to 2.
        eager (bool, optional): Use eager execution (good for debugging but much slower). Defaults to False.

    Returns:
        tf.keras.Model: Trained Zoobot model.
    """

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
        # each GPU will calculate loss (hence gradients) for that device's sub-batch
        # within that sub-batch, loss uses tf.keras.reduction setting
        # I chose NONE (i.e. no reduction) and do subbatch_loss/full_batch_size (not sub-batch size)
        # this means the absolute loss and summed gradients don't change with more devices
        # TODO blog post on this    
        assert strategy.num_replicas_in_sync == gpus
    else:
        logging.info('Using single or no GPU, not distributed')
        # does nothing, just a convenience for clean code
        context_manager = contextlib.nullcontext()

    train_image_paths = list(train_catalog['file_loc'])
    val_image_paths = list(val_catalog['file_loc'])
    test_image_paths = list(test_catalog['file_loc'])

    # format is [{label_col: 0, label_col: 12}, {label_col: 3, label_col: 14}, ...]
    train_labels = train_catalog[schema.label_cols].values
    val_labels = val_catalog[schema.label_cols].values
    test_labels = test_catalog[schema.label_cols].values

    logging.info('Example path: {}'.format(train_image_paths[0]))
    logging.info('Example labels: {}'.format(train_labels[0]))

    logging.info(f'Will check if paths valid: {check_valid_paths}')
    train_dataset = get_image_dataset(
        train_image_paths, labels=train_labels, requested_img_size=requested_img_size, check_valid_paths=check_valid_paths, greyscale=greyscale
    )
    val_dataset = get_image_dataset(
        val_image_paths, labels=val_labels, requested_img_size=requested_img_size, check_valid_paths=check_valid_paths, greyscale=greyscale
    )
    test_dataset = get_image_dataset(
        test_image_paths, labels=test_labels, requested_img_size=requested_img_size, check_valid_paths=check_valid_paths, greyscale=greyscale
    )

    # specify augmentations
    train_transforms = default_transforms(
        # no need to specify greyscale here
        # tensorflow will greyscale in get_image_dataset i.e. on load, while pytorch doesn't so needs specifying here
        # may refactor to avoid inconsistency 
        crop_scale_bounds=crop_scale_bounds,
        crop_ratio_bounds=crop_ratio_bounds,
        resize_after_crop=resize_after_crop
    )
    # TODO should be clearer, not magic 1's (e.g. default_train_transforms, etc.)
    # TODO if always_augment, use train augments at test time (TODO rename this too)
    inference_transforms = default_transforms(
        crop_scale_bounds=(1., 1.),
        crop_ratio_bounds=(1., 1.),
        resize_after_crop=resize_after_crop
    )
    # apply augmentations
    train_dataset = add_transforms_to_dataset(train_dataset, train_transforms)
    # if always_augment:
        # logging.warning('always_augment=True, applying augmentations to val and test datasets')
    val_dataset = add_transforms_to_dataset(val_dataset, inference_transforms)
    test_dataset = add_transforms_to_dataset(test_dataset, inference_transforms)

    # batch, shuffle, prefetch
    train_dataset = train_dataset.shuffle(5000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)

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

        # reduction=None will give per-example loss. Still summed (probability-multiplied) across questions.
        multiquestion_loss = losses.get_multiquestion_loss(
            schema.question_index_groups,
            sum_over_questions=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        # NONE reduction over loss, to be clear about how it's reduced vs. batch size
        # get the average loss, averaging over subbatch size (as TF *sums* gradients)
        def loss(x, y): return tf.reduce_sum(multiquestion_loss(x, y)) / (batch_size/gpus)        
        """
        TF actually has a built-in for this which just automatically gets num_replicas and does 

        per_replica_batch_size = per_example_loss.shape[0]
        global_batch_size = per_replica_batch_size * num_replicas
        return reduce_sum(per_example_loss) / global_batch_size

        but it's simple enough here that I'll just do it explicitly
        """
        # def loss(x, y): return tf.nn.compute_average_loss(per_example_loss=multiquestion_loss(x, y), global_batch_size=batch_size)  


        # be careful to define this within the context_manager, so it is also mirrored if on multi-gpu
        extra_metrics = [
            # this currently only works on 1 GPU - see Keras issue
            # custom_metrics.LossPerQuestion(
            #     name='loss_per_question',
            #     question_index_groups=schema.question_index_groups
            # )
        ]

    # https://docs.wandb.ai/guides/track/config#efficient-initialization
    if wandb.run is not None:  # user might not be using wandb
        wandb.config.update({
            'random_state': random_state,
            'epochs': epochs,
            'gpus': gpus,
            'precision': mixed_precision,
            'batch_size': batch_size,
            'greyscale': not color,
            'crop_scale_bounds': crop_scale_bounds,
            'crop_ratio_bounds': crop_ratio_bounds,
            'resize_after_crop': resize_after_crop,
            'framework': 'tensorflow',
            # tf doesn't automatically log model init args
            'architecture_name': architecture_name,  # only EfficientNet is currenty implemented
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            # TODO drop_connect_rate not implemented
            'epochs': epochs,
            'patience': patience
        })


    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999),
        metrics=extra_metrics,
        jit_compile=False  # don't use XLA, it fails on multi-GPU. Might consider on one GPU.
    )
    model.summary()

    trainer = training_config.Trainer(
        # parameters for how to train e.g. epochs, patience
        log_dir=save_dir,
        epochs=epochs,
        patience=patience
    )

    best_trained_model = trainer.fit(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        eager=eager,
        verbose=1
    )

    return best_trained_model
