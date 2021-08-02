.. _components:

Components
==========

Training deep learning models involves many steps and choices.
Zoobot aims to provide a simple function for each step, tailored to astronomers classifying galaxies.
You can compose these functions for your own research: either reproducing/improving the DECaLS classifications or finetuning a trained classifier for your own task.

This section explains each of the steps available. 
If you just want to dive in, start with an example script like ``finetune_minimal.py``.

.. _overview_loading:

Loading Data
------------

Zoobot uses Tensorflow to train models. 
Tensorflow expects input data as a `tf.dataset <https://www.tensorflow.org/guide/data>`_ class, and so Zoobot includes utilities to create these.

The easiest way to load data into a tf.dataset is directly from images (png or jpg) on disk.

.. code-block:: python

    from zoobot.data_utils import image_datasets

    paths_train = ['path/to/image_a.png', 'path/to/image_b.png']
    labels_train = [1, 0]
    file_format = 'png'
    requested_img_size = 300  # if images are saved on disk at a different resolution, they will be resized to this resolution
    batch_size = 64

    raw_train_dataset = image_datasets.get_image_dataset(paths_train, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_train)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_val)


Loading images from disk is simple but slow. 
If fast training is important, you can save those images as binary TFRecords.
See :ref:`reproducing DECaLS <reproducing_decals>` for more and `create_shards.py <https://github.com/mwalmsley/zoobot/blob/main/create_shards.py>`_ for a full example.

.. code-block:: python

    shard_config = ShardConfig(shard_dir='directory/tfrecords', size=300)

    shard_config.prepare_shards(
        labelled_catalog,  # pandas.DataFrame. Columns must include 'id_str', 'file_loc' and every item in 'columns_to_save' (labels)
        unlabelled_catalog,  # pandas.DataFrame. Columns must include 'id_str' and 'file_loc'.
        train_test_fraction=0.8,  # 80% train, 20% test
        columns_to_save=['ring']  # labels
    )

The binary TFRecords can then be loaded back quickly during training.

.. code-block:: python

    from zoobot.data_utils import tfrecord_datasets

    train_records = ['directory/tfrecords/s300_shard_0.tfrecord', 'directory/tfrecords/s300_shard_1.tfrecord']
    # similarly for val_records

    raw_train_dataset = tfrecord_datasets.get_dataset(train_records, columns_to_save, batch_size, shuffle=True)
    raw_val_dataset = tfrecord_datasets.get_dataset(val_records, columns_to_save, batch_size, shuffle=False)

.. warning:: 

    Saving TFRecords is slow, fiddly, and takes up more disk space than the original images. 
    Only do so if loading speed is crucial (e.g. you are training many models on hundreds of thousands of images)

.. _overview_preprocessing:

Preprocessing Data
------------------

Images may need preprocessing - deterministic tweaks - before being sent to the model.
For example, images are usually saved as 0-255 integers and should be rescaled to 0-1 floats.

.. code-block:: python

    preprocess_config = preprocess.PreprocessingConfig(
    label_cols=['label'],  # image_datasets.get_image_dataset will put the labels arg under the 'label' key for each batch
    input_size=requested_img_size,
    normalise_from_uint8=True,  # divide by 255
    make_greyscale=True,  # take the mean over RGB channels
    permute_channels=False  # swap channels around randomly (no need when making greyscale anwyay)
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, preprocess_config)

preprocess.PreprocessingConfig is essentially a dict recording your preprocessing choices.
Re-use ``preprocess_config`` to ensure your train, validation, test and ultimately prediction data are all preprocessed the same way.

Training
--------

Zoobot trains the convolutional neural network `EfficientNet <https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html>`_, implemented in `tf.keras <https://www.tensorflow.org/guide/keras/sequential_model>`_.
Random augmentations (crops, flips and rotations) will be applied by the first layers of the network
(using `tf.keras.layers.experimental.preprocessing <https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing>`_).

The exact model and loss to use depend on if you are :ref:`reproducing DECaLS <reproducing_decals>` or :ref:`finetuning <finetuning>`. 
Click each link for a specific guide.

The general steps are the same: define the model architecture, select a loss function and optimizer, configure training options, and begin training.

.. code-block:: 

    model = define_model.get_model(
    ...  # options depend on what you're doing
    )

    model.compile(
    loss=loss,  # loss depends on what you're doing
    optimizer=tf.keras.optimizers.Adam()
    )

    train_config = training_config.TrainConfig(
    log_dir='save/model/here',
    epochs=50,
    patience=10  # early stopping: end training if no improvement for this many epochs
    )

    training_config.train_estimator(
    model, 
    train_config,  # parameters for how to train e.g. epochs, patience
    train_dataset,
    val_dataset
    )

Making Predictions
------------------

You can :ref:`load <overview_loading>`  and :ref:`preprocess <overview_preprocessing>` the prediction data just as for the training and validation data.

Making predictions is then as easy as:

.. code-block:: 

    predictions = model.predict(pred_dataset)

See the end of `finetune_minimal.py <https://github.com/mwalmsley/zoobot/blob/main/finetune_minimal.py>`_ for a complete example.

.. To make life even easier, 

.. .. code-block:: 

..     file_format = 'png'  # jpg or png supported. FITS is NOT supported (PRs welcome)
..     predict_on_images.predict(
..         label_cols=label_cols,
..         file_format=file_format,
..         checkpoint_dir=checkpoint_dir,
..         save_loc=save_loc,
..         n_samples=n_samples,  # number of dropout forward passes
..         batch_size=batch_size,
..         initial_size=initial_size,
..         crop_size=crop_size,
..         resize_size=resize_size,
..         paths_to_predict=list(pd.read_csv('data/decals_dr_full_eval_df.csv')['local_png_loc'].apply(lambda x: x.replace('/data/phys-zooniverse/chri5177/png_native/dr5', '/raid/scratch/walml/galaxy_zoo/decals/png')))
..     )

.. .. code-block:: 

..     predict_on_images.predict(
..         label_cols=label_cols,
..         file_format=file_format,
..         checkpoint_dir=checkpoint_dir,
..         save_loc=save_loc,
..         n_samples=n_samples,  # number of dropout forward passes
..         batch_size=batch_size,
..         initial_size=initial_size,
..         crop_size=crop_size,
..         resize_size=resize_size,
..         folder_to_predict=folder_to_predict,
..         recursive=True  # if you also want to search subfolders, subsubfolders, etc
..     )
