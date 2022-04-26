.. _overview_components:

Components
==========

Training deep learning models involves many steps and choices.
Zoobot aims to provide a simple component for each step, tailored to astronomers classifying galaxies.
You can compose these components together for your own research.

This section explains each of the main components available. 
If you just want to dive in, start with an example script like ``finetune_minimal.py``.


- :ref:`Loading Data <overview_loading>`
- :ref:`Preprocessing Data <overview_preprocessing>`
- :ref:`Training <overview_training>`
- :ref:`Predictions and Representations <overview_predictions>`

.. - :ref:`Fine-tuning <overview_finetuning>`

You can put these together to solve tasks.
For example, you might replicate the GZ DECaLS classifier (loading, preprocesssing, training),
calculate new representations on new galaxies (predictions and representations),
and then use those representations to cluster the galaxies (with scikit-learn etc.)

You can see practical guides to typical tasks under Guides on the left hand sidebar.

.. _overview_loading:

Loading Data
------------

Zoobot uses Tensorflow to train models. 
Tensorflow expects input data as a `tf.dataset <https://www.tensorflow.org/guide/data>`_ class, and so Zoobot includes utilities to create these.

The easiest way to load data into a tf.dataset is directly from images (png or jpg) on disk.

.. code-block:: python

    from zoobot.data_utils import image_datasets

    paths_train = ['path/to/image_a.jpg', 'path/to/image_b.jpg']
    labels_train = [1, 0]
    file_format = 'jpg'
    requested_img_size = 300  # if images are saved on disk at a different resolution, they will be resized to this resolution
    batch_size = 64

    raw_train_dataset = image_datasets.get_image_dataset(paths_train, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_train)
    raw_val_dataset = image_datasets.get_image_dataset(paths_val, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size, labels=labels_val)


You can save those images as binary TFRecords, which may be faster to load (provided you have an excellent disk read speed).
See :ref:`Training from Scratch <training_from_scratch>` for more.

.. code-block:: python

    shard_config = ShardConfig(shard_dir='directory/tfrecords', size=300)

    shard_config.prepare_shards(
        labelled_catalog,  # pandas.DataFrame. Columns must include 'id_str', 'file_loc' and every item in 'columns_to_save' (labels)
        unlabelled_catalog,  # pandas.DataFrame. Columns must include 'id_str' and 'file_loc'.
        train_test_fraction=0.8,  # 80% train, 20% test
        columns_to_save=['ring']  # labels
    )

The binary TFRecords can then be loaded back during training.

.. code-block:: python

    from zoobot.data_utils import tfrecord_datasets

    train_records = ['directory/tfrecords/s300_shard_0.tfrecord', 'directory/tfrecords/s300_shard_1.tfrecord']
    # similarly for val_records

    raw_train_dataset = tfrecord_datasets.get_dataset(train_records, columns_to_save, batch_size, shuffle=True)
    raw_val_dataset = tfrecord_datasets.get_dataset(val_records, columns_to_save, batch_size, shuffle=False)

.. warning:: 

    Saving TFRecords is slow, fiddly, and takes up more disk space than the original images. 
    Only do so if loading speed is crucial (e.g. you are training many models on hundreds of thousands of images)
    Also note that the TFRecords may be slower to load than the original images, depending on your disk read speed, because they are much larger than e.g. compressed jpegs.
    TFRecords will be deprecated in future!

.. _overview_preprocessing:

Preprocessing Data
------------------

Images may need preprocessing - deterministic tweaks - before being sent to the model.
For example, images are usually saved as 0-255 integers and should be rescaled to 0-1 floats.

With Tensorflow, we provide functions for this under ``zoobot/tensorflow/preprocess``.

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

With PyTorch, preprocessing happens in the DataModule you define. 
Personally, I find this a little simpler.
See `zoobot/pytorch/datasets/decals_dr8.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/datasets/decals_dr8.py>`_ for a working example to adjust. 

.. _overview_training:

Training
--------

Zoobot trains the convolutional neural network `EfficientNet <https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html>`_.

The exact model and loss to use depend on if you are :ref:`reproducing DECaLS <training_from_scratch>` or :ref:`finetuning <finetuning_guide>`. 
Click each link for a specific guide.

With Tensorflow, training is done by `tf.keras <https://www.tensorflow.org/guide/keras/sequential_model>`_.
Random augmentations (crops, flips and rotations) will be applied by the first layers of the network
(using `tf.keras.layers.experimental.preprocessing <https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing>`_).

With PyTorch, training is done by `PyTorch Lightning <https://www.pytorchlightning.ai/>`_.
Random augmentations are applied by specifing the list of ``transforms`` within your DataModule (again, see `zoobot/pytorch/datasets/decals_dr8.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/datasets/decals_dr8.py>`).

.. note:: 

    The PyTorch version of Zoobot also includes ResNet50 architecture options, which perform a little worse but are a common benchmark - see :ref:`datanotes`.


.. _overview_predictions:


Predictions and Representations
-------------------------------

You can :ref:`load <overview_loading>`  and :ref:`preprocess <overview_preprocessing>` the prediction data just as for the training and validation data.

Making predictions is then as easy as:

.. code-block:: 

    # the API is the same for TensorFlow and PyTorch, happily
    predictions = model.predict(pred_dataset)

See the end of `finetune_minimal.py <https://github.com/mwalmsley/zoobot/blob/main/finetune_minimal.py>`_ for a complete (TensorFlow) example.

