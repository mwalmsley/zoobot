.. _training_with_tensorflow:

Training from Scratch with TensorFlow
=========================================

For training from scratch with either TensorFlow or PyTorch, you should have first defined a schema and a catalog. See the :ref:`Training from Scratch <training_from_scratch>` guide, and then come back here.

.. note:: 

    If you just want to use the classifier, you don't need to make it from scratch.
    We provide :ref:`pretrained weights and precalculated representations <datanotes>`.
    You can even start from these and :ref:`finetune <finetuning_guide>` to your problem.


Creating Shards
-----------------

It's quite slow to train a model using normal images, and so we first encode them as several TFRecords, a format which is much faster to read.
Make these with the functions in `create_shards.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/data_utils/create_shards.py>`__.

For a working example, see `decals_dr5_to_shards.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/decals_dr5_to_shards.py>`__.
You can run this with command-line arguments for your catalog location and where the TFRecords should be placed e.g.

.. code-block:: bash

    python decals_dr5_to_shards.py --labelled-catalog path/to/my_catalog.csv --shard-dir folder/for/shards --img-size 300  --eval-size 5000

More options are available, and you may need to adjust the label columns; see the examples in `decals_dr5_to_shards.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/decals_dr5_to_shards.py>`__.
I like to use these options to make very small shards for quick debugging: ``python create_shards.py --labelled-catalog data/decals/prepared_catalogs/my_subfolder/labelled_catalog.csv --shard-dir data/decals/shards/decals_debug --eval-size 100 --max-labelled 500 --max-unlabelled 300 --img-size 32``.

Training on Shards
--------------------

Now you can train a CNN using those shards. 
`train_with_keras.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/training/train_with_keras.py>`__ has the code to do this.
This has a .train() function with the following arguments:

.. code-block:: python

    from zoobot.tensorflow.training import train_with_keras

    train_with_keras.train(
        # absolutely crucial arguments
        save_dir,  # save model here
        schema,  # answer these questions
        # input data as TFRecords
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
        eager=False  # set True for easier debugging but slower training
    )

Check the function docstring (and comments in the function itself) for further details.

There are two complete working examples which you can copy and adapt. Both scripts are simply convenient command-line wrappers around ``train_with_keras.train``.

`zoobot/tensorflow/examples/train_model_on_shards.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/train_model_on_catalog.py>`__ demonstrates training a model on shards you've created. 
You can use the shards created by the worked example in the section above.

`replication/tensorflow/train_model_on_decals_dr5_splits.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/train_model.py>`__
trains a model on the shards used by W+22a (themselves created by `decals_dr5_to_shards.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/decals_dr5_to_shards.py>`__).
This example is the script used to create the pretrained TensorFlow models shared under :ref:`datanotes`.


Making New Predictions
--------------------------

.. note:: 

    Making new predictions is also demonstrated in the [Google Colab notebook](https://colab.research.google.com/drive/1miKj3HVmt7NP6t7xnxaz7V4fFquwucW2?usp=sharing), which can be run in your browser

Once trained, the model can be used to make new predictions on either folders of images (png, jpeg) or TFRecords. For example:

.. code-block:: python

    from zoobot.predictions import predict_on_dataset

    file_format = 'png'
    unordered_image_paths = predict_on_dataset.paths_in_folder('data/example_images', file_format=file_format, recursive=False)
    # unordered_image_paths = df['paths']   # you might instead just use a catalog

    # Load the images as a tf.data.Dataset, just as for training
    initial_size = 300  # image size the model expects, not size on disk
    batch_size = 64
    raw_image_ds = image_datasets.get_image_dataset([str(x) for x in unordered_image_paths], file_format, initial_size, batch_size)
    preprocessing_config = preprocess.PreprocessingConfig(
        label_cols=[],  # no labels are needed, we're only doing predictions
        input_size=initial_size,
        make_greyscale=True,
        normalise_from_uint8=True
    )
    image_ds = preprocess.preprocess_dataset(raw_image_ds, preprocessing_config)

    model = define_model.load_model(
        checkpoint_loc=checkpoint_loc,  # see data/pretrained_models
        include_top=True,  # finetuning? use False and add your own top
        input_size=initial_size,
        crop_size=crop_size,
        resize_size=resize_size,
        expect_partial=True # hides some warnings
    )

    predict_on_dataset.predict(
        image_ds=image_ds,
        model=model,
        n_samples=n_samples,  # number of dropout forward passes
        label_cols=['ring'],  # used for output csv header only
        save_loc='output/folder/ring_predictions.csv'
    )

There is a complete working example at `make_predictions.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/make_predictions.py>`_.
This example shows how to make predictions on new galaxies (by default), and how to make predictions with the custom finetuned model from ``finetime_minimal.py`` (commented out).
Check out the code to see both versions.

If you'd like to make predictions about a new galaxy problem, for which you don't have tens of thousands of labels, you will want to finetune the model - see the :ref:`Finetuning Guide <finetuning_guide>` 

.. note::

    In the GZ DECaLS paper, we only used galaxies classified in GZD-5 even for questions which did not change between GZD-1/2 and GZD-5.
    In the GZ LegS paper, we train the models using GZD-1/2 and GZD-8 classifications as well.
