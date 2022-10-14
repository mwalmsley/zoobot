.. _training_with_tensorflow:

Training from Scratch with TensorFlow
=========================================

For training from scratch with either TensorFlow or PyTorch, you should have first defined a schema and a catalog. See the :ref:`Training from Scratch <training_from_scratch>` guide, and then come back here.

.. note:: 

    If you just want to use the classifier, you don't need to make it from scratch.
    We provide :ref:`pretrained weights and precalculated representations <datanotes>`.
    You can even start from these and :ref:`finetune <finetuning_guide>` to your problem.

Training on Catalog of Images
-------------------------------

`train_with_keras.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/training/train_with_keras.py>`__ has a .train() function with the following arguments:

.. code-block:: python

    from zoobot.tensorflow.training import train_with_keras

    train_with_keras.train(
        save_dir,  # save model here
        schema,  # answer these questions
        # input data - specify *either* catalog (to be split) or the splits themselves
        catalog=None,
        train_catalog=None,
        val_catalog=None,
        test_catalog=None,
        # model training parameters
        batch_size=256,
        dropout_rate=0.2,
        epochs=1000,
        patience=8,
        # augmentation parameters
        color=False,
        img_size_to_load=300,  # resizing on load *before* augmentations, will skip if given same size as on disk
        resize_size=224,  # resizing *during* augmentations, will skip if given appropriate crop
        # ideally, set shard_img_size * crop_factor ~= resize_size to skip resizing
        crop_factor=0.75,
        always_augment=False,
        # hardware parameters
        mixed_precision=True,
        gpus=2,
        eager=False,  # Enable eager mode. Set True for easier debugging but slower training
    )

Check the function docstring (and comments in the function itself) for further details.

There are two complete working examples which you can copy and adapt. Both scripts are simply convenient command-line wrappers around ``train_with_keras.train``.

`zoobot/tensorflow/examples/train_model_on_catalog.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/train_model_on_catalog.py>`__ demonstrates training a model on an arbitrary catalog.
This might be useful if you're training from scratch on your own data.

`replication/tensorflow/train_model_on_decals_dr5_splits.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/train_model.py>`__
trains a model on the DECaLS DR5 catalog, as was done by W+22a.
This example is the script used to create the pretrained TensorFlow models shared under :ref:`datanotes`.


Making New Predictions
--------------------------

.. note:: 

    Making new predictions is also demonstrated in the [Google Colab notebook](https://colab.research.google.com/drive/1miKj3HVmt7NP6t7xnxaz7V4fFquwucW2?usp=sharing), which can be run in your browser

Once trained, the model can be used to make new predictions on either folders of images (png, jpeg) or catalogs listing images. For example:

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
    In the GZ DESI paper (upcoming), we train the models using GZD-1/2 and GZD-8 classifications as well.
