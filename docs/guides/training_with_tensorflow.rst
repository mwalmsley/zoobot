.. _training_with_tensorflow:

Training from Scratch with TensorFlow
=========================================

For training from scratch with either TensorFlow or PyTorch, you should have first defined a schema and a catalog. See the :ref:`Training from Scratch <training_from_scratch>` guide, and then come back here.

Creating Shards
---------------

It's quite slow to train a model using normal images, and so we first encode them as several TFRecords, a format which is much faster to read.
Make these with the helper functions in `create_shards.py <https://github.com/mwalmsley/zoobot/blob/pytorch/zoobot/tensorflow/data_utils/create_shards.py>`__.
These expect a csv listing volunteer votes and image paths.
See `decals_dr5_to_shards.py <https://github.com/mwalmsley/zoobot/blob/pytorch/zoobot/tensorflow/examples/decals_dr5_to_shards.py>`__ for a full working example, run like:

.. code-block:: bash

    python decals_dr5_to_shards.py --labelled-catalog path/to/my_catalog.csv --shard-dir folder/for/shards --img-size 300  --eval-size 5000

More options are available, and you may need to adjust the label columns; see the example for more details.

I like to use these options to make very small shards for quick debugging: 

.. code-block:: bash

    python decals_dr5_to_shards.py \
        --labelled-catalog data/decals/prepared_catalogs/my_subfolder/labelled_catalog.csv \
        --shard-dir data/decals/shards/decals_debug \
        --img-size 300 \
        --eval-size 100 \
        --max-labelled 500 \
        --max-unlabelled 300 \
        --img-size 32


Training on Shards
------------------

Now you can train a CNN using those shards. `training_config.py <https://github.com/mwalmsley/zoobot/blob/pytorch/zoobot/tensorflow/training/training_config.py>`__. has the code to do this. 
Use it in your own code like so:

.. code-block:: python

    from zoobot.training import training_config
    from zoobot.estimators import define_model
    from zoobot import losses

    model = define_model.get_model(
      output_dim=len(schema.label_cols),  # see train_model.py
      input_size=initial_size, 
      crop_size=int(initial_size * 0.75),  # small zoom
      resize_size=resize_size,
      channels=1  # this makes images greyscale before training (recommended)
    )
  
    # dirichlet-multinomial log-likelihood per answer
    # see the GZ DECaLS paper for more
    multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
    loss = lambda x, y: multiquestion_loss(x, y) / batch_size
    # dividing by batch size here, instead of automatically, helps do distributed training

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam()
    )

    train_config = training_config.TrainConfig(
      log_dir='save/model/here',
      epochs=50,
      patience=10
    )

    training_config.train_estimator(
      model, 
      train_config,  # parameters for how to train e.g. epochs, patience
      train_dataset,  # tf.data.Dataset, see train_model.py
      test_dataset,  # similarly
      eager=True  # slow, helpful for debugging. Set False when happy.
    )


There is a complete working example at `train_model.py <https://github.com/mwalmsley/zoobot/blob/pytorch/zoobot/tensorflow/examples/train_model.py>`__ which you can copy and adapt.
I've skipped loading the training and test datasets in the above, for clarity - see the worked example.

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

There is a complete working example at `make_predictions.py <https://github.com/mwalmsley/zoobot/blob/pytorch/zoobot/tensorflow/examples/make_predictions.py>`_.
This example shows how to make predictions on new galaxies (by default), and how to make predictions with the custom finetuned model from ``finetime_minimal.py`` (commented out).
Check out the code to see both versions.

.. note::

    In the GZ DECaLS paper, we only used galaxies classified in GZD-5 even for questions which did not change between GZD-1/2 and GZD-5.
    In the GZ LegS paper, we train the models using GZD-1/2 and GZD-8 classifications as well.
