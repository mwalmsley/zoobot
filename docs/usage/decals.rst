.. _reproducing_decals:

Training from Scratch
=====================

This code was used to create the automated classifications for GZ DECaLS.
It can be re-used for new Galaxy Zoo projects or as a baseline or starting point to improve on our performance.

.. note:: 

    If you just want to use the classifier, you don't need to make it from scratch.
    We provide `pretrained weights <https://github.com/mwalmsley/zoobot/tree/main/data>`_.
    Start from these and :ref:`finetune <finetuning>` to your problem.

You will need galaxy images and volunteer classifications.
For GZD-5, these are available at `<https://zenodo.org/record/4196267>`_.
You will also need a fairly good GPU - we used an NVIDIA V100. 
You might get away with a worse GPU by lowering the batch size (we used 128) or the image size, but this may affect performance.

The high-level approach to create a CNN is:

- Define the decision tree asked of volunteers in ``schemas.py``. *Already done for GZD-5 and GZ2.*
- Prepare a catalog with your images and labels (matching the decision tree)
- Create TFRecord shards (groups of images encoded for fast reading) from your catalog with `create_shards.py <https://github.com/mwalmsley/zoobot/blob/main/create_shards.py>`__
- Train the CNN on those shards with `train_model.py <https://github.com/mwalmsley/zoobot/blob/main/train_model.py>`__.

Galaxy Zoo uses a decision tree where the questions asked depend upon the previous answers.
The decision tree is defined under `schemas.py <https://github.com/mwalmsley/zoobot/blob/zoobot/schemas.py>`_ and `label_metadata.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/label_metadata.py>`_.
The GZ2 and GZ DECaLS decision trees are already defined for you; for other projects, you'll need to define your own (it's easy, just follow the same pattern).

Create a catalog with all your labelled galaxies.
The catalog should be a csv file with rows of (unique) galaxies and columns including:

- id_str, a string that uniquely identifies each galaxy (e.g. the iauname)
- file_loc, the absolute location of the galaxy image on disk. This is expected to be a .png of any size, but you could easily extend it for other filetypes if needed.
- a column with the total votes for each label you want to predict, matching the schema (above).  For GZD-5, this is e.g. smooth-or-featured_smooth, smooth-or-featured_featured-or-disk, etc.

It's quite slow to train a model using normal images, and so we first encode them as several TFRecords, a format which is much faster to read.
Make these with `create_shards.py <https://github.com/mwalmsley/zoobot/blob/main/create_shards.py>`__, passing in your catalog location and where the TFRecords should be placed e.g.

.. code-block:: bash

    python create_shards.py --labelled-catalog path/to/my_catalog.csv --shard-dir folder/for/shards --img-size 300  --eval-size 5000

More options are available, and you may need to adjust the label columns; see `create_shards.py <https://github.com/mwalmsley/zoobot/blob/main/create_shards.py>`__.

.. TODO document training_config, link

Now you can train a CNN using those shards. ``training/training_config.py`` has the code to do this. 
Use it in your own code like so:

.. code-block:: python

    from zoobot.training import training_config

    model = define_model.get_model(
      output_dim=len(schema.label_cols),
      input_size=initial_size, 
      crop_size=int(initial_size * 0.75),
      resize_size=resize_size
    )
  
    # dirichlet-multinomial log-likelihood per answer - see the paper for more
    loss = losses.get_multiquestion_loss(schema.question_index_groups)

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
      train_dataset,
      test_dataset
    )


There is a complete working example at `train_model.py <https://github.com/mwalmsley/zoobot/blob/main/train_model.py>`__ which you can copy and adapt.

Once trained, the model can be used to make new predictions on either folders of images (png, jpeg) or TFRecords. For example:

.. code-block:: python

    folder_to_predict = 'folder/with/images'
    file_format = 'png'  # jpg or png supported. FITS is NOT supported (PRs welcome)
    predict_on_images.predict(
        schema=schema,
        file_format=file_format,
        folder_to_predict=folder_to_predict,
        checkpoint_dir=checkpoint_dir,
        save_loc=save_loc,
        n_samples=n_samples,  # number of dropout forward passes
        batch_size=batch_size,
        initial_size=initial_size,
        crop_size=crop_size,
        final_size=final_size
    )

There is a complete working example at `make_predictions.py <https://github.com/mwalmsley/zoobot/blob/main/make_predictions.py>`_.

.. note::

    In the DECaLS paper, we only used galaxies classified in GZD-5 even for questions which did not change between GZD-1/2 and GZD-5.
    It would be straightforward (and appreciated) to retrain the models using GZD-1/2 classifications as well, to improve performance.
