.. _training_with_pytorch:

Training from Scratch with PyTorch
=========================================

For training from scratch with either TensorFlow or PyTorch, you should have first defined a schema and a catalog. See the :ref:`Training from Scratch <training_from_scratch>` guide, and then come back here.

Creating a DataModule
----------------------

With the PyTorch version, you need to define a `PyTorch Lightning DataModule <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_ that describes how to load the images listed in your catalog and how to divide them into train/validation/test sets. 
See `zoobot/pytorch/datasets/decals_dr8.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/datasets/decals_dr8.py>`_ for a working example to adjust. 

.. note:: 

    There is no need to create TFRecord shards of your images when training with PyTorch

Training
---------

Now you can train a CNN using those shards. `zoobot/pytorch/training/train_with_pytorch_lightning.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/training/train_with_pytorch_lightning.py>`__. has the code to do this. 
This has a .train() function with the following arguments:

.. code-block:: python

    from zoobot.pytorch.training import train_with_pytorch_lightning

    train_with_pytorch_lightning.train(
        # absolutely crucial arguments
        save_dir,  # save model here
        schema,  # answer these questions
        # input data - specify *either* catalog (to be split) or the splits themselves
        catalog=None,
        train_catalog=None,
        val_catalog=None,
        test_catalog=None,
        # model training parameters
        model_architecture='efficientnet',  # or resnet_detectron, or resnet_torchvision
        batch_size=256,
        epochs=1000,
        patience=8,
        # augmentation parameters
        color=False,
        resize_size=224,
        crop_scale_bounds=(0.7, 0.8),  # random crop factor
        crop_ratio_bounds=(0.9, 1.1),  # aspect ratio of that crop
        # hardware parameters
        nodes=1,
        gpus=2,
        num_workers=4,
        prefetch_factor=4,
        mixed_precision=False,
        # replication parameters
        random_state=42,
        wandb_logger=None

Check the function docstring (and comments in the function itself) for further details.

There are two complete working examples which you can copy and adapt. Both scripts are simply convenient command-line wrappers around ``train_with_keras.train``.

`zoobot/pytorch/examples/train_model_on_catalog.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/train_model_on_catalog.py>`__ demonstrates training a model on a volunteer catalog. 
This example provides the whole catalog to ``train_with_pytorch_lightning.train``, which then automatically splits it into train/validation/test subsets.
You will need to provide your own catalog. I will add an example volunteer catalog to the ``data`` folder at some point TODO

`replication/pytorch/train_model_on_decals_dr5_splits.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/train_model.py>`__
demonstrates training a model on a volunteer catalog already split into train/validation/test subsets, but is otherwise very similar.
This example is the script used to create the pretrained models shared under :ref:`datanotes`.


Predictions
------------

Once trained, the model can be used to make new predictions on either folders of images (png, jpeg) or TFRecords. 
I have not yet made an example for this (TODO), but the process is very similar to the TensorFlow version:
load the model from the PyTorch Lightning checkpoint, load the DataModule to predict on and call model.predict(datamodule)

.. note::

    In the GZ DECaLS paper, we only used galaxies classified in GZD-5 even for questions which did not change between GZD-1/2 and GZD-5.
    In the GZ LegS paper, we train the models using GZD-1/2 and GZD-8 classifications as well.
