
Loading Data
--------------------------

Using GalaxyDataModule
=========================

Zoobot often includes code like:

.. code-block:: python

    from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

    datamodule = GalaxyDataModule(
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=batch_size,
        label_cols=['is_cool_galaxy']
        # ... 
    )

Note the import - Zoobot actually doesn't have any code for loading data! 
That's in the separate repository `mwalmsley/galaxy-datasets <https://github.com/mwalmsley/galaxy-datasets/>`_.

``galaxy-datasets`` has custom code to turn catalogs of galaxies into the ``LightningDataModule`` that Lightning `expects <https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html>`_.
Each ``LightningDataModule`` has attributes like ``.train_dataloader()`` and ``.predict_dataloader()`` that Lightning's ``Trainer`` object uses to demand data when training, making predictions, and so forth.

You can pass ``GalaxyDataModule`` train, val, test and predict catalogs. Each catalog needs the columns:

* ``file_loc``: the path to the image file
* ``id_str``: a unique identifier for the galaxy
* plus any columns for labels, which you will specify with ``label_cols``. Setting ``label_cols=None`` will load the data without labels (returning batches of (image, id_str)).

``GalaxyDataModule`` will load the images from disk and apply any transformations you specify. Specify transforms one of three ways:

* through the `default arguments <https://github.com/mwalmsley/galaxy-datasets/blob/main/galaxy_datasets/pytorch/galaxy_datamodule.py>`_ of ``GalaxyDataModule`` (e.g. ``GalaxyDataModule(resize_after_crop=(128, 128))``)
* through a torchvision or albumentations ``Compose`` object e.g. ``GalaxyDataModule(custom_torchvision_transforms=Compose([RandomHorizontalFlip(), RandomVerticalFlip()]))``
* through a tuple of ``Compose`` objects. The first element will be used for the train dataloaders, and the second for the other dataloaders.

Using the default arguments is simplest and should work well for loading Galaxy-Zoo-like ``jpg`` images. Passing Compose objects offers full customization (short of writing your own ``LightningDataModule``). On that note...

I Want To Do It Myself
========================

Using ``galaxy-datasets`` is optional. Zoobot is designed to work with any PyTorch ``LightningDataModule`` that returns batches of (images, labels). 
And advanced users can pass data to Zoobot's encoder however they like (see :doc:`advanced_finetuning`).

Images should be PyTorch tensors of shape (batch_size, channels, height, width).
Values should be floats normalized from 0 to 1 (though in practice, Zoobot can handle other ranges provided you use end-to-end finetuning).
If you are presenting flux values, you should apply a dynamic range rescaling like ``np.arcsinh`` before normalizing to [0, 1].
Galaxies should appear large and centered in the image.
