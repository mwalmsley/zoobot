.. _how_the_code_fits_together:

How the Code Fits Together
===========================

The Zoobot package has many classes and methods.
This guide aims to be a map summarising how they fit together.

.. note:: For simplicity, we will only consider the PyTorch version (see :ref:`pytorch_or_tensorflow`).

Defining PyTorch Models
-------------------------

The deep learning part is the simplest piece. 
``define_model.py`` has functions to that define pure PyTorch ``nn.Modules`` (a.k.a. models).

Encoders (a.k.a. models that take an image and compress it to a representation vector) are defined using the third party library ``timm``.
Specifically, ``timm.create_model(architecture_name)`` is used to get the EfficientNet, ResNet, ViT, etc. architectures used to encode our galaxy images.
This is helpful because defining complicated architectures becomes someone else's job (thanks, Ross Wightman!) 

Heads (a.k.a. models that take a representation vector and make a prediction) are defined using ``torch.nn.Sequential``. 
The function :func:`zoobot.pytorch.estimators.define_model.get_pytorch_dirichlet_head`, for example, returns the custom head used to predict vote counts (see :ref:`training_on_vote_counts`).

The encoders and heads in ``define_model.py`` are used for both training from scratch and finetuning

Training with PyTorch Lightning
--------------------------------

PyTorch requires a lot of boilerplate code to train models, especially at scale (e.g. multi-node, multi-GPU).
We use PyTorch Lightning, a third party wrapper API, to make this boilerplate code someone else's job as well.

The core Zoobot classes you'll use - :class:`ZoobotTree <zoobot.pytorch.estimators.define_model.ZoobotTree>`, :class:`FinetuneableZoobotClassifier <zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier>` and :class:`FinetuneableZoobotTree <zoobot.pytorch.training.finetune.FinetuneableZoobotTree>` - 
are all "LightningModule" classes.
These classes have (custom) methods like ``training_step``, ``validation_step``, etc., which specify what should happen at each training stage.

:class:`FinetuneableZoobotClassifier <zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier>` and :class:`FinetuneableZoobotTree <zoobot.pytorch.training.finetune.FinetuneableZoobotTree>`
are actually subclasses of a non-user-facing abstract class, :class:`FinetuneableZoobotAbstract <zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract>`.
:class:`FinetuneableZoobotAbstract <zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract>` has specifying how to finetune a general PyTorch model,
which `FinetuneableZoobotClassifier <zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier>` and :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotTree` inherit.

:class:`ZoobotTree <zoobot.pytorch.estimators.define_model.ZoobotTree>` is similar to :class:`FinetuneableZoobotAbstract <zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract>` but has methods for training from scratch.

Some generic methods (like logging) are defined in ``define_model.py`` and called by both :class:`ZoobotTree <zoobot.pytorch.estimators.define_model.ZoobotTree>` and :class:`FinetuneableZoobotAbstract <zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract>`

LightningModules can be passed to a Lightning ``Trainer`` object. This handles running the training in practice (e.g. how to distribute training onto a GPU, how many epochs to run, etc.).

So when we do:

.. code-block:: python

    model = FinetuneableZoobotTree(...)
    trainer = get_trainer(...)
    trainer.fit(model, datamodule)

We are:

- Defining a PyTorch encoder and head (inside ``FinetuneableZoobotTree``)
- Wrapping them in a LightningModule specifying how to train them (``FinetuneableZoobotTree``)
- Fitting the LightningModule using Lighting's ``Trainer`` class

Slightly confusingly, Lightning's ``Trainer`` can also be used to make predictions:

.. code-block:: python

    trainer.predict(model, datamodule)

and that's how we make predictions with :func:`zoobot.pytorch.predictions.predict_on_catalog.predict`.
  
Loading Data
--------------------------

You might notice ``datamodule`` in the examples above. 
Zoobot often includes code like:

.. code-block:: python

    from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

    datamodule = GalaxyDataModule(
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=batch_size,
        # ... 
    )

Note the import - Zoobot actually doesn't have any code for loading data! 
That's in the separate repository `mwalmsley/galaxy-datasets <https://github.com/mwalmsley/galaxy-datasets/>`.

``galaxy-datasets`` has custom code to turn catalogs of galaxies into the ``LightningDataModule``s that Lightning `expects https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html<>`_.
These ``LightningDataModule``s themselves have attributes like ``.train_dataloader()`` and ``.predict_dataloader()`` that Lightning's ``Trainer`` object uses to demand data when training, making predictions, and so forth.

As you can see, there's quite a few layers (pun intended) to training Zoobot models. But we hope this setup is both simple to use and easy to extend, whichever (PyTorch) frameworks you're using.
