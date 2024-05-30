.. _how_the_code_fits_together:

How the Code Fits Together
===========================

The Zoobot package has many classes and methods.
This guide aims to be a map summarising how they fit together.

The Map
-------------------------

The Zoobot package has two roles:

1. **Finetuning**: ``pytorch/training/finetune.py`` is the heart of the package. You will use these classes to load pretrained models and finetune them on new data.
2. **Training from Scratch** ``pytorch/estimators/define_model.py`` and ``pytorch/training/train_with_pytorch_lightning.py`` create and train the Zoobot models from scratch. These are *not required* for finetuning and will eventually be migrated out.

Let's zoom in on the finetuning part.

Finetuning with Zoobot Classes
--------------------------------


There are three Zoobot classes for finetuning:

1. :class:`FinetuneableZoobotClassifier <zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier>` for classification tasks (including multi-class). 
2. :class:`FinetuneableZoobotRegressor <zoobot.pytorch.training.finetune.FinetuneableZoobotRegressor>` for regression tasks (including on a unit interval e.g. a fraction).
3. :class:`FinetuneableZoobotTree <zoobot.pytorch.training.finetune.FinetuneableZoobotTree>` for training on a tree of labels (e.g. Galaxy Zoo vote counts). 

Each user-facing class is actually a subclass of a non-user-facing abstract class, :class:`FinetuneableZoobotAbstract <zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract>`.
:class:`FinetuneableZoobotAbstract <zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract>` has specifying how to finetune a general PyTorch model,
which the user-facing classes inherit. 

`FinetuneableZoobotAbstract <zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract>` controls the core finetuning process: loading a model, accepting arguments controlling the finetuning process, and running the finetuning.
The user-facing class adds features specific to that type of task. For example, :class:`FinetuneableZoobotClassifier <zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier>` adds additional arguments like `num_classes`.
It also specifies an appropriate head and a loss function.



Finetuning with PyTorch Lightning
-----------------------------------


are all "LightningModule" classes.
These classes have (custom) methods like ``training_step``, ``validation_step``, etc., which specify what should happen at each training stage.


Zoobot is written in PyTorch, a popular deep learning library for Python. 
PyTorch requires a lot of boilerplate code to train models, especially at scale (e.g. multi-node, multi-GPU).
We use PyTorch Lightning, a third party wrapper API, to make this boilerplate code someone else's job.


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
  
As you can see, there's quite a few layers (pun intended) to training Zoobot models. But we hope this setup is both simple to use and easy to extend, whichever (PyTorch) frameworks you're using.


.. The deep learning part is the simplest piece. 
.. ``define_model.py`` has functions to that define pure PyTorch ``nn.Modules`` (a.k.a. models).

.. Encoders (a.k.a. models that take an image and compress it to a representation vector) are defined using the third party library ``timm``.
.. Specifically, ``timm.create_model(architecture_name)`` is used to get the EfficientNet, ResNet, ViT, etc. architectures used to encode our galaxy images.
.. This is helpful because defining complicated architectures becomes someone else's job (thanks, Ross Wightman!) 

.. Heads (a.k.a. models that take a representation vector and make a prediction) are defined using ``torch.nn.Sequential``. 
.. The function :func:`zoobot.pytorch.estimators.define_model.get_pytorch_dirichlet_head`, for example, returns the custom head used to predict vote counts (see :ref:`training_on_vote_counts`).

.. The encoders and heads in ``define_model.py`` are used for both training from scratch and finetuning
