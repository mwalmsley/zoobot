.. _finetuning_guide:

Finetuning
=====================================

Galaxy Zoo answers the most common morphology questions: does this galaxy have spiral arms, is it merging, etc. 
But what if you want to answer a different question?

**You can finetune our automated classifier to solve new tasks on new galaxy images.**

Zoobot has been trained to simultaneously answer all of the Galaxy Zoo questions.
This provides a good starting point to be taught other morphology-related tasks using very little new data.

The high-level approach is:

1. Load the trained model, replacing the head (output layers) to match your task
2. Retrain the model on your new task, typically with a low learning rate outside the new head

You will likely only need a small amount of labelled images; a few hundred is a good starting point. 
Retraining (finetuning) this model requires much less time and labels than starting from scratch.

.. note:: 

    If you do want to start from scratch, Zoobot can do that as well - see zoobot/benchmarks.
    But we provide many pretrained models so hopefully you won't need to.


Examples
---------------------

Zoobot includes many working examples of finetuning: 

- `Google Colab notebook <https://colab.research.google.com/drive/1A_-M3Sz5maQmyfW2A7rEu-g_Zi0RMGz5?usp=sharing>`__ (recommended starting point)
- `finetune_binary_classification.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/finetuning/finetune_binary_classification.py>`__ (script version of the Colab notebook)
- `finetune_counts_full_tree.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/pytorch/examples/finetuning/finetune_counts_full_tree.py>`__ (for finetuning on a complicated GZ-style decision tree)

Below, for less familiar readers, we walk through the ``finetune_binary_classification.py`` example in detail.

Background
---------------------

Fine-tuning, also known as transfer learning, is when a model trained on one task is partially retrained for use on another related task.
This can drastically reduce the amount of labelled data needed.
For a general introduction, see `this excellent blog post <https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>`_.

Here, we will finetune a pretrained Zoobot model to find ringed galaxies.

Load Pretrained Model
---------------------

Neural networks, like any statistical model, are trained by fitting their free parameters to data.
The free parameters in neural networks are called weights.
When we load a network, we are reading the fitted values of the weights from a file saved on disk.
These files are called checkpoints (like video game save files - computer scientists are nerds too).

:meth:`zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier` loads the weights of a pretrained Zoobot model from a checkpoint file:

.. code-block:: python

    model = finetune.FinetuneableZoobotClassifier(
      name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',  # which pretrained model to download
      num_classes=2,
      n_blocks=0
    )

You can see the list of pretrained models at :doc:`/pretrained_models`.

What about the other arguments?
When loading the checkpoint, FinetuneableZoobotClassifier will automatically change the head layer to suit a classification problem (hence, ``Classifier``).
``num_classes=2`` specifies how many classes we have, Here, two classes (a.k.a. binary classification).
``n_blocks=0`` specifies how many inner blocks (groups of layers, excluding the output layer) we want to finetune.
0 indicates no other blocks, so we will only be changing the weights of the output layer.


Prepare Galaxy Data
---------------------

We will also need some galaxy images.

.. code-block:: python

    data_dir = '/Users/user/repos/galaxy-datasets/roots/demo_rings'
    train_catalog, _ = demo_rings(root=data_dir, download=True, train=True)

This downloads the demo rings dataset. ``train_catalog`` is a table of galaxies with three crucial columns. 

- ``id_str``, any string uniquely identifying each galaxy
- ``file_loc``, a path to the image (jpg, png, or fits) containing the galaxy
- ``ring``, the label (either 0 or 1)

Then we can use ``GalaxyDataModule`` to tell PyTorch to load the images and labels in this catalog:

.. code-block:: python

    datamodule = GalaxyDataModule(
      label_cols=['ring'],
      catalog=train_catalog,
      batch_size=32
    )

``label_cols`` specifies which columns are the labels to predict. It must be a list.
In a more complicated example, we might predict the labels in many columns at once.
Here, there's only one label column, but it should still be a list.

``batch_size=32`` specifies how many images to show to the network in one go. 
If your computer throws out-of-memory errors, you may need to reduce this.
If training is very slow, you can increase this.

``GalaxyDataModule`` has many other options for specifying how to transform the images before passing them to the network ("augmentations")
See the `code <https://github.com/mwalmsley/galaxy-datasets/blob/main/galaxy_datasets/pytorch/galaxy_datamodule.py#L18>`__ (in another repo).


Run the Finetuning
---------------------

Now we have loaded our pretrained model (with a new automatically-replaced head) and specified our data, we are ready to run the finetuning.

.. code-block:: python

    trainer = finetune.get_trainer(save_dir, accelerator='cpu', max_epochs=100)

The ``trainer`` object is used to specify how I would like my model to be trained. 
Here, I want to train with a CPU for up to 100 epochs (stopping early if the validation loss stops improving).
For more options, see the docstring: :func:`zoobot.pytorch.training.finetune.get_trainer`

Then we use it to fit our pretrained model:

.. code-block:: python

    trainer.fit(model, datamodule)

This uses the AdamW optimizer and the cross-entropy loss.
Other types of problem will need different losses.
``FinetuneableZoobotTree`` has a loss designed for GZ-style decision trees.

``model`` has now been fit to the training data. You can use it to make new predictions - see the full example for more.

The new weights, including the new head, have been saved to ``save_dir``.
You can load them at any time to make predictions later.

.. code-block:: python

    finetuned_model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(best_checkpoint)

Now go do some science!
