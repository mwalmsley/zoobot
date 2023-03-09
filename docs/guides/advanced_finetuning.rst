.. _advanced_finetuning:

Advanced Finetuning
=====================


Zoobot includes the :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier` and :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotTree`
classes to help you finetune Zoobot on classification or decision tree problems, respectively. 
But what about other problems, like regression or object detection?

Here's how to integrate pretrained Zoobot models into your own code.

Using Zoobot's Encoder Directly
------------------------------------

To get Zoobot's encoder, load the model and access the .encoder attribute:

.. code-block:: 

    model = ZoobotTree.load_from_checkpoint(pretrained_checkpoint_loc)
    encoder = model.encoder

    model = FinetuneableZoobotClassifier.load_from_checkpoint(finetuned_checkpoint_loc)
    encoder = model.encoder

    # for ZoobotTree, there's also a utility function to do this in one line
    encoder = finetune.load_pretrained_encoder(pretrained_checkpoint_loc)

:class:`zoobot.pytorch.estimators.define_model.ZoobotTree`, :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier` and :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotTree`
all have ``.encoder`` and ``.head`` attributes. These are the plain PyTorch (Sequential) models used for encoding or task predictions.
The Zoobot classes simply wrap these with instructions for training, logging, checkpointing, and so on.

Now you can use the encoder like any PyTorch Sequential for any machine learning task. We did this to `add contrastive learning <https://arxiv.org/abs/2206.11927>`_. Go nuts.


Subclassing FinetuneableZoobotAbstract
---------------------------------------

If you'd like to finetune Zoobot on a new task that isn't classification or vote counts,
you could instead subclass ``FinetuneableZoobotAbstract``.
This is less general but avoids having to write out your own finetuning training code in e.g. PyTorch Lightning.

For example, to make a regression version:

.. code-block:: 

    
    class FinetuneableZoobotRegression(FinetuneableZoobotAbstract):

        def __init__(
            self,
            foo,
            **super_kwargs
        ):

            super().__init__(**super_kwargs)

            self.foo = foo
            self.loss = torch.nn.MSELoss()
            self.head = torch.nn.Sequential(...)

    # see zoobot/pytorch/training/finetune.py for more examples and all methods required

You can then finetune this new class just as with e.g. FinetuneableZoobotClassifier.


Extracting Frozen Representations
----------------------------------------

Once you've finetuned to your survey, or if you're using a pretrained survey, (SDSS, Hubble, DECaLS/DESI, and soon HSC),
the representations can be stored as frozen vectors and used as features.
We use this at Galaxy Zoo to power our upcoming similary search and anomaly-finding tools.

As above, we can get Zoobot's encoder from the .encoder attribute:

.. code-block:: 

    # can load from either ZoobotTree (if trained from scratch) or FinetuneableZoobotTree (if finetuned)
    encoder = finetune.FinetuneableZoobotTree.load_from_checkpoint(checkpoint_loc).encoder

``encoder`` is a PyTorch Sequential object, so we could use ``encoder.predict()`` to calculate our representations.
But then we'd have to deal with batching, looping, etc. 
To avoid this boilerplate, Zoobot includes a PyTorch Lightning class that lets you pass ``encoder`` to the same :func:`zoobot.pytorch.predictions.predict_on_catalog.predict`
utility function used for making predictions with a full Zoobot model.

.. code-block:: 

    # convert to simple pytorch lightning model
    model = representations.ZoobotEncoder(encoder=encoder, pyramid=False)

    predict_on_catalog.predict(
        catalog,
        model,
        n_samples=1,
        label_cols=label_cols,
        save_loc=save_loc,
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs
    )

See 

We plan on adding precalculated representations for all our DESI galaxies - but we haven't done it yet. Sorry.
Please raise an issue if you really need these.

The representations are typically quite high-dimensional (1280 for EfficientNetB0) and therefore highly redundant.
We suggest using PCA to compress them down to a more reasonable dimension (e.g. 40) while preserving most of the information.
This was our approach in the `Practical Morphology Tools paper <https://arxiv.org/abs/2110.12735>`_.
