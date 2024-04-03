.. _advanced_finetuning:

Advanced Finetuning
=====================


Zoobot includes the :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier`, :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotRegressor`, and :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotTree`
classes to help you finetune Zoobot on classification, regression, or decision tree problems, respectively. 
But what about other problems, like object detection?

Here's how to integrate pretrained Zoobot models into your own code.

Using Zoobot's Encoder Directly
------------------------------------

To get Zoobot's encoder, load any Finetuneable class and grab the encoder attribute:

.. code-block:: python

    model = FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano')
    encoder = model.encoder

or, because Zoobot encoders are `timm` models, you can just directly use `timm`:

.. code-block:: python

    import timm

    encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano', pretrained=True, num_classes=0)


You can use it like any other `timm` model. For example, we did this to `add contrastive learning <https://arxiv.org/abs/2206.11927>`_. Good luck!



Subclassing FinetuneableZoobotAbstract
---------------------------------------

If you'd like to finetune Zoobot on a new task that isn't classification, regression, or vote counts,
you could instead subclass :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract`.
This lets you use our finetuning code with your own head and loss functions.

Imagine there wasn't a regression version and you wanted to finetune Zoobot on a regression task. You could do:

.. code-block:: python

    
    class FinetuneableZoobotCustomRegression(FinetuneableZoobotAbstract):

        def __init__(
            self,
            foo,
            **super_kwargs
        ):

            super().__init__(**super_kwargs)

            self.foo = foo
            self.loss = torch.nn.SomeCrazyLoss()
            self.head = torch.nn.Sequential(my_crazy_head)

    # see zoobot/pytorch/training/finetune.py for more examples and all methods required

You can then finetune this new class just as with e.g. :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotRegressor`.


Extracting Frozen Representations
----------------------------------------

Once you've finetuned to your survey, or if you're using a pretrained survey, (SDSS, Hubble, DECaLS/DESI, and soon HSC),
the representations can be stored as frozen vectors and used as features.
We use this at Galaxy Zoo to power our upcoming similary search and anomaly-finding tools.

As above, we can get Zoobot's encoder from the .encoder attribute. We could use ``encoder()`` to calculate our representations.
But then we'd have to deal with batching, looping, etc. 
To avoid this boilerplate, Zoobot includes a PyTorch Lightning class that lets you pass ``encoder`` to the same :func:`zoobot.pytorch.predictions.predict_on_catalog.predict`
utility function used for making predictions with a full Zoobot model.

.. code-block:: python

    from zoobot.pytorch.training import representations

    # convert to simple pytorch lightning model
    lightning_encoder = ZoobotEncoder.load_from_name('hf_hub:mwalmsley/zoobot-encoder-convnext_nano')

    predict_on_catalog.predict(
        catalog,
        lightning_encoder,
        n_samples=1,
        label_cols=label_cols,
        save_loc=save_loc,
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs
    )

See `zoobot/pytorch/examples/representations <https://github.com/mwalmsley/zoobot/tree/main/zoobot/pytorch/examples/representations>`_ for a full working example.

We are sharing precalculated representations for all our DESI galaxies, and soon for HSC as well.
Check the data notes at :doc:/data_access.

The representations are typically quite high-dimensional (e.g. 1280 for EfficientNetB0) and therefore highly redundant.
We suggest using PCA to compress them down to a more reasonable dimension (e.g. 40) while preserving most of the information.
This was our approach in the `Practical Morphology Tools paper <https://arxiv.org/abs/2110.12735>`_.
