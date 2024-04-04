finetune
--------------------------------

Use these classes and methods to finetune a pretrained Zoobot model.

See the `README <https://github.com/mwalmsley/zoobot>`_ for a minimal example.
See zoobot/pytorch/examples for more worked examples.

.. autoclass:: zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract
    :members: configure_optimizers

|

.. autoclass:: zoobot.pytorch.training.finetune.FinetuneableZoobotClassifier

|

.. autoclass:: zoobot.pytorch.training.finetune.FinetuneableZoobotRegressor

|

.. autoclass:: zoobot.pytorch.training.finetune.FinetuneableZoobotTree

|

.. autoclass:: zoobot.pytorch.training.finetune.LinearHead
    :members: forward

|

.. autofunction:: zoobot.pytorch.training.finetune.load_pretrained_zoobot

|

.. autofunction:: zoobot.pytorch.training.finetune.get_trainer

|

.. autofunction:: zoobot.pytorch.training.finetune.download_from_name

|