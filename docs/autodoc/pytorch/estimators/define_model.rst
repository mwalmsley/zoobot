.. _define_model:

define_model
-------------

This module defines Zoobot's components. 

:func:`.get_pytorch_encoder` and :func:`.get_pytorch_dirichlet_head` define the encoder and head, respectively, in PyTorch.
:class:`zoobot.pytorch.estimators.define_model.ZoobotTree` wraps these components in a PyTorch LightningModule describing how to train them. 

.. autoclass:: zoobot.pytorch.estimators.define_model.ZoobotTree

|

.. autofunction:: zoobot.pytorch.estimators.define_model.get_pytorch_encoder

|

.. autofunction:: zoobot.pytorch.estimators.define_model.get_pytorch_dirichlet_head
