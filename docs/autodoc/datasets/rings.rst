rings
===================

This module contains utilities to load the "advanced" rings dataset.
This dataset was used in the morphology tools paper.

The practical complications in creating it distract from minimal working examples - new users should start with ``finetune_minimal.py``.
However, it does contain more rings and more reliable labels than the simple dataset used in that example.
If you're willing to accept some extra complexity, read on...


.. autofunction:: zoobot.tensorflow.datasets.rings.get_advanced_ring_image_dataset

|

.. autofunction:: zoobot.tensorflow.datasets.rings.get_advanced_ring_feature_dataset

|

.. autofunction:: zoobot.tensorflow.datasets.rings.get_random_ring_catalogs

|

.. autofunction:: zoobot.tensorflow.datasets.rings.get_rough_class_from_ring_fraction
