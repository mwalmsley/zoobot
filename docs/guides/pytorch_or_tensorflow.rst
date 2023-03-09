.. _pytorch_or_tensorflow:



PyTorch or TensorFlow?
===========================

:.. warning:: You should use the PyTorch version if possible. This is being actively developed and has the latest features.

Zoobot is really two separate sets of code: `zoobot/pytorch <https://github.com/mwalmsley/zoobot/tree/main/zoobot/pytorch>`_ and `zoobot/tensorflow <https://github.com/mwalmsley/zoobot/tree/main/zoobot/tensorflow>`_.
They can both train the same EfficientNet model architecture on the same Galaxy Zoo data in the same way, for extracting representations and for finetuning - but they use different underlying deep learning frameworks to do so.

We originally created two versions of Zoobot so that astronomers can use their preferred framework.
But maintaining two almost entirely separate sets of code is too much work for our current resources (Mike's time, basically).
Going forward, the PyTorch version will be actively developed and gain new features, while the TensorFlow version will be kept up-to-date but will not otherwise improve.

Tell Me More About What's Different
-------------------------------------

The TensorFlow version was the original version.
It was used for the `GZ DECaLS catalog <https://arxiv.org/abs/2102.08414>`_ and the `practical morphology tools <https://arxiv.org/abs/2110.12735>`_ paper.
You can train EfficientNetB0 and achieve the same performance as with PyTorch (see the "benchmarks folder").
You can also finetune the trained model, although the process is slightly clunkier.

The PyTorch version was introduced to support other researchers and to integrate with Bootstrap Your Own Latent for the `towards foundation models <https://arxiv.org/abs/2206.11927>`_ paper.
This version is actively developed and includes the latest features.

PyTorch-specific features include:
- Any architecture option from timm (including ResNet and Max-ViT)
- Improved interface for easy finetuning
- Layerwise learning rate decay during finetuning
- Integration with AstroAugmentations (courtesy Micah Bowles) for custom astronomy image augmentations
- Per-question loss tracking on WandB


Can I have a JAX version?
----------------------------

Only if you build it yourself.
