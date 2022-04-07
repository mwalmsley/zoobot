.. _pytorch_or_tensorflow:



PyTorch or TensorFlow?
===========================

Zoobot is really two separate sets of code: `zoobot/pytorch <https://github.com/mwalmsley/zoobot/tree/pytorch/zoobot/pytorch>`_ and `zoobot/tensorflow <https://github.com/mwalmsley/zoobot/tree/pytorch/zoobot/tensorflow>`_.
They both do (almost) exactly the same thing - train the same model architecture on the same Galaxy Zoo data in the same way, for extracting representations and for finetuning - but they use different underlying deep learning frameworks to do so.

We have created two versions of Zoobot so that you can use your preferred framework.
This is especially important in a lab where your colleagues all use one framework, or if your work involves integrating Zoobot with a deep learning approach only available in one framework.


Which Version Should I Use?
----------------------------

- If your research involves an approach only published in one framework, there is no choice.
- If your colleagues all use one framework, choose that.
- If you're in a hurry and have much more experience with one framework, choose that.
- If you want the best performance with Zoobot, choose the PyTorch version. This has the latest features.
- If you're new to deep learning, it's a toss-up. The TensorFlow version has more documentation (because it's older) while the PyTorch version is simpler to load data with. You might want to check out `this blog post <https://walmsley.dev/posts/deep-learning-for-astro>`_ for astronomers getting started with deep learning.

Tell Me More About What's Different
-------------------------------------

The biggest difference is how the training data is loaded.
The TensorFlow Zoobot currently uses TFRecord shards (binary-encoded stacks of images) while the PyTorch version uses the images directly.
This makes training more flexible: with shards, changing the training data requires making new shards, which is slow (a few hours).
Avoiding shards also saves disk space: TFRecord shards take much up more disk space than the original images.

The TensorFlow version has been around longer.
It has more working examples (see https://github.com/mwalmsley/zoobot/tree/pytorch/zoobot/tensorflow/examples>`_).
It has also been used in published work: both the GZ DECaLS catalog and the "practical representation tools" paper used the TensorFlow version.

The PyTorch version is new and includes the latest features but has less examples and documentation and has not yet been formally published.

PyTorch-specific features include:
- ResNet50 architecture option (with both detectron2 and torchvision's implementations) 
- Integration with AstroAugmentations (courtesy Micah Bowles) for custom astronomy image augmentations


Can I have a JAX version?
----------------------------

Only if you build it yourself.
