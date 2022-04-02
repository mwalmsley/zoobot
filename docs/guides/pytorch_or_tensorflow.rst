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

This makes training more flexible: with shards, changing the training data requires making new shards, which is slow (a few hours).
Avoiding shards also saves disk space: TFRecord shards take much up more disk space than the original images.


Currently,  and is used for the GZ DECaLS and probably the GZ LegS catalogs, (e.g. AstroAugmentations, courtesy Micah Bowles).


Can I have a JAX version?
----------------------------

Only if you build it yourself.
