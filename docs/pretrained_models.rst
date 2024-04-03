.. pretrainedmodels:

Pretrained Models
------------------

Loading Models
==========================

Pretrained models are available via HuggingFace (|:hugging:|) with

.. code-block:: python

    from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier  
    # or FinetuneableZoobotRegressor, or FinetuneableZoobotTree

    model = FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano')

For more options (e.g. loading the ``timm`` encoder directly) see :doc:`guides/advanced_finetuning`.

Available Models
==========================

Zoobot includes weights for the following pretrained models:


.. list-table::
   :widths: 70 35 35 35 35
   :header-rows: 1

   * - Architecture
     - Parameters
     - Test loss
     - Finetune
     - HF |:hugging:|
   * - ConvNeXT-Nano
     - 15.6M
     - 19.23
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-convnext_nano>`__
   * - ConvNeXT-Small 
     - 58.5M
     - 19.14 
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-convnext_small>`__
   * - ConvNeXT-Base 
     - 88.6M
     - **19.04**
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-convnext_base>`__
   * - ConvNeXT-Large 
     - 197.8M
     - 19.09
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-convnext_large>`__
   * - MaxViT-Small
     - 64.9M
     - 19.20
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-maxvit_rmlp_small_rw_224>`__
   * - MaxViT-Base
     - 124.5
     - 19.09
     - Yes
     - TODO
   * - Max-ViT-Large
     - 211.8M
     - 19.18
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-maxvit_large_tf_224>`__
   * - EfficientNetB0 
     - 5.33M
     - 19.48
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-efficientnet_b0>`__
   * - EfficientNetV2-S
     - 48.3M
     - 19.33
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-tf_efficientnetv2_s>`__
   * - ResNet18
     - 11.7M
     - 19.43
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-resnet18>`__
   * - ResNet50
     - 25.6M
     - 19.83
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-resnet50>`__


.. note:: 

    Missing a model you need? Reach out! There's a good chance we can train any model supported by `timm <https://github.com/huggingface/pytorch-image-models>`_.


Which model should I use?
===========================

We suggest the PyTorch EfficientNetB0 224-pixel model for most users.

Zoobot will prioritise PyTorch going forward. For more, see here.
The TensorFlow models currently perform just as well as the PyTorch equivalents but will not benefit from any future updates.

EfficientNetB0 is a small yet capable modern architecture. 
The ResNet50 models perform slightly worse than EfficientNet, but are a very common architecture and may be useful as benchmarks or as part of other frameworks (like detectron2, for segmentation).

It's unclear if color information improves overall performance at predicting GZ votes.
For CNNs, the change in performance is not significant. For ViT, it is measureable but small.
We suggesst including color if it is expected to be important to your specific task, such as hunting green peas.

Larger input images (300px vs 224px) may provide a small boost in performance at predicting GZ votes.
However, the models require more memory and train/finetune slightly more slowly.
You may want to start with a 224px model and experiment with "upgrading" once you're happy everything works.


All models are trained on the GZ Evo dataset described in the `Towards Foundation Models paper <https://arxiv.org/abs/2206.11927>`_.
This dataset includes 550k galaxy images and 92M votes drawn from every major Galaxy Zoo campaign: GZ2, GZ Hubble, GZ CANDELS, and GZ DECaLS/DESI.

All models are trained on the same images shown to Galaxy Zoo volunteers.
These are typically 424 pixels across.
The images are transformed using the galaxy-datasets default transforms (random off-center crop/zoom, flips, rotation) and then resized to the desired input size (224px or 300px) and, for 1-channel models, channel-averaged.

We also include a few additional ad-hoc models `on Dropbox <https://www.dropbox.com/scl/fo/l1l7frgy12wtmsbm0hihb/h?dl=0&rlkey=sq5wevuhxs7ku5ki4cwhbhm5j>`_.

- EfficientNetB0 models pretrained only on GZ DECaLS GZD-5. For reference/comparison.
- EfficientNetB0 models pretrained with smaller images (128px and 64px). For debugging.



.. What about the images?
.. --------------------------

.. You can find most of our datasets on the `galaxy-datasets repo <https://github.com/mwalmsley/galaxy-datasets>`_.
.. The datasets are self-downloading and have loading functions for both PyTorch and TensorFlow.
