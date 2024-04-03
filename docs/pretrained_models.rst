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
     - 19.83
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-resnet18>`__
   * - ResNet50
     - 25.6M
     - 19.43
     - Yes
     - `Link <https://huggingface.co/mwalmsley/zoobot-encoder-resnet50>`__


.. note:: 

    Missing a model you need? Reach out! There's a good chance we can train any model supported by `timm <https://github.com/huggingface/pytorch-image-models>`_.


Which model should I use?
===========================

We suggest starting with ConvNeXT-Nano for most users.
ConvNeXT-Nano performs very well while still being small enough to train on a single gaming GPU.
You will be able to experiment quickly.

For maximum performance, you could swap ConvNeXT-Nano for ConvNeXT-Small or ConvNeXT-Base.
MaxViT-Base also performs well and includes an ingenious attention mechanism, if you're interested in that.
All these models are much larger and need cluster-grade GPUs (e.g. V100 or above).

Other models are included for reference or as benchmarks.
EfficientNetB0 is equivalent to the model used in the GZ DECaLS and GZ DESI papers.
ResNet18 and ResNet50 are classics of the genre and may be useful for comparison or as part of other frameworks (like detectron2, for segmentation).


How were the models trained?
===============================

The models were trained as part of the report `Scaling Laws for Galaxy Images <TODO>`_.
This report systematically investigates how increasing labelled galaxy data and model size improves performance
and leads to adaptable models that generalise well to new tasks and new telescopes.

All models are trained on the GZ Evo dataset,
which includes 820k images and 100M+ volunteer votes drawn from every major Galaxy Zoo campaign: GZ2, GZ UKIDSS (unpublished), GZ Hubble, GZ CANDELS, GZ DECaLS/DESI, and GZ Cosmic Dawn (HSC, in prep.).
They learn an adaptable representation of galaxy images by training to answer every Galaxy Zoo question at once.
