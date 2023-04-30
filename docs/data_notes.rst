.. _datanotes:

Pretrained Models
=================

Zoobot includes weights for the following pretrained models. 

.. list-table:: PyTorch Models
   :widths: 70 35 35 35 35
   :header-rows: 1

   * - Architecture
     - Input Size 
     - Channels
     - Finetune
     - Link
   * - EfficientNetB0
     - 224px
     - 1
     - Yes
     - `Link <https://www.dropbox.com/s/7ixwo59imjfz4ay/effnetb0_greyscale_224px.ckpt?dl=0>`__
   * - EfficientNetB0 
     - 300px
     - 1 
     - Yes
     - `Link <https://www.dropbox.com/s/izvqagd6rkhi4lq/effnetb0_greyscale_300px.ckpt?dl=0>`__
   * - EfficientNetB0 
     - 300px
     - 3
     - Yes
     - WIP
   * - EfficientNetB0 
     - 224px
     - 3
     - Yes
     - WIP
   * - ResNet50 
     - 300px
     - 1
     - Yes
     - `Link <https://www.dropbox.com/s/hvvpy2dar0v1wti/resnet50_greyscale_300px.ckpt?dl=0>`__
   * - ResNet50
     - 224px
     - 1
     - Yes
     - `Link <https://www.dropbox.com/s/copj2576v9uso16/resnet50_greyscale_224px.ckpt?dl=0>`__
   * - ResNet18 
     - 300px
     - 1
     - Yes
     - `Link <https://www.dropbox.com/s/th1irihafkr3wqp/resnet18_greyscale_300px.ckpt?dl=0>`__
   * - ResNet18
     - 224px
     - 1
     - Yes
     - `Link <https://www.dropbox.com/s/on21ri74rbz0qi1/resnet18_greyscale_224px.ckpt?dl=0>`__
   * - Max-ViT Tiny
     - 224px
     - 1
     - Not yet
     - `Link <https://www.dropbox.com/s/pndcgi6wxh9wuqb/maxvittiny_greyscale_224px.ckpt?dl=0>`__



.. list-table:: TensorFlow Models
   :widths: 70 35 35 35 35
   :header-rows: 1

   * - Architecture
     - Input Size 
     - Channels
     - Finetune
     - Link
   * - EfficientNetB0 
     - 300px
     - 1 
     - Yes
     - `Link <https://www.dropbox.com/scl/fo/h8xtoij1wf61oubqhj85x/h?dl=0&rlkey=g80xo368hbacae9465f4pb1q5>`__
   * - EfficientNetB0 
     - 224px
     - 1 
     - Yes
     - WIP


.. note:: 

    Missing a model you need? Reach out! There's a good chance we can train any small-ish model supported by `timm <https://github.com/huggingface/pytorch-image-models>`_.

All models are trained on the GZ Evo dataset described in the `Towards Foundation Models paper <https://arxiv.org/abs/2206.11927>`_.
This dataset includes 550k galaxy images and 92M votes drawn from every major Galaxy Zoo campaign: GZ2, GZ Hubble, GZ CANDELS, and GZ DECaLS/DESI.

All models are trained on the same images shown to Galaxy Zoo volunteers.
These are typically 424 pixels across.
The images are transformed using the galaxy-datasets default transforms (random off-center crop/zoom, flips, rotation) and then resized to the desired input size (224px or 300px) and, for 1-channel models, channel-averaged.

We also include a few additional ad-hoc models `on Dropbox <https://www.dropbox.com/scl/fo/l1l7frgy12wtmsbm0hihb/h?dl=0&rlkey=sq5wevuhxs7ku5ki4cwhbhm5j>`_.

- EfficientNetB0 models pretrained only on GZ DECaLS GZD-5. For reference/comparison.
- EfficientNetB0 models pretrained with smaller images (128px and 64px). For debugging.


Which model should I use?
--------------------------

We suggest the PyTorch EfficientNetB0 single-channel 224-pixel model for most users.

Zoobot will prioritise PyTorch going forward. For more, see here.
The TensorFlow models currently perform just as well as the PyTorch equivalents but will not benefit from any future updates.

EfficientNetB0 is a small yet capable modern architecture. 
The ResNet50 models perform slightly worse than EfficientNet, but are a very common architecture and may be useful as benchmarks or as part of other frameworks (like detectron2, for segmentation).

Color information does not improve overall performance at predicting GZ votes.
This is a little surprising, but we're confident it's true for our datasets (see the benchmarks folder for our tests).
However, it might be useful to include for other tasks where color is critical, such as hunting certain anomalous galaxies.

Larger input images (300px vs 224px) provide a very small boost in performance at predicting GZ votes, on our benchmarks.
However, the models require more memory and train/finetune slightly more slowly.
You may want to start with a 224px model and experiment with "upgrading" once you're happy everything works.


What about the images?
--------------------------

You can find most of our datasets on the `galaxy-datasets repo <https://github.com/mwalmsley/galaxy-datasets>`_.
The datasets are self-downloading and have loading functions for both PyTorch and TensorFlow.
