.. _datanotes:

Data Notes
==========

Zoobot includes three datasets you might be interested in:

- Weights for trained GZ DECaLS models (with excellent performance at answering GZ DECaLS questions)
- Representations calculated from trained GZ DECaLS models for galaxies in either GZ DECaLS (approx. 340k) or a Galaxy Zoo 2 subset (approx. 240k).
- Catalogues of ring galaxies. There are two catalogues: basic, derived from "ring" tags on the Galaxy Zoo forum, or advanced, derived from the GZ DECaLS "ring" vote fraction.
  
Pretrained weights and representations are available `from Dropbox <https://www.dropbox.com/sh/asqia51m1u3ccl1/AAD2XZz-AtG-ZShLiPRBrRzqa?dl=0>`_.
Ring catalogs and a subset of galaxy images and are available under the `data <https://github.com/mwalmsley/zoobot/tree/pytorch/data>`_ folder. 
Full morphology catalogs and all galaxy images are available from the `Galaxy Zoo DECaLS data release Zenodo deposit <https://doi.org/10.5281/zenodo.4196266>`_.

Weights
-----------------

We provide pretrained weights for models trained on GZ DECaLS GZD-5 campaign volunteer votes. 

All models are trained on a dataset split into 70% training, 10% validation (for early stopping), and 20% test subsets.
This is denoted by ``train_only_dr5`` in the name.
To verify that the TensorFlow/PyTorch versions of Zoobot perform similarly, the split is the same for both versions and all models.

The exact training scripts are in the `replication <https://github.com/mwalmsley/zoobot/tree/pytorch/replication>`_ folder.
These record important details like batch size, mixed precision, etc.

See :ref:`the DECaLS guide <training_from_scratch>` for pedagogical details on how to train models from scratch and how you might train on new Galaxy Zoo campaigns.

.. note:: 
    
    Models trained on all labelled Legacy Survey DR8 images have been created but have not yet been published. If this is particularly interesting/urgent for you, please get in touch.

TensorFlow
...........

We provide pretrained weights for an EfficientNet B0 CNN (defined here) trained on the GZ DECaLS GZD-5 campaign volunteer votes. You can download them `from Dropbox <https://www.dropbox.com/sh/asqia51m1u3ccl1/AAD2XZz-AtG-ZShLiPRBrRzqa?dl=0>`_.

- ``effnet_train_only_dr5_greyscale_tf`` is trained on the colour (3-channel grz) images shown to volunteers, but the images are averaged across bands before being input.
- ``effnet_train_only_dr5_color_tf`` is identical but without averaging across bands. This approach was not used for the GZ DECaLS catalog to avoid bias, but may be useful for e.g. anomaly-finding.


.. note:: 

    W+22a trained ensembles of CNNs to answer Galaxy Zoo DECaLS questions. See W+22a for details.
    The exact weights for the actual models used in W+22 are not available because loading weights requires the underlying TensorFlow code to be identical,
    but that code has subsequently been refactored to create the Zoobot package and hence the weights do not load correctly. 
    The weights provided here are equivalent in every respect other than the random seed used for training.
    Also note that W+22a's GZ DECaLS catalog predictions are made by CNN trained on all labelled galaxies, not a training subset.

PyTorch
........

We provide pretrained weights for EfficientNetB0, as with TensorFlow above, and for ResNet50. You can download them `from Dropbox <https://www.dropbox.com/sh/asqia51m1u3ccl1/AAD2XZz-AtG-ZShLiPRBrRzqa?dl=0>`_.

- ``efficientnet_dr5_pytorch`` and ``efficientnet_dr5_pytorch_color`` are trained similarly to the TensorFlow equivalents.
- ``resnet_detectron_dr5_pytorch`` and ``resnet_detectron_dr5_pytorch_color`` are trained using the detectron2 implementation of ResNet50 (with unfrozen batch norm layers). See here.
- ``resnet50_torchvision_dr5_pytorch_color`` is trained using the torchvision implementation of ResNet50. Only color is supported. See here.

The ResNet50 models perform slightly worse than EfficientNet, but are a very common architecture and may be useful as benchmarks or as part of other frameworks (like detectron2, for segmentation).

.. warning:: 
    
    The PyTorch models are likely to change further as we experiment with different training and augmentation strategies.


Representations
-----------------

In Practical Morphology Tools (W+22b), we used the CNN above (trained on all galaxies) to calculate representations for those galaxies.
These representations are available `from here <https://www.dropbox.com/sh/asqia51m1u3ccl1/AAD2XZz-AtG-ZShLiPRBrRzqa?dl=0>`_ for GZ DECaLS DR5 galaxies and most GZ2 galaxies. 

``cnn_features_decals.parquet`` contains the representations calculated for the approx. 340k GZ DECaLS galaxies.
See W+22a for a description of GZD-5.
Galaxies can be crossmatched to other catalogues (e.g. the GZ DECaLS automatic morphology catalogue) by ``iauname``.

``cnn_features_gz2.parquet`` is the representations calculated by the *same* model, i.e. without retraining on labelled SDSS GZ2 images,
for the approx 240k images classifed in Galaxy Zoo 2's main sample (Willet 2013). 
These are still fairly good (see W+22b), implying the CNN can sometimes generalise well to slightly different surveys. 
However, they could likely be improved by using a model trained on GZ2 directly. The Zoobot code makes this straightforward - if you give it a go, let me know what happens! 
The galaxies can be cross-matched to the Galaxy Zoo 2 catalogues on the "id_str" column, which is equal to the GZ2 objid (e.g. ``588018090547020096``).

.. note:: 

    The representations have 1280 features per galaxy. These features are highly redundant (because the CNN has no reason to make them independent).
    They can be effectively compressed by e.g. PCA into around ~40 features while preserving ~95% of the variation.
    The compressed representations are much more practical to work with for e.g. clustering, anomaly-finding, active learning, visualisation with umap, etc.
    See `this example <https://github.com/mwalmsley/zoobot/tree/pytorch/zoobot/shared/compress_representations.py>`_ for how to compress the representations (essentially, just apply sklearn's ``IterativePCA``).
    See :ref:`representations_guide` for more details on the representations.


Catalogues
-----------------

W+22b investigated finding rings in DECaLS DR5 images either using the representations as-is (``cnn_features_decals.parquet``, ``cnn_features_gz2.parquet``, see above), or fine-tuning those representations.
We have included the code used to carry out this fine-tuning in this repository, both as practical working examples and for reproducibility.
You might like to improve on our work or to use this as a starting point to be swapped out for your own target galaxies.

To carry out the fine-tuning with our example scripts, you will need the catalogues of labelled rings and the images.
This repository includes two catalogues under `data <https://github.com/mwalmsley/zoobot/tree/pytorch/data>`_ : ``example_ring_catalog_basic.csv`` and ``example_ring_catalog_advanced.parquet``.

``example_ring_catalog_basic.csv`` is a basic catalogue used for demonstration purposes in ``finetune_minimal.py``.
Ring labels are assigned depending on if each GZD-5 galaxy was tagged as "ring" by any volunteers on the Galaxy Zoo forum. 

``example_ring_catalog_advanced.parquet`` is the catalogue of ring galaxies we actually used for training/validation/testing in W+22b.
Ring labels are assigned depending on how many GZD-5 volunteers answered the GZ DECaLS "Is there anything odd" question with "Ring".
If more than 25% answered "Ring", the label is 1. If less than 5% answered ring, the label is 0.
Other galaxies are removed.
For full details and additional selection cuts, see `W+22b Sec 4.2 <https://arxiv.org/pdf/2110.12735.pdf>`_.

The columns include:

- the galaxy ``iauname`` (unique id)
- the previously-published automatic vote fractions for the smooth/featured and edge-on GZ DECaLS questions
- the volunteer vote fraction for the "ring" answer to the question "are there any of these rare features"
- the relative path to the image (e.g. ``J000/J0000001.png``). 

You can download the images referenced in both catalogues from the `Galaxy Zoo DECaLS data release <https://doi.org/10.5281/zenodo.4196266>`_.
Note that all the images are approx. 100GB. We have split them into several .zip chunks to make this process slightly less painful. 

The original data is from the DECaLS survey; please acknowledge them appropriately (see W+22a for an example).
