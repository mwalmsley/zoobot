.. _datanotes:

Data Notes
==========

Zoobot includes three datasets you might be interested in:

- Weights for trained GZ DECaLS models (with excellent performance at answering GZ DECaLS questions)
- Representations calculated from trained GZ DECaLS models for galaxies in either GZ DECaLS (approx. 340k) or the Galaxy Zoo 2 "Galaxy Challenge" kaggle subset (approx. 60k).
- Catalogues of ring galaxies. There are two catalogues: basic, derived from "ring" tags on the Galaxy Zoo forum, or advanced, derived from the GZ DECaLS "ring" vote fraction.
  
Where the data is small, we have included it with the repository (see the ``data`` folder). Where the data is large, you can download it from Zenodo (see the links below).

You can also download previously-published volunteer vote fractions, automatic vote fractions, and images from the `Galaxy Zoo DECaLS data release deposit <https://doi.org/10.5281/zenodo.4196266>`_.

Weights
-----------------

TensorFlow
...........

We provide pretrained weights for an EfficientNet B0 CNN trained on the GZ DECaLS volunteer votes under `data/pretrained_models <https://github.com/mwalmsley/zoobot/tree/pytorch/data/pretrained_models>`_ . 

- ``replicated_train_only_greyscale_tf`` is trained on the colour (3-channel grz) images shown to volunteers, but the images are averaged across bands before being input.
- ``replicated_train_only_color_tf`` is identical but without averaging across bands. This approach was not used for the GZ DECaLS catalog to avoid bias, but may be useful for e.g. anomaly-finding.

`train_set_only` denotes that the model was trained only on a training subset, while the GZ DECaLS catalog predictions are by CNN trained on all labelled galaxies.

Both models may be replicated using the code under `replication <https://github.com/mwalmsley/zoobot/tree/pytorch/replication>`_.
See :ref:`the DECaLS guide <training_from_scratch>` for pedagogical details on how they were created and how you might train on new Galaxy Zoo campaigns.

.. note:: 

    W+22 trained ensembles of CNNs to answer Galaxy Zoo DECaLS questions. See W+22 for details.
    The exact weights for the actual models used in W+22 are not available because loading weights requires the underlying TensorFlow code to be identical,
    but that code has subsequently been refactored to create the Zoobot package and hence the weights do not load correctly. 
    The weights provided here are equivalent in every respect other than the random seed used for training.

PyTorch
........

We provide pretrained weights 


Representations
-----------------

In W+22, we used the CNN above (trained on all galaxies) to calculate representations for those galaxies.
These representations are available `from Zenodo TODO <TODO>`_ for GZ DECaLS DR5 galaxies and a subset of GZ2 galaxies. 

The most significant file is "cnn_features_decals.parquet".
This file contains the representations calculated for the approx. 340k GZ DECaLS galaxies.
See W+22 for a description of GZD-5.
Galaxies can be crossmatched to other catalogues (e.g. the GZ DECaLS automatic morphology catalogue) by ``iauname``.

"cnn_features_gz2.parquet" is the representations calculated by the *same* model, i.e. without retraining on labelled SDSS GZ2 images,
for the approx 240k images classifed in Galaxy Zoo 2 (Willet 2013). 
These are still fairly good (see W+22), implying the CNN can sometimes generalise well to slightly different surveys. 
However, they could likely be improved by using a model trained on GZ2 directly. The Zoobot code makes this straightforward. 
The galaxies can be cross-matched to the Galaxy Zoo 2 catalogues on the "id_str" column, which is equal to the GZ2 objid (e.g. ``588018090547020096``).


Catalogues
-----------------


W+22 investigated finding rings in DECaLS DR5 images either using the representations as-is ("cnn_features_decals.parquet", "cnn_features_gz2.parquet"), or fine-tuning those representations.
We have included the code used to carry out this fine-tuning in this repository, both as practical working examples and for reproducibility.
You might like to improve on our work or to use this as a starting point to be swapped out for your own target galaxies.

To carry out the fine-tuning with our example scripts, you will need the catalogues of labelled rings and the images.
This repository includes two catalogues under ``data``: "example_ring_catalog_basic.csv" and "example_ring_catalog_advanced.parquet".

"example_ring_catalog_basic.csv" assigns the GZD-5 galaxies a label depending on if they were tagged as "ring" by any volunteers on the Galaxy Zoo forum. 
This basic catalogue is used for demonstration purposes in ``finetune_minimal.py``.

"example_ring_catalog_advanced.parquet", under ``data``, is the catalogue of labelled GZD-5 ring galaxies we actually used for training/validation/testing in W+22.

The columns include:

- the galaxy ``iauname`` (unique id)
- the previously-published automatic vote fractions for the smooth/featured and edge-on GZ DECaLS questions
- the volunteer vote fraction for the "ring" answer to the question "are there any of these rare features"
- the relative path to the image (e.g. ``J000/J0000001.png``). 
  
Only galaxies with a smooth vote fraction < 0.75 and edge-on vote fraction < 0.25 are included (see W+22).

You can download the images referenced in both catalogues from the `Galaxy Zoo DECaLS data release <https://doi.org/10.5281/zenodo.4196266>`_.
Note that all the images are approx. 100GB. We have split them into several chunks to make this process slightly less painful. 
The original data is from the DECaLS survey; please acknowledge them appropriately (see W+22 for an example).
