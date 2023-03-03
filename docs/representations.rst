
Representations
-----------------

.. not currently shown, needs remaking

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

