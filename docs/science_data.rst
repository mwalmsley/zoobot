.. sciencedata:

Science Data
-------------

The goal of Zoobot is to do science. Here are some science-ready datasets created with Zoobot.

Precalulated Representations
=============================

.. warning:: 

    New for Zoobot v2! We're really excited to see what you build. Reach out for help.

Zoobot v2 now includes precalculated representations for galaxies in the Galaxy Zoo DESI data release.

You could use these to power a similarity search, anomaly recommendation system, multi-modal model, 
or really anything else that needs a short vector summarizing the morphology in a galaxy image.


.. list-table::
   :widths: 35 35 35 35 35 35
   :header-rows: 1

   * - dr8_id
     - ra
     - dec
     - pca_feat_0
     - pca_feat_1
     - ...
   * - TODO
     - TODO
     - TODO
     - TODO
     - TODO
     - ...

``dr8_id`` is the unique identifier for the galaxy in the DESI Legacy Surveys DR8 release and can be crossmatched with the GZ DESI catalogs, below.
It is formed with ``{brickid}_{objid}`` where brickid is the unique identifier for the brick in the Legacy Surveys and objid is the unique identifier for the object in the brick.
``RA`` and ``Dec`` are in degrees. 
The PCA features are the first N principal components representation (which is otherwse impractically large to work with).

Galaxy Zoo Morphology
=======================

Zoobot was used to create a detailed morphology catalog for every (extended, brighter than r=19) galaxy in the DESI Legacy Surveys (8.7M galaxies).

We aim to provide both representations and an updated morphology catalog for DESI-LS DR10.
