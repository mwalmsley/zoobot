.. sciencedata:

Science Data
-------------

The goal of Zoobot is to do science. Here are some science-ready datasets created with Zoobot.

Precalulated Representations
=============================

.. warning:: 

    New for Zoobot v2! We're really excited to see what you build. Reach out for help.

Zoobot v2 now includes precalculated representations for galaxies in the Galaxy Zoo DESI data release.
Download `here <https://www.dropbox.com/scl/fi/ml33hzv4ak1lwffm0fucn/representations_pca_40_with_coords.parquet?rlkey=xu3dwfjc5ando7lkbgk89slpb&dl=0>`_ (2.5GB)

You could use these to power a similarity search, anomaly recommendation system, the vision part of a multi-modal model, 
or really anything else that needs a short vector summarizing the morphology in a galaxy image.




.. list-table::
   :widths: 35 35 35 35 35 35
   :header-rows: 1

   * - id_str
     - ra
     - dec
     - feat_pca_0
     - feat_pca_1
     - ...
   * - 303240_2499
     - 4.021870
     - 3.512972	
     - 0.257407
     - -7.414328	
     - ...

``id_str`` is the unique identifier for the galaxy in the DESI Legacy Surveys DR8 release and can be crossmatched with the GZ DESI catalog (below) ``dr8_id`` key.
It is formed with ``{brickid}_{objid}`` where brickid is the unique identifier for the brick in the Legacy Surveys and objid is the unique identifier for the object in the brick.
``RA`` and ``Dec`` are in degrees. 
The PCA features are the first 40 principal components representation (which is otherwse impractically large to work with).


Galaxy Zoo Morphology
=======================

Zoobot was used to create a detailed morphology catalog for every (extended, brighter than r=19) galaxy in the DESI Legacy Surveys (8.7M galaxies).
The catalog and schema are available from `Zenodo <https://zenodo.org/records/8360385>`_.
For new users, we suggest starting with the ``gz_desi_deep_learning_catalog_friendly.parquet`` catalog file.

We previously used Zoobot to create a similar catalog for `DECaLS DR5 <https://zenodo.org/records/4573248>`_. 
This has now been superceded by the GZ DESI catalog above (which includes the same galaxies, and many more).

We aim to provide both representations and an updated morphology catalog for DESI-LS DR10, but we need to redownload all the images first |:neutral_face:|.

Future catalogs will include morphology measurements for HSC, JWST, and Euclid galaxies (likely in that order).
