.. Zoobot documentation master file, created by
   sphinx-quickstart on Mon Mar 15 15:03:45 2021.

Zoobot Documentation
====================


Guides
------

Zoobot makes it easy to train a state-of-the-art deep learning classifier to solve your galaxy morphology problem.
For example, you can train a classifier to find ring galaxies in under `200 lines of code <https://github.com/mwalmsley/zoobot/blob/main/finetune_minimal.py>`_.

.. figure:: finetuning_rings.png
   :alt: Ring galaxies found using Zoobot
   

   *Ring galaxies found using Zoobot and 212 labelled examples.*


You do not need to be a machine learning expert to use Zoobot. 
Zoobot includes :ref:`components <components>` for common tasks like loading images, managing training, and making predictions.
You simply need to assemble these together. 

You will only need a small amount of labelled images; a few hundred is a good starting point. 
This is because Zoobot includes a classifier already trained to answer Galaxy Zoo questions for DECaLS galaxies.
Retraining (finetuning) this model requires much less time and labels than starting from scratch.

If you do want to start from scratch, to reproduce or improve upon the pretrained classifier, :ref:`Zoobot can do that as well <reproducing_decals>`.

Check out the guides below. Each includes working example scripts you can run and adapt.

.. toctree::
   :maxdepth: 2

   usage/overview
   usage/finetuning
   usage/decals



API Reference
-------------

Look here for information on a specific function, class or
method.

.. toctree::
   :maxdepth: 2

   autodoc/data_utils
   autodoc/estimators
   autodoc/training
   autodoc/predictions
   autodoc/schemas
   autodoc/label_metadata

Indices
^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. https://www.sphinx-doc.org/en/master/usage/quickstart.html
.. https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
.. https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/
.. https://brendanhasz.github.io/2019/01/05/sphinx.html useful summary
.. run (in docs)    make html   to build
