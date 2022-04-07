.. Zoobot documentation master file, created by
   sphinx-quickstart on Mon Mar 15 15:03:45 2021.

Zoobot Documentation
====================

Zoobot makes it easy to train a state-of-the-art deep learning classifier to solve your galaxy morphology problem.
For example, you can train a classifier to find ring galaxies in under `200 lines of code <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/finetune_minimal.py>`_.

.. figure:: finetuning_rings.png
   :alt: Ring galaxies found using Zoobot
   

   *Ring galaxies found using Zoobot and 212 labelled examples.*

Zoobot is intended for three tasks: training new models, calculating representations, and applying finetuning. 
You can find practical guides to each task below.
Each includes working example scripts you can run and adapt.

.. toctree::
   :maxdepth: 2

   /guides/guides


You do not need to be a machine learning expert to use Zoobot. 
Zoobot includes :ref:`components <overview_components>` for common tasks like loading images, managing training, and making predictions.
You simply need to assemble these together. 

.. toctree::
   :maxdepth: 2

   components/overview


Zoobot includes pretrained weights, precalculated representations, example ring galaxy catalogues, and more. See here for a guide to the data:

.. toctree::
   :maxdepth: 2

   data_notes


API Reference
-------------

Look here for information on a specific function, class or
method.

.. toctree::
   :maxdepth: 2

   autodoc/api

Indices
^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Sphinx links below
.. https://www.sphinx-doc.org/en/master/usage/quickstart.html
.. https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
.. https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/
.. https://brendanhasz.github.io/2019/01/05/sphinx.html useful summary

.. To build:
.. install sphinx https://www.sphinx-doc.org/en/master/usage/installation.html is confusing, you can just use pip install -U sphinx
.. run from in docs folder:    make html

.. docs/autodoc contains the tree that sphinx uses to add automatic documentation
.. it needs folders and files matching the python source
.. you will need to add a new {folder}.rst, a new folder, and a new {file}.rst