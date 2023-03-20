.. Zoobot documentation master file, created by
   sphinx-quickstart on Mon Mar 15 15:03:45 2021.

Zoobot Documentation
====================

Zoobot makes it easy to finetune a state-of-the-art deep learning classifier to solve your galaxy morphology problem.
For example, you can finetune a classifier to find ring galaxies with `just a few hundred examples <https://colab.research.google.com/drive/17bb_KbA2J6yrIm4p4Ue_lEBHMNC1I9Jd?usp=sharing>`_.

.. figure:: finetuning_rings.png
   :alt: Ring galaxies found using Zoobot
   

   *Ring galaxies found using Zoobot and 212 labelled examples.*

The easiest way to learn to use Zoobot is simply to use Zoobot. 
We suggest you start with our worked examples.
The `Colab notebook <https://colab.research.google.com/drive/17bb_KbA2J6yrIm4p4Ue_lEBHMNC1I9Jd?usp=sharing>`_ is the fastest way to get started.
See the README for many scripts that you can run and adapt locally.


Guides
-------------

If you'd like more explanation and context, we've written these guides.

.. toctree::
   :maxdepth: 2

   /guides/guides

Pretrained Models
------------------

To choose and download a pretrained model, see here.

.. toctree::
   :maxdepth: 2

   data_notes


API reference
--------------

Look here for information on a specific function, class or
method.

.. toctree::
   :maxdepth: 2

   autodoc/api


.. You do not need to be a machine learning expert to use Zoobot. 
.. Zoobot includes :ref:`components <overview_components>` for common tasks like loading images, managing training, and making predictions.
.. You simply need to assemble these together. 

.. .. toctree::
..    :maxdepth: 2

..    components/overview



.. Indices
.. ^^^^^^^

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. Sphinx links below
.. https://www.sphinx-doc.org/en/master/usage/quickstart.html
.. https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
.. https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/
.. https://brendanhasz.github.io/2019/01/05/sphinx.html useful summary

.. To build:
.. install sphinx https://www.sphinx-doc.org/en/master/usage/installation.html is confusing, you can just use pip install -U sphinx
.. run from in docs folder:    make html

.. can also check docs with
.. make linkcheck
.. (thanks, BS!)

.. docs/autodoc contains the tree that sphinx uses to add automatic documentation
.. it needs folders and files matching the python source
.. you will need to add a new {folder}.rst, a new folder, and a new {file}.rst