.. Zoobot documentation master file, created by
   sphinx-quickstart on Mon Mar 15 15:03:45 2021.

Zoobot Documentation
====================

Zoobot makes it easy to finetune a state-of-the-art deep learning classifier to solve your galaxy morphology problem.
For example, you can finetune a classifier to find ring galaxies with `just a few hundred examples <https://colab.research.google.com/drive/1A_-M3Sz5maQmyfW2A7rEu-g_Zi0RMGz5?usp=sharing>`_.

.. figure:: finetuning_rings.png
   :alt: Ring galaxies found using Zoobot
   

   *Ring galaxies found using Zoobot and 212 labelled examples.*

The easiest way to learn to use Zoobot is simply to use Zoobot. 
We suggest you start with our worked examples.

* This `Colab notebook <https://colab.research.google.com/drive/1A_-M3Sz5maQmyfW2A7rEu-g_Zi0RMGz5?usp=sharing>`_ will walk you through using Zoobot to classify galaxy images.
* There's a similar `notebook <https://colab.research.google.com/drive/1MmsjkEvNPvnLRTlJ9Yxm7sZ2uVsfplhD?usp=sharing>`_ for using Zoobot for regression on galaxy images.

For more explanation, read on.

User Guides
-------------

We've written these guides to add explanation and context.

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

We've added docstrings to all the key methods you might use. Feel free to check the code or reach out if you have questions.

.. toctree::
   :maxdepth: 4

   autodoc/pytorch
.. different level to not expand schema too much
.. toctree::
   :maxdepth: 3

   autodoc/shared


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
.. and pip install furo
.. run from in docs folder:    make html

.. can also check docs with
.. make linkcheck
.. (thanks, BS!)

.. docs/autodoc contains the tree that sphinx uses to add automatic documentation
.. it needs folders and files matching the python source
.. you will need to add a new {folder}.rst, a new folder, and a new {file}.rst