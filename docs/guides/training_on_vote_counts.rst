.. _training_on_vote_counts:

Training on Vote Counts
=======================

Citizen science projects like Galaxy Zoo typically record the total votes for each answer to each question.
The questions are often arranged in a sequence - a decision tree - where the question asked depends on the previous answers.
In this scenario, some questions might be asked to many volunteers - any recieve many votes for their answers - while some questions might only be asked to a handful of volunteers.

Zoobot includes a custom-designed loss (Dirichlet-Multinomial) to learn from these vote counts.
:class:`zoobot.pytorch.estimators.define_model.ZoobotTree` and :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotTree` both use this loss.
But to do so, they need to know:

- the vote counts for each image, provided via catalog columns
- which answers belong to which questions, provided via the :class:`zoobot.shared.schemas.Schema` object.

Creating a Catalog
------------------

Create a catalog recording, for each galaxy, what votes the volunteers gave and where the galaxy image is saved.

Specifically, the catalog should be a table with rows of (unique) galaxies and columns including:

- ``id_str``, a string that uniquely identifies each galaxy (e.g. the iauname, like ``J012345``, or the decals ``{brickid}_{objid}``, like ``1856_67919``)
- ``file_loc``, the absolute location of the galaxy image on disk. This is expected to be a .png or .jpg of any size, but you could easily extend it for other filetypes if needed.
- a column with the number of votes for each question you want to predict, matching the schema (above).  For GZD-5, this is e.g. ``smooth-or-featured_smooth``, ``smooth-or-featured_featured-or-disk``, etc.

For example:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - id_str
     - file_loc
     - smooth-or-featured_smooth
     - smooth-or-featured_featured-or-disk
   * - J101419
     - /path/to/J101419.jpg
     - 12
     - 28
   * - J101420
     - /path/to/J101420.jpg
     - 17
     - 23

.. warning::

    Answers with zero votes should be listed as 0 in the catalog, not left blank or set to nan.
    This ensures the number of votes can be summed to get the total votes for a question.


Specifying the Decision Tree using a Schema
--------------------------------------------

.. note:: 

    You only need to specify the decision tree if you are training on vote counts.
    If you are fine-tuning on a classification problem, you will be using the cross-entropy loss and therefore do not need a schema.
    See :ref:`the finetuning guide <finetuning_guide>` for more.


To train a model on Galaxy Zoo's decision trees, we need to know what the questions and answers are (the "pairs"), and which questions are asked following which answers (the "dependencies").

`galaxy_datasets.shared.label_metadata <https://github.com/mwalmsley/galaxy-datasets/blob/main/galaxy_datasets/shared/label_metadata.py>`__ is essentially many manually-written dicts that describe these relationships. For example:

.. code-block:: python

    # inside github/mwalmsley/galaxy-datasets/shared/label_metadata.py

    gz2_pairs = {
        'smooth-or-featured': ['_smooth', '_featured-or-disk'],
        'disk-edge-on': ['_yes', '_no'],
        'has-spiral-arms': ['_yes', '_no']
        # etc
    }

    gz2_dependencies = {
        'smooth-or-featured': None,  # always asked
        'disk-edge-on': 'smooth-or-featured_featured-or-disk',
        'has-spiral-arms': 'smooth-or-featured_featured-or-disk'
        # etc
    }

`zoobot.shared.schemas <https://github.com/mwalmsley/zoobot/blob/main/zoobot/shared/schemas.py>`__ contains the ``Schema`` class. 
``Schema`` objects have methods and properties which are more convenient for interpreting the decision tree than a simple dict.
Most importantly, when training a model, your chosen ``schema`` tells the models which answers relate to which questions.

To create a new ``Schema``, pass in your pairs and dependencies:

.. code-block:: python

    from zoobot.shared.schemas import Schema

    schema = Schema(gz2_pairs , gz2_dependencies)

The decision trees for all major GZ projects are already specified in `label_metadata.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/shared/label_metadata.py>`_. 
For other projects, you'll need to define your own (it's tedious but simple, just follow the same pattern).
