.. _training_from_scratch:

Training from Scratch
=====================

Zoobot was originally used to train models on GZ DECaLS. Those models predicted the published automated classifications.
The same code could be re-used to train new models on other Galaxy Zoo projects.
You could also extend the code (e.g. by changing the architecture, preprocessing, etc) to improve performance.

.. note:: 

    If you just want to use the classifier, you don't need to make it from scratch.
    We provide :ref:`pretrained weights and precalculated representations <datanotes>`.
    You can even start from these and :ref:`finetune <finetuning_guide>` to your problem.

You will need galaxy images and volunteer classifications.
For Galaxy Zoo DECaLS (GZD-5), these are available at `<https://doi.org/10.5281/zenodo.4196266>`_.
You will also need a fairly good GPU - we used an NVIDIA V100. 
You might get away with a worse GPU by lowering the batch size (we used 128, 64 works too) or the image size, but this may affect performance.

The high-level approach to create a CNN is:

- Specify the decision tree asked of volunteers
- Prepare a catalog with your images and labels (matching the decision tree)
- Convert your images into a format that's fast to read - either .jpeg, for pytorch, or TFRecord shards, for TensorFlow
- Train the CNN on the converted images

.. note:: 

    The PyTorch version of zoobot does not require the images to be written into TFRecord shards. The images are read directly.
    This makes it easier to adjust your training data and requires less disk space.
    We recommend using the PyTorch version of zoobot where possible - see :ref:`Should I use PyTorch or TensorFlow? <pytorch_or_tensorflow>`


Specifying the Decision Tree using a Schema
--------------------------------------------

.. note:: 

    You only need to specify the decision tree if you are training from scratch.
    Knowing about the decision tree is only required for the multi-question loss function introduced with Zoobot for Galaxy Zoo.
    If you are fine-tuning, you are likely to be using a typical loss function (such as binary cross-entropy) and therefore do not need a schema.
    See :ref:`the finetuning guide <finetuning_guide>` for more.


To train a model on Galaxy Zoo's decision trees, we need to know what the questions and answers are (the "pairs"), and which questions are asked following which answers (the "dependencies").

.. Galaxy Zoo uses a decision tree where the questions asked depend upon the previous answers.
.. For example, volunteers are only asked the question "How many spiral arms?" if they previously answered "Yes" to "Does this galaxy have spiral arms?"

.. When training a model, it's very important to know how many volunteers were asked each question because this affects how confident we should be in the label.
.. 10 of 20 volunteers saying "Two spiral arms" is a more confident label than 1 of 2 volunteers.
.. Our model should be penalised more (have a higher loss) when it's wrong about confident labels (with many volunteer answers) than uncertain labels (with few volunteer answers).

`zoobot.shared.label_metadata <https://github.com/mwalmsley/zoobot/blob/main/zoobot/shared/label_metadata.py>`__ is essentially many manually-written dicts that describe these relationships. For example:

.. code-block:: python

    # inside zoobot/shared/label_metadata.py

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
Most importantly, when training a model, your chosen ``schema`` is used to create the multi-question loss.


.. code-block:: python

    # when training a model
    # see zoobot/tensorflow/examples/train_model.py, see zoobot/pytorch/examples/train_model.py

    from zoobot.shared import label_metadata, schemas
    from zoobot.tensorflow.training import losses  # or pytorch.training, equivalently

    # loading the dicts saved in label_metadata
    # (did you know you can access any variable with this pattern, not just functions?)
    question_answer_pairs = label_metadata.decals_pairs
    dependencies = label_metadata.gz2_and_decals_dependencies

    # creating a Schema object from those dicts
    schema = schemas.Schema(question_answer_pairs, dependencies)

    # using the Schema object to define the complicated multi-question loss, informed by the decision tree structure
    multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
    # the details of this are only important if you want to adjust how the multi-question loss works


The decision trees for GZ2, GZ DECaLS 1/2 and GZ DECaLS 5/8 are already specified in `label_metadata.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/shared/label_metadata.py>`_. 
For other projects, you'll need to define your own (it's easy, just follow the same pattern).

Once the decision tree is specified, you'll need to gather the images and volunteer responses.


Creating a Catalog
------------------

Create a catalog recording, for each galaxy, what votes the volunteers gave and where the galaxy image is saved.

Specifically, the catalog should be a table with rows of (unique) galaxies and columns including:

- ``id_str``, a string that uniquely identifies each galaxy (e.g. the iauname, like ``J012345``, or the decals ``{brickid}_{objid}``, like ``1856_67919``)
- ``file_loc``, the absolute location of the galaxy image on disk. This is expected to be a .png or .jpg of any size, but you could easily extend it for other filetypes if needed.
- a column with the number of votes for each question you want to predict, matching the schema (above).  For GZD-5, this is e.g. smooth-or-featured_smooth, smooth-or-featured_featured-or-disk, etc.

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

Next Steps
----------

We are now ready to make the final adjustments to our data and then train our model.

Exactly how the data is loaded depends on if you're using the PyTorch or TensorFlow version of Zoobot. 

With the TensorFlow version, the images should be saved as TFRecords (stacks of binary-encoded images, designed to be read very quickly to speed up training).
Any static adjustments (for example, converting to greyscale) should be done when saving the TFRecords.
Stochastic adjustments (for example, rotation augmentations) happen when the images are input to the model, as the first few layers of the model are tf.keras.layers.preprocessing layers.
See the :ref:`Training with TensorFlow <training_with_tensorflow>` guide.

.. note:: 

    I will probably remove the TFRecord feature and have the TensorFlow version load the images directly, as the PyTorch version does now.
    I will also adjust the model to not include these preprocessing layers, to allow more flexibilty with stochastic adjustments.
    Any help would be very welcome and would be credited appropriately.

With the PyTorch version, you need to define a `PyTorch Lightning DataModule <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_ that describes how to load the images listed in your catalog and how to divide them into train/validation/test sets. 
To train as fast as possible, any static adjustments should already have been done to those images.
Stochastic adjustments happen when the images are read from those paths into memory, using the PyTorch dataloaders you define in your DataModule.
See the :ref:`Training with PyTorch <training_with_pytorch>` guide.

.. note:: 

    The PyTorch example uses stochastic adjustments from the new AstroAugmentations package by Micah Bowles. These are optional and were not used for the GZ DECaLS paper.
    However, we believe they will improve performance (vs. standard augmentations) and hope to present results on this soon.
