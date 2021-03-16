.. _finetuning:

Fine-Tuning the Galaxy Zoo Classifier
=====================================

Galaxy Zoo answers the most common morphology questions: does this galaxy have spiral arms, is it merging, etc. 
But what if you want to answer a different question?

**You can finetune our automated classifier to solve new tasks or to solve the same tasks on new surveys.**

The Galaxy Zoo classifier has been trained to simultaneously answer all of the Galaxy Zoo questions and has learned a useful general representation of galaxy morphology.
This general representation is a good starting point for other morphology-related tasks, letting you (re)train a classifier using very little data.

The high-level approach is:

1. Load the trained model, replacing the head (output layers) to match your task
2. Retrain *only* the new head, leaving the rest of the model frozen
3. Optionally, once the new head is trained, unfreeze the rest of the model and train with a low learning rate

Zoobot includes two complete working examples:

finetune_minimal.py shows how to partially finetune a model (1 and 2 only) to classify ring galaxies using an example dataset.
It is designed to be easy to understand.

finetune_advanced.py solves the same problem with some additional tips and tricks: filtering the dataset, balancing the classes (rings are rare), and unfreezing the model (3). 

.. Readers familiar with python and machine learning may prefer to read, copy and adapt the example scripts directly. 

.. Below, for less familiar readers, I walk through the ``finetune_minimal.py`` example in detail.

.. Fine-Tuning Step by Step
.. ^^^^^^^^^^^^^^^^^^^^^^^^

.. Fine-tuning, also known as transfer learning, is when a model trained on one task is partially retrained for use on another related task.
.. This can drastically reduce the amount of labelled data needed.
.. For a general introduction, see `this excellent blog post <https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>`_.

