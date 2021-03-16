Fine-Tuning the Galaxy Zoo Classifier
=====================================

Galaxy Zoo answers the most common morphology questions: does this galaxy have spiral arms, is it merging, etc. 
But what if you want to answer a different question?

You can finetune our automated classifier to solve new tasks or to solve the same tasks on new surveys.

Fine-tuning, also known as transfer learning, is when a model trained on one task is partially retrained for use on another related task (or similar data).
Hopefully, the model will have learned an effective representation of the data from the first task, and so might perform better on the related task than a model trained from scratch.
For a general introduction, see the excellent blog post at `<https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>`_.

The Galaxy Zoo classifier has been trained to simultaneously answer all of the Galaxy Zoo questions and has learned a useful general representation of galaxy morphology.
You can benefit from this general representation by fine-tuning the classifier for your task.

The high-level approach is:

1. Load the pretrained model, replacing the head (output layers) to match your task
2. Train *only* the new head, leaving the rest of the model frozen
3. Optionally, once the new head is trained, unfreeze the rest of the model and train with a low learning rate

There are two complete working examples:

finetune_minimal.py shows how to partially finetune a model (1 and 2 only) to classify ring galaxies using an example dataset.
It is designed to be easy to understand.

finetune_advanced.py solves the same problem with some additional tips and tricks: filtering the dataset, balancing the classes (rings are rare), and unfreezing the model (3). 

.. note::
    
    Feel free to copy these example scripts and adapt them to your problem.
