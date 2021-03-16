Fine-Tuning a Pre-Trained CNN
=============================

Fine-tuning, aka transfer learning, is when a model trained on one task is partially retrained for use on another related task (or similar data).
Hopefully, the model will have learned an effective representation of the data from the first task, and so might perform better on the related task than a model trained from scratch.
For a general introduction, see the excellent blog post at `<https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>`_.

The CNN have been trained to solve a variety of morphology tasks (i.e. to simultaneously answer all of the Galaxy Zoo questions) and so is likely to have learned a useful general representation of galaxy morphology.
You can benefit from this general representation by fine-tuning the CNN for your task.

The general approach is:

1. Load the pretrained model, replacing the head (output layers) to match your task
2. Train *only* the new head, leaving the rest of the model frozen
3. Optionally, once the new head is trained, unfreeze the rest of the model and train with a low learning rate

There is a complete working example (for 1. and 2.) at ``finetune.py``.
