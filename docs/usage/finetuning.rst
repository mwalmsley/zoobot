.. _finetuning:

Finetuning Guide
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

Examples
---------------------

Zoobot includes two complete working examples:

`finetune_minimal.py <https://github.com/mwalmsley/zoobot/blob/main/finetune_minimal.py>`_ shows how to partially finetune a model (1 and 2 only) to classify ring galaxies using an example dataset.
It is designed to be easy to understand.

`finetune_advanced.py <https://github.com/mwalmsley/zoobot/blob/main/finetune_advanced.py>`_ solves the same problem with some additional tips and tricks: filtering the dataset, balancing the classes (rings are rare), and unfreezing the model (3). 

Readers familiar with python and machine learning may prefer to read, copy and adapt the example scripts directly. 

Below, for less familiar readers, I walk through the `finetune_minimal.py <https://github.com/mwalmsley/zoobot/blob/main/finetune_minimal.py>`__ example in detail.

Background
---------------------

Fine-tuning, also known as transfer learning, is when a model trained on one task is partially retrained for use on another related task.
This can drastically reduce the amount of labelled data needed.
For a general introduction, see `this excellent blog post <https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>`_.


Load Pretrained Model
---------------------

.. code-block:: 

    base_model = define_model.load_model(
        pretrained_checkpoint,
        include_top=False,  # do not include the head used for GZ DECaLS - we will add our own head
        input_size=requested_img_size,
        crop_size=crop_size,  # model augmentation layers apply a crop...
        resize_size=resize_size,  # ...and then apply a resize
        output_dim=None  # headless so no effect
    )

.. code-block:: 

    base_model.trainable = False  # freeze the headless model (no training allowed)


Replace Output Layers
---------------------

.. code-block:: 

    new_head = tf.keras.Sequential([
        layers.InputLayer(input_shape=(7,7,1280)),  # base model output shape
        layers.GlobalAveragePooling2D(),
        # TODO the following layers will likely need some experimentation
        layers.Dropout(0.75),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.75),
        layers.Dense(1, activation="sigmoid", name='sigmoid_output')
    ])

output should be one neuron w/ sigmoid for binary classification...

layers.Dense(3, activation="softmax", name="softmax_output")  # ...or N neurons w/ softmax for N-class classification


Stick the new head on the pretrained base model

.. code-block:: 

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(requested_img_size, requested_img_size, 1)),
        base_model,
        new_head
    ])

Train 
-----------

.. code-block:: 

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal learning rate is okay
        metrics=['accuracy']
    )

.. code-block:: 

    train_config = training_config.TrainConfig(
        log_dir='save/model/here',
        epochs=80,
        patience=10  # early stopping: if val loss does not improve for this many epochs in a row, end training
    )

    training_config.train_estimator(
        model,
        train_config,  # how to train e.g. epochs, patience
        train_dataset,
        val_dataset
    )

Will save to...