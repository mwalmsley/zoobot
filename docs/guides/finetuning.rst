.. _finetuning_guide:

Finetuning
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

You will likely only need a small amount of labelled images; a few hundred is a good starting point. 
This is because Zoobot includes a classifier already trained to answer Galaxy Zoo questions for DECaLS galaxies.
Retraining (finetuning) this model requires much less time and labels than starting from scratch.
If you do want to start from scratch, to reproduce or improve upon the pretrained classifier, :ref:`Zoobot can do that as well <training_from_scratch>`.

.. note:: 

    This guide uses code for the TensorFlow version of Zoobot.
    The conceptual approach is exactly the same with the PyTorch version.
    I haven't written examples yet - sorry! 
    For now, here are two links for fine-tuning vision models with `PyTorch Vision <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/computer_vision_fine_tuning.py>`__ and `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/computer_vision_fine_tuning.py>`__.


Examples
---------------------

Zoobot includes two complete working examples:

`finetune_minimal.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/finetune_minimal.py>`_ shows how to partially finetune a model (1 and 2 only) to classify ring galaxies using an example dataset.
It is designed to be easy to understand.

`finetune_advanced.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/finetune_advanced.py>`_ solves the same problem with some additional tips and tricks: filtering the dataset, balancing the classes (rings are rare), and unfreezing the model (3). 

Readers familiar with python and machine learning may prefer to read, copy and adapt the example scripts directly. 

Below, for less familiar readers, I walk through the `finetune_minimal.py <https://github.com/mwalmsley/zoobot/blob/main/zoobot/tensorflow/examples/finetune_minimal.py>`__ example in detail.

Background
---------------------

Fine-tuning, also known as transfer learning, is when a model trained on one task is partially retrained for use on another related task.
This can drastically reduce the amount of labelled data needed.
For a general introduction, see `this excellent blog post <https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>`_.


Load Pretrained Model
---------------------

Neural networks, like any statistical model, are trained by fitting their free parameters to data.
The free parameters in neural networks are called weights.

Recreating a previously trained neural network takes two steps: defining the model (how the neurons connect) and then loading the weights.
:meth:`zoobot.estimators.define_model.load_model` does both steps for you. 
It defines the model used for GZ DECaLS, and then loads the weights for that model.
By passing ``include_top=False``, the final layers of the network (those used to make a prediction) will not be loaded.
We will add our own top layers shortly.

.. code-block:: 

    base_model = define_model.load_model(
        pretrained_checkpoint,
        include_top=False,  # do not include the head used for GZ DECaLS - we will add our own head
        input_size=requested_img_size,
        crop_size=crop_size,  # model augmentation layers apply a crop...
        resize_size=resize_size,  # ...and then apply a resize
        output_dim=None  # headless so no effect
    )

We would like to retrain this model to work well on a new, related problem - here, classifying ring galaxies.
We don't have enough data to retrain the whole model - GZ DECaLS required hundreds of thousands of labels.
Instead, we will freeze the model (without any top layers) and then add our own top layers.
We can then train only the new top layer.

.. code-block:: 

    base_model.trainable = False  # freeze the headless model (no training allowed)


Replace Output Layers
---------------------

Our top layer needs to be small enough that we can train it with very little data.
A few dense layers will do.

Here, I use two Dense layers with very aggressive dropout to prevent overfitting.
Combining several dense layers allows for nonlinear behaviour, which can be useful.

The final layer is a single neuron with a sigmoid activation function.
This will always give an output between 0 and 1, which is appropriate for binary classification.
For multiclass classification, I might instead use 
``layers.Dense(3, activation="softmax", name="softmax_output")``.

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

Now we stick the new head on top of the pretrained base model:

.. code-block:: 

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(requested_img_size, requested_img_size, 1)),
        base_model,
        new_head
    ])

Train 
-----------

The base model remains frozen, while the head is free to train (as we never set ``new_head.trainable = False``).
Training the overall model will therefore only affect the new head.

For a binary classification problem, I am using the binary cross-entropy.
Other types of problem will need different losses.
I am using the adam optimizer, which is nearly always a great choice - it's very robust!

.. code-block:: 

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal learning rate is okay
        metrics=['accuracy']
    )

I define how I would like my model to be trained, with 80 epochs and stopping early if the validation loss does not improve after 10 consecutive epochs:

.. code-block:: 

    train_config = training_config.TrainConfig(
        log_dir='save/model/here',
        epochs=80,
        patience=10  # early stopping: if val loss does not improve for this many epochs in a row, end training
    )

And then we train!

.. code-block:: 

    training_config.train_estimator(
        model,
        train_config,  # how to train e.g. epochs, patience
        train_dataset,
        val_dataset
    )

``model`` has now been fit to the training data. You can use it to make new predictions - see the full example for more.

The new weights, including the new head, have been saved to log_dir/checkpoint. 
You can load them at any time to make predictions later or continue training - just be sure to define your model, including the new head, in exactly the same way.

Now go do some science!
