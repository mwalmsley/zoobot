.. _choosing_parameters:

Choosing Parameters
=====================================

All FinetuneableZoobot classes share a common set of parameters for controlling the finetuning process. These can have a big effect on performance.


Finetuning is fast and easy to experiment with, so we recommend trying different parameters to see what works best for your dataset.
This guide provides some explanation for each option.

We list the key parameters below in rough order of importance. 
See :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract` for the full list of parameters.

``learning_rate``
...............................

Learning rate sets how fast the model parameters are updated during training.
Zoobot uses the adaptive optimizer ``AdamW``.
Adaptive optimizers adjust the learning rate for each parameter based on the mean and variance of the previous gradients.
This means you don't need to tune the learning rate as carefully as you would with a fixed learning rate optimizer like SGD.
We find a learning of 1e-4 is a good starting point for most tasks.

If you find the model is not learning, you can try increasing the learning rate.
If you see the model loss is varying wildly, or the train loss decreases much faster than the validation loss (overfitting), you can try decreasing the learning rate.
Increasing ``n_blocks`` (below) often requires a lower learning rate, as the model will adjust more parameters for each batch.


``n_blocks``
...............................

Deep learning models are often divided into blocks of layers. 
For example, a ResNet model might have 4 blocks, each containing a number of convolutional layers. 
The ``n_blocks`` parameter specifies how many of these blocks to finetune. 

By default, ``n_blocks=0``, and so only the head is finetuned.
This is a good choice when you have a small dataset or when you want to avoid overfitting.
Finetuning only the head is sometimes called transfer learning. 
It's equivalent to calculating representations with the pretrained model and then training a new one-layer model on top of those representations.

You can experiment with increasing ``n_blocks`` to finetune more of the model.
This works best for larger datasets (typically more than 1k examples).
To finetune the full model, keep increasing ``n_blocks``; Zoobot will raise an error if you try to finetune more blocks than the model has.
Our recommended encoder, ``ConvNext``, has 5 blocks.


``lr_decay``
...............................

The common intuition for deep learning is that lower blocks (near the input) learn simple general features and higher blocks (near the output) learn more complex features specific to your task.
It is often useful to adjust the learning rate to be lower for lower blocks, which have already been pretrained to recognise simple galaxy features.

Learning rate decay reduces the learning rate by block.
For example, with ``learning_rate=1e-4`` and ``lr_decay=0.75`` (the default):

* The highest block has a learning rate of 1e-4 * (0.75^0) = 1e-4
* The second-highest block has a learning rate of 1e-4 * (0.75^1) = 7.5e-5
* The third-highest block has a learning rate of 1e-4 * (0.75^2) = 5.6e-5

and so on.

Decreasing ``lr_decay`` will exponentially decrease the learning rate for lower blocks.

In the extreme cases:

* Setting ``learning_rate=0`` will disable learning in all blocks except the first block (0^0=1), equivalent to ``n_blocks=1``.
* Setting ``lr_decay=1`` will give all blocks the same learning rate.

The head always uses the full learning rate.

``weight_decay``
...............................

Weight decay is a regularization term that penalizes large weights.
When using Zoobot's default ``AdamW`` optimizer, it is closely related to L2 regularization, though there's some subtlety - see https://arxiv.org/abs/1711.05101.
Increasing weight decay will increase the penalty on large weights, which can help prevent overfitting, but may slow or even stop training.
By default, Zoobot uses a small weight decay of 0.05.


``dropout_prob``
...............................

Dropout is a regularization technique that randomly sets some activations to zero during training.
Similarly to weight decay, dropout can help prevent overfitting.
Zoobot uses a dropout probability of 0.5 by default.


``cosine_schedule`` and friends
.................................

Gradually reduce the learning rate during training can slightly improve results by finding a better minimum near convergence.
This process is called learning rate scheduling.
Zoobot includes a cosine learning rate schedule, which reduces the learning rate according to a cosine function.

The cosine schedule is controlled by the following parameters:

* ``cosine_schedule`` to enable the scheduler. 
* ``warmup_epochs`` to linearly increase the learning rate from 0 to ``learning_rate`` over the first ``warmup_epochs`` epochs, before applying cosine scheduling.
* ``max_cosine_epochs`` sets how many epochs it takes to decay to the final learning rate (below). Warmup epochs don't count.
* ``max_learning_rate_reduction_factor`` controls the final learning rate (``learning_rate`` * ``max_learning_rate_reduction_factor``).
 