
import logging

import tensorflow as tf

from zoobot.tensorflow.estimators import efficientnet_standard


def define_headless_efficientnet(input_shape=None, get_effnet=efficientnet_standard.EfficientNetB0, use_imagenet_weights=False, **kwargs):
    """
    Define efficientnet model to train.
    Thin wrapper around ``get_effnet``, an efficientnet creation function from ``efficientnet_standard``, that ensures the appropriate args.

    Additional keyword arguments are passed to ``get_effnet``.

    Args:
        input_shape (tuple, optional): Expected input shape e.g. (224, 224, 1). Defaults to None.
        get_effnet (function, optional): Efficientnet creation function from ``efficientnet_standard``. Defaults to efficientnet_standard.EfficientNetB0.
    
    Returns:
        [type]: [description]
    """
    model = tf.keras.models.Sequential(name='headless_efficientnet')
    logging.info('Building efficientnet to expect input {}, after any preprocessing layers'.format(input_shape))


    if use_imagenet_weights:
        logging.warning('Using imagenet weights - not recommended!')
        weights = 'imagenet'  # split variable names to be clear this isn't one of my checkpoints to load
    else:
        weights = None

    # classes probably does nothing without include_top
    effnet = get_effnet(
        input_shape=input_shape,
        weights=weights,
        include_top=False,  # no final three layers: pooling, dropout and dense
        classes=None,  # headless so has no effect
        **kwargs
    )
    model.add(effnet)

    return model


def custom_top_dirichlet(output_dim):
    """
    Final dense layer used in GZ DECaLS (after global pooling). 
    ``output_dim`` neurons with an activation of ``tf.nn.sigmoid(x) * 100. + 1.``, chosen to ensure 1-100 output range
    This range is suitable for parameters of Dirichlet distribution.

    Use with Functional API e.g. x = custom_top_dirichlet(output_dim)(x)

    Args:
        output_dim (int): Dimension of dense layer e.g. 34 for decision tree with 34 answers

    Returns:
        (tf.keras.layers.Dense): suitable for predicting Dirichlet distributions, as above.
    """
    return tf.keras.layers.Dense(output_dim, activation=lambda x: tf.nn.sigmoid(x) * 100. + 1.)  # one params per answer, 1-100 range
