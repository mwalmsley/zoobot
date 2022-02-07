
import logging

import  tensorflow as tf

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
    model = tf.keras.models.Sequential()
    logging.info('Building efficientnet to expect input {}, after any preprocessing layers'.format(input_shape))


    if use_imagenet_weights:
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


def custom_top_dirichlet(model, output_dim):
    """
    Final dense layer used in GZ DECaLS (after global pooling). 
    ``output_dim`` neurons with an activation of ``tf.nn.sigmoid(x) * 100. + 1.``, chosen to ensure 1-100 output range
    This range is suitable for parameters of Dirichlet distribution.

    Args:
        model (tf.keras.Model): Model to which to add this dense layer
        output_dim (int): Dimension of dense layer e.g. 34 for decision tree with 34 answers
    """
    # model.add(tf.keras.layers.Dense(output_dim, activation=lambda x: tf.nn.sigmoid(x) * 20. + .2))  # one params per answer, 1-20 range (mebe fractionally worse)
    model.add(tf.keras.layers.Dense(output_dim, activation=lambda x: tf.nn.sigmoid(x) * 100. + 1.))  # one params per answer, 1-100 range


# def custom_top_multinomial(model, output_dim, schema, batch_size):
#     model.add(tf.keras.layers.Dense(output_dim))
#     model.add(tf.keras.layers.Lambda(lambda x: tf.concat([tf.nn.softmax(x[:, q[0]:q[1]+1]) for q in schema.question_index_groups], axis=1), output_shape=[batch_size, output_dim]))        

# def custom_top_beta(model, output_dim, schema):
#     model.add(tf.keras.layers.Dense(output_dim * 2, activation=lambda x: tf.nn.sigmoid(x) * 100. + 1.))  # two params, 1-100 range
    # model.add(tf.keras.layers.Reshape((output_dim, 2)))  # as dimension 2



# def custom_top_dirichlet_reparam(model, output_dim, schema):

#     dense_units = (len(schema.answers) + len(schema.questions))
#     model.add(tf.keras.layers.Dense(
#         dense_units, 
#         # kernel_initializer=tf.keras.initializers.Ones(),
#         # bias_initializer=tf.keras.initializers.Zeros(),
#         activation=None
#         # bias_constraint=tf.keras.constraints.NonNeg(),
#         # kernel_constraint=tf.keras.constraints.NonNeg()
#         ))  # one m per answer, one s per question

#     # keras functional model, split architecture
#     x = tf.keras.layers.InputLayer(input_shape=dense_units)  # not including batch in shape, but still has (unknown) batch dim
#     # not really a for loop, just constructing the graph
#     n_answers = len(schema.answers)
#     alpha_list = []
#     for q_n in range(len(schema.question_index_groups)):
#         q_indices = schema.question_index_groups[q_n]
#         q_start = q_indices[0]
#         q_end = q_indices[1]
#         q_mean = 1e-8 + tf.nn.softmax(x[:, q_start:q_end+1])
#         # might also use a softmax here to constrain to (0.01, 10) or similar. Currently any pos. value (via abs or relu)
#         q_precision = tf.expand_dims(0.01 + tf.math.abs(x[:, n_answers + q_n]), axis=1)  # expand_dims to avoid slicing axis 1 away
#         # q_precision = tf.expand_dims(0.01 + x[:, n_answers + q_n], axis=1)  # expand_dims to avoid slicing axis 1 away
#         q_alpha = q_mean * q_precision
#         alpha_list.append(q_alpha)
#     alpha = tf.concat(alpha_list, axis=1)
#     # print(x.shape, alpha.shape)
#     top_model = tf.keras.Model(inputs=x, outputs=alpha)

#     model.add(top_model)
