
import logging

import  tensorflow as tf

from zoobot.estimators import efficientnet_standard


def define_headless_efficientnet(output_dim, input_shape=None, batch_size=None, add_channels=False, get_effnet=efficientnet_standard.EfficientNetB0, **kwargs):
    # batch size arg does nothing
    # add_channels arg does nothing

    logging.info(f'Model output dim: {output_dim}')
    model = tf.keras.models.Sequential()
    logging.info('Building efficientnet to expect input {}, after any preprocessing layers'.format(input_shape))

    # classes probably does nothing without include_top
    effnet = get_effnet(
        input_shape=input_shape,
        # input_tensor=tf.keras.Input(shape=input_shape, batch_size=batch_size),
        weights=None,
        include_top=False,
        classes=output_dim,
        **kwargs
    )
    model.add(effnet)

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    # note - no dropout on final layer

    # custom_top_multinomial(model, output_dim, schema, batch_size)
    # custom_top_dirichlet(model, output_dim, schema)        

    # will be updated by callback
    # model.step = tf.Variable(0, dtype=tf.int64, name='model_step', trainable=False)

     # my loss only works with run_config shards, the new custom shards are vote fraction labelled
    # loss_func = lambda x, y: losses.multiquestion_loss(x, y, question_index_groups=schema.question_index_groups)

    # compile with something like:
    # custom_mses = [bayesian_estimator_funcs.CustomMSEByColumn(name=q.text, start_col=start_col, end_col=end_col) for q, (start_col, end_col) in schema.named_index_groups.items()]
    # model.compile(
    #     loss=loss_func,
    #     optimizer=tf.keras.optimizers.Adam(),
    #     metrics=custom_mses
    # )

    return model


def custom_top_dirichlet(model, output_dim):
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
#     x = tf.keras.Input(shape=dense_units)  # not including batch in shape, but still has (unknown) batch dim
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
