import tensorflow as tf
import tensorflow_probability as tfp
# def calculate_binomial_loss(labels, predictions):
#     scalar_predictions = get_scalar_prediction(predictions)  # softmax, get the 2nd neuron
#     return binomial_loss(labels, scalar_predictions)


# def get_scalar_prediction(prediction):
#     return tf.nn.softmax(prediction)[:, 1]

    
# def get_indices_from_label_cols(label_cols, questions):
#     """
#     Get indices for use with tf.dynamic_slice
#     Example use:

#     questions = ['q1', 'q2']
#     label_cols = ['q1_a1', 'q1_a2', 'q2_a1', 'q2_a2']

#     Returns:
#     indices = [0, 0, 1, 1]
#     """
#     raise NotImplementedError('This has been deprecated, use get_schema above')
#     # indices = np.zeros(len(label_cols))
#     # for question_n, question in enumerate(questions):
#     #     for column_n, label_col in enumerate(label_cols):
#     #         if label_col.startswith(question):
#     #             indices[column_n] = question_n
#     # return tf.constant(indices.astype(int), dtype=tf.int32)


def dirichlet_loss(labels_for_q, concentrations_for_q):
    """
    Negative log likelihood of ``labels_for_q`` being drawn from Dirichlet-Multinomial distribution with ``concentrations_for_q`` concentrations.
    This loss is for one question. Sum over multiple questions if needed (assumes independent).

    Args:
        labels_for_q (tf.constant): observed labels (count of volunteer responses) of shape (batch, answer)
        concentrations_for_q (tf.constant): concentrations for multinomial-dirichlet predicting the observed labels, of shape (batch, answer)

    Returns:
        tf.constant: negative log. prob per galaxy, of shape (batch_dim).
    """
    total_count = tf.reduce_sum(labels_for_q, axis=1)
    dist = tfp.distributions.DirichletMultinomial(total_count, concentrations_for_q, validate_args=True)
    # print(total_count)
    # print(labels_for_q)
    # print(concentrations_for_q)
    # print(dist.prob(labels_for_q))
    return -dist.log_prob(labels_for_q)  # important minus sign


# def beta_loss(labels_for_q, alpha_for_q, beta_for_q, n_answers):
#     # labels (batch, answer)
#     # params (batch, param)
#     # print(alpha_for_q.shape)
#     # print(beta_for_q.shape)
#     total_counts = tf.reduce_sum(labels_for_q, axis=1)

#     if n_answers > 2:
#         a_losses = []
#         for answer_n in range(n_answers):
#             labels_for_a = labels_for_q[:, answer_n]
#             alpha_for_a = alpha_for_q[:, answer_n]
#             beta_for_a = beta_for_q[:, answer_n]
#             a_losses.append(beta_log_prob(labels_for_a, alpha_for_a, beta_for_a, total_counts))
#         loss_by_question = -tf.reduce_sum(a_losses, axis=0)  # 0th axis is list index,1st is batch dim. Important minus!
#     else:  # ignore the prediction for the other answer - one (alpha, beta) pair is enough. Loss only calculated for one answer (makes sense, I think)
#         return -beta_log_prob(labels_for_q[:, 0], alpha_for_q[:, 0], beta_for_q[:, 0], total_counts)  # important minus!
#     return loss_by_question

# def beta_log_prob(successes, alpha, beta, total_counts):
#     # print(total_counts.shape)  # (batch)
#     # print(successes.shape)  # (batch)
#     # print(alpha.shape)  # (batch)
#     # print(beta.shape)  # (batch)
#     # print('Alpha:', np.around(alpha.numpy(), 4))
#     # print('Beta:', np.around(beta.numpy(), 4))
#     # print('Successes: ', successes)
#     # print('Total counts: ', total_counts)
#     beta_bin_dist = tfp.distributions.BetaBinomial(total_counts, alpha, beta, validate_args=True)
#     log_prob = beta_bin_dist.log_prob(successes)  # (batch) shape output
#     # print('Log prob: ', np.around(log_prob.numpy(), 4))
#     return log_prob


# def multiquestion_beta_loss(labels, predictions, question_index_groups):
#     q_losses = []
#     for q_n in range(len(question_index_groups)):
#         q_indices = question_index_groups[q_n]
#         q_start = q_indices[0]
#         q_end = q_indices[1]
#         total_count = labels[:, q_start:q_end+1]
#         q_
    
#     # TODO binary questions should either skip second loss output or not be calculated/enforced identical



def get_multiquestion_loss(question_index_groups):
    """
    Get subclass of tf.keras.losses.Loss which wraps ``calculate_multiquestion_loss`` and sums over batch.

    Args:
        question_index_groups (list): Answer indices for each question i.e. [(question.start_index, question.end_index), ...] for all questions. Useful for slicing model predictions by question.

    Returns:
        MultiquestionLoss: see above.
    """

    class MultiquestionLoss(tf.keras.losses.Loss):

        def call(self, labels, predictions):
            return calculate_multiquestion_loss(labels, predictions, question_index_groups)

    return MultiquestionLoss(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)


def calculate_multiquestion_loss(labels, predictions, question_index_groups):
    """
    The full decision tree loss used for training GZ DECaLS models

    Negative log likelihood of observing ``labels`` (volunteer answers to all questions)
    from Dirichlet-Multinomial distributions for each question, using concentrations ``predictions``.
    
    Args:
        labels (tf.Tensor): (galaxy, k successes) where k successes dimension is indexed by question_index_groups.
        predictions (tf.Tensor):  Dirichlet concentrations, matching shape of labels
        question_index_groups (list): Paired (tuple) integers of (first, last) indices of answers to each question, listed for all questions.
    
    Returns:
        tf.Tensor: neg. log likelihood of shape (batch, question).
    """
    # very important that question_index_groups is fixed and discrete, else tf.function autograph will mess up 
    q_losses = []
    # will give shape errors if model output dim is not labels dim, which can happen if losses.py substrings are missing an answer
    for q_n in range(len(question_index_groups)):
        q_indices = question_index_groups[q_n]
        q_start = q_indices[0]
        q_end = q_indices[1]

        q_loss = dirichlet_loss(labels[:, q_start:q_end+1], predictions[:, q_start:q_end+1])

        q_losses.append(q_loss)
    
    total_loss = tf.stack(q_losses, axis=1)

    return total_loss  # leave the reduce_sum to the tf.keras.losses.Loss base class, loss should keep the batch size. 
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss will auto-reduce (sum) over the batch anyway


# def multinomial_loss(successes, expected_probs, output_dim=2):
#     """
#     For this to be correct, predictions must sum to 1 and successes must sum to n_trials (i.e. all answers to each question are known)
#     Negative log loss, of course
    
#     Args:
#         successes (tf.Tensor): (galaxy, k_successes) where k_successes is indexed by each answer (e.g. [:, 0] = smooth votes, [:, 1] = featured votes)
#         expected_probs (tf.Tensor): coin-toss probability of success, same dimensions as successes
#         output_dim (int, optional): Number of answers (i.e. successes.shape[1]). Defaults to 2. TODO may remove?
    
#     Returns:
#         tf.Tensor: neg log likelihood of k_successes observed across all answers. With batch dimension.
#     """
#     # successes x, probs p: tf.sum(x*log(p)). Each vector x, p of length k.
#     # important minus!
#     loss = -tf.reduce_sum(input_tensor=successes * tf.math.log(expected_probs + tf.constant(1e-8, dtype=tf.float32)), axis=1)

#     return loss
