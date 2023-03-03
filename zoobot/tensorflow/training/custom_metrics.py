import tensorflow as tf

from zoobot.tensorflow.training import losses


class LossPerQuestion(tf.keras.metrics.Metric):

  def __init__(self, question_index_groups, name='loss_per_question'):
    super(LossPerQuestion, self).__init__(name=name)

    self.question_index_groups = question_index_groups

    self.multiq_loss_func = losses.get_multiquestion_loss(question_index_groups=self.question_index_groups, reduction=tf.keras.losses.Reduction.NONE)  # doesn't sum over questions    

    # dict of weights, keyed by question_n
    self.question_weights = {}
    for question_n in range(len(self.question_index_groups)):
        self.question_weights[question_n] = self.add_weight(name=f'questions/question_{question_n}_loss', initializer='zeros')

    # also track num. galaxies
    self.num_galaxies = self.add_weight(name='num_galaxies', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):

    multiq_loss = tf.cast(self.multiq_loss_func(y_true, y_pred), tf.float32)

    for question_n in range(len(self.question_index_groups)):
        total_loss_for_question_in_batch = tf.reduce_sum(multiq_loss[:, question_n])
        self.question_weights[question_n].assign_add(total_loss_for_question_in_batch)

    self.num_galaxies.assign_add(tf.cast(len(y_true), dtype=tf.float32))
 

  def result(self):

    metric_result = {}
    for weight in self.question_weights.values():
      # tf.print(weight/self.num_galaxies)
      # .ref() is the hashable string that you'd imagine .name would give, .name is some unhashable weird TF object 
      metric_result[weight.name] = weight/self.num_galaxies  # total loss for q across all batches, divide by total num galaxies

    # tf.print(metric_result)
    # return weight/self.num_galaxies
    return metric_result
    # return {'something': self.question_weights[0], 'something_else': self.num_galaxies}
    # TODO rename with 
    # return self.question_weights[0]
