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

  def update_state(self, y_true, y_pred, sample_weight=None):

    multiq_loss = self.multiq_loss_func(y_true, y_pred)

    for question_n in range(len(self.question_index_groups)):
        mean_loss_for_question = tf.reduce_mean(multiq_loss[:, question_n])
        self.question_weights[question_n].assign_add(mean_loss_for_question)

  def result(self):
    # TODO rename with 
    return self.question_weights
