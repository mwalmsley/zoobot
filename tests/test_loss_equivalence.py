import pytest

import tensorflow_probability as tfp
import torch
import numpy as np

from zoobot.tensorflow.training import losses as tf_losses
from zoobot.pytorch.training import losses as torch_losses


@pytest.fixture
def single_question_context():
    # use factory pattern to get different values each time called within a test
    # https://docs.pytest.org/en/6.2.x/fixture.html#factories-as-fixtures
    def _single_question_context(batch_size=10, answers=3):
        total_count = np.ones(batch_size) * 5
        concentrations = np.random.rand(batch_size, answers)
        # generated with tfp, but just to get some mock labels - doesn't bias
        labels = tfp.distributions.DirichletMultinomial(total_count, concentrations).sample().numpy()
        return labels, total_count, concentrations
    return _single_question_context


def test_single_question_loss(single_question_context):

    labels, total_count, concentrations = single_question_context()

    # tf will broadcast total_count e.g. 5 -> [5, 5, 5...] by batch shape
    # tf will also accept int or float total_counts
    log_prob_tf = tf_losses.get_dirichlet_neg_log_prob(labels, total_count, concentrations).numpy()

    # pyro won't broadcast total count, needs to be = batch shape
    # pyro will only accept total_count dtype = labels dtype
    log_prob_torch = torch_losses.get_dirichlet_neg_log_prob(torch.from_numpy(labels), torch.Tensor(total_count), torch.from_numpy(concentrations)).numpy()

    assert np.isclose(log_prob_tf, log_prob_torch).all()


def test_multiquestion_loss(single_question_context):
    labels_a, _, concentrations_a = single_question_context(answers=2)
    labels_b, _, concentrations_b = single_question_context(answers=3)

    # assert not np.isclose(labels_a, labels_b).all()  # will be diffferent data due to factory fixture

    labels_both = np.concatenate([labels_a, labels_b], axis=1)
    # total_count added up per question within func
    concentrations_both = np.concatenate([concentrations_a, concentrations_b], axis=1)

    question_index_groups = [(0, 1), (2, 4)]  # indices 0,1 for q a, 2,3,4 for q b

    # tf accepts np args
    log_prob_tf = tf_losses.calculate_multiquestion_loss(labels_both, concentrations_both, question_index_groups).numpy()
    log_prob_torch = torch_losses.calculate_multiquestion_loss(torch.from_numpy(labels_both), torch.from_numpy(concentrations_both), question_index_groups).numpy()

    assert np.isclose(log_prob_tf, log_prob_torch).all()
