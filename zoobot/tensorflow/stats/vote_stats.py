import tensorflow as tf
import numpy as np

from zoobot.tensorflow.stats import dirichlet_stats


def get_expected_votes_ml(concentrations, question, votes_for_base_question: int, schema, round_votes):
            # (send all concentrations not per-question concentrations, they are all potentially relevant)
    prob_of_answers = dirichlet_stats.dirichlet_prob_of_answers(concentrations, schema)  # mean over both models. Prob given q is asked!
    prev_q = question.asked_after
    if prev_q is None:
        expected_votes = tf.ones(len(concentrations)) * votes_for_base_question
    else:
        joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
        expected_votes = joint_p_of_asked * votes_for_base_question
    if round_votes:
        return tf.round(expected_votes)
    else:
        return expected_votes


def get_expected_votes_human(label_df, question, votes_for_base_question: int, schema, round_votes):
    # this expects 
    
    prob_of_answers = label_df[[a + '_fraction' for a in schema.label_cols]].values
    # some will be nan as fractions are often nan (as 0 of 0)

    prev_q = question.asked_after
    if prev_q is None:
        expected_votes = tf.ones(len(label_df)) * votes_for_base_question
    else:
        joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
        # for humans, its just votes_for_base_question * the product of all the fractions leading to that q
        expected_votes = joint_p_of_asked * votes_for_base_question
    if round_votes:
        return tf.round(expected_votes)
    else:
        return expected_votes
