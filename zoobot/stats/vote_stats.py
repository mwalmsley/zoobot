# import tensorflow as tf

# from zoobot.stats import dirichlet_stats


# def get_expected_votes_ml(samples, q, retirement, schema, round_votes):
#     prob_of_answers = dirichlet_stats.dirichlet_prob_of_answers(samples, schema)  # mean over both models. Prob given q is asked!
#     prev_q = q.asked_after
#     if prev_q is None:
#         expected_votes = tf.ones(len(samples)) * retirement
#     else:
#         joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
#         expected_votes = joint_p_of_asked * retirement
#     if round_votes:
#         return tf.round(expected_votes)
#     else:
#         return expected_votes


# def get_expected_votes_human(label_df, q, retirement, schema, round_votes):
#     prob_of_answers = label_df[[a + '_fraction' for a in schema.label_cols]].values
#     prev_q = q.asked_after
#     if prev_q is None:
#         expected_votes = tf.ones(len(label_df)) * retirement
#     else:
#         joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
#         expected_votes = joint_p_of_asked * retirement
#     if round_votes:
#         return tf.round(expected_votes)
#     else:
#         return expected_votes
