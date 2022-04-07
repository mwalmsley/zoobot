import json
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from zoobot.tensorflow.stats import mixture_stats

class EqualMixture():

    def __init__(self):
        """
        Mixture of abstract distributions.
        Abstract class implementing broadcasting along the final axis (which model) for common funcs e.g. log_prob, mean_log_prob etc

        Raises:
            NotImplementedError: Abstract class!
        """
        raise NotImplementedError

    @property
    def batch_shape(self):
        return self.distributions[0].batch_shape

    @property
    def event_shape(self):
        return self.distributions[0].event_shape
    
    def log_prob(self, x):
        log_probs_by_dist = [dist.log_prob(x) for dist in self.distributions]
        return tf.transpose(log_probs_by_dist)
    
    def prob(self, x):  # just the same, but prob not log prob
        return tf.math.exp(self.log_prob(x))

    def mean_prob(self, x):
        prob = self.prob(x)
        return tf.reduce_mean(prob, axis=-1)

    def mean_log_prob(self, x):
        return tf.math.log(self.mean_prob(x))

    def mean(self):  # i.e. expected 
        return tf.reduce_mean([dist.mean() for dist in self.distributions], axis=0)

    def mean_cdf(self, x):  # as integration is separable 
        # will fail as cdf is not implemented
        return tf.reduce_mean([dist.cdf(x) for dist in self.distributions], axis=0)

    def cdf(self, x):
        raise NotImplementedError('Can only be calculated with batch_dim shaped x, unlike .prob etc, needs custom implementation')



class DirichletEqualMixture(EqualMixture):

    def __init__(self, concentrations):
        """
        Equally-weighted mixture of Dirichlet distributions.
        Basically, the Dirichlet version of a Gaussian mixture model.
        Useful for combining predictions from multiple models or multiple forward passes (with MC Dropout and augmentations).

        Specifically, self.distributions will be a list of tfp.distributions.Dirichlet, each with concentrations (galaxies, answers, specific model_index)
        Broadcasting works along galaxies dimension by subclassing ``EqualMixture``.

        Args:
            concentrations (np.ndarray): predicted dirichlet parameters for one galaxy, of shape (galaxies, answers, models)
        """
        self.concentrations = concentrations.astype(np.float32)
        self.n_distributions = self.concentrations.shape[2]
        self.distributions = [
            tfp.distributions.Dirichlet(concentrations[:, :, n], validate_args=True)
            for n in range(self.n_distributions)
        ]

    def entropy_estimate(self):
        """
        Returns:
            np.ndarray: midpoint between entropy bounds
        """
        upper = self.entropy_upper_bound()
        lower = self.entropy_lower_bound()
        return lower + (upper-lower)/2.  # midpoint between bounds

    def entropy_upper_bound(self):
        """
        Returns:
            np.ndarray: upper bound on entropy. See ``mixture_stats.entropy_upper_bound``.
        """
        return np.array([mixture_stats.entropy_upper_bound(galaxy_conc, weights=np.ones(self.n_distributions)/self.n_distributions) for galaxy_conc in self.concentrations])

    def entropy_lower_bound(self):
        """
        Returns:
            np.ndarray: lower bound on entropy. See ``mixture_stats.entropy_lower_bound``.
        """
        return np.array([mixture_stats.entropy_lower_bound(galaxy_conc, weights=np.ones(self.n_distributions)/self.n_distributions) for galaxy_conc in self.concentrations])


    # def to_beta(self, answer_index, batch_dim):
    #     """
    #     Convert from Dirichlet (multivariate) to Beta (bivariate) by adding up all the parameters to every answer but one.
    #     E.g. 3-variate dirichlet(concentrations=[2, 3, 6]) converts to bivariate beta(a=2, b=3+6) for ``answer_index=0``
    #     Intuitively, this is like going from "is it a, b, or c?" to "is it a or not-a?"

    #     Args:
    #         answer_index (int): which answer to use as first concentration. Other answers will have concentrations summed.
    #         batch_dim (int): batch dimension i.e. num dirichlet distributions in this mixture.

    #     Returns:
    #         BetaEqualMixture: beta(answer index, not answer index) with a batch dimension of num. dirichlet distributions in this mixture
    #     """
    #     assert self.batch_shape == 1  # beta uses batch shape for cdf(x), so needs to be able to broadcast, so can only do one galaxy at a time
    #     answer_concentrations = self.concentrations[:, answer_index]
    #     # may give index errors
    #     other_concentrations = self.concentrations[:, :answer_index].sum(axis=1) + self.concentrations[:, answer_index+1:].sum(axis=1)
    #     beta_concentrations = np.stack([answer_concentrations, other_concentrations], axis=1)
    #     return BetaEqualMixture(beta_concentrations, batch_dim=batch_dim)



# class BetaEqualMixture(DirichletEqualMixture):

#     def __init__(self, concentrations, batch_dim):
#         """
#         Batch dim is the number of models in the mixture, 

#         Only supports galaxy_dim = 1 i.e. one galaxy at a time.
    
#         As with DirichletEqualMixture, self.distributions is list of tfp.distributions.Beta, each with dimension (galaxy, answer)

#         Only used for calculating confidence intervals for a specific answer - which is currently impractically slow anyway.

#         Args:
#             concentrations ([type]): [description]
#             batch_dim ([type]): [description]
#         """
#         assert concentrations.shape[0] == 1
#         assert concentrations.shape[1]  == 2
#         # print(concentrations.shape)
#         self.batch_dim = batch_dim
#         self.concentrations = np.stack([np.squeeze(concentrations) for _ in range(batch_dim)], axis=0).astype(np.float32)
#         # print(self.concentrations.shape)
#         assert self.concentrations.ndim == 3
#         self.n_distributions = self.concentrations.shape[2]
#         self.distributions = [
#             tfp.distributions.Beta(
#                 concentration0=self.concentrations[:, 0, n],
#                 concentration1=self.concentrations[:, 1, n],
#                 validate_args=True)
#             for n in range(self.n_distributions)
#         ]

#     @property
#     def standard_sample_grid(self):
#         return np.linspace(1e-8, 1. - 1e-8, num=self.batch_dim)

#     def mean_mode(self):  # approximate only
#         assert self.batch_dim >= 50
#         # print(self.mean_prob(self.standard_sample_grid).shape)
#         mode_index = np.argmax(self.mean_prob(self.standard_sample_grid))
#         return self.standard_sample_grid[mode_index]  # bit lazy but it works

#     def confidence_interval(self, interval_width):
#         assert self.batch_dim >= 50
#         # dist must be unimodal (though the mode can be at extremes)
#         cdf = self.mean_cdf(self.standard_sample_grid)
#         mode = self.mean_mode()
#         # print(mode)
#         mode_cdf = self.mean_cdf(mode)[0]  # just using the first, batch size broadcasting gives batch_size identical results
#         # print(mode_cdf)
#         return confidence_interval_from_cdf(self.standard_sample_grid, cdf, mode_cdf, interval_width)

#     def cdf(self, x):
#         # for dist in self.distributions:
#             # dist_with_batch_matching_x = tfp.distributions.Beta()
#         # print(x)
#         cdfs = [dist.cdf(x) for dist in self.distributions]
#         return tf.transpose(cdfs)


class DirichletMultinomialEqualMixture(EqualMixture):

    def __init__(self, total_votes, concentrations, *args, **kwargs):
        """

        Mixture of equally-weighted Dirichlet-Multinomial distributions.
        Useful for interpreting the aggregate predictions of many models in terms of expected volunteer responses.

        Uses the same interface as tfp, but can aggregate across models (by subclassing ``EqualMixture``).

        Args:
            total_votes (np.ndarray): total votes for some question, of shape (galaxies)
            concentrations (np.ndarray): of shape (galaxies, answer, model). Answers for one question only due to total votes.
        """
        
        self.total_votes = np.array(total_votes).astype(np.float32)
        self.concentrations = concentrations.astype(np.float32)
        self.n_distributions = self.concentrations.shape[2]
        self.distributions = [
            tfp.distributions.DirichletMultinomial(self.total_votes, self.concentrations[:, :, n], validate_args=True)
            for n in range(self.n_distributions)
        ]

def load_all_concentrations(df, concentration_cols):
    temp = []
    for col in concentration_cols:
        temp.append(np.stack(df[col].apply(json.loads).values, axis=0))
    return np.stack(temp, axis=2).transpose(0, 2, 1)


def dirichlet_prob_of_answers(concentrations, schema, temperature=None):
    # badly named vs posteriors, actually gives predicted vote fractions of answers...
    # mean probability (including dropout) of an answer being given. 
    # concentrations has (batch, answer, dropout) shape
    p_of_answers = []
    for q in schema.questions:
        concentrations_by_q = concentrations[:, q.start_index:q.end_index+1, :]
        p_of_answers.append(DirichletMultinomialEqualMixture(total_votes=1, concentrations=concentrations_by_q).mean().numpy())

    p_of_answers = np.concatenate(p_of_answers, axis=1)
    return p_of_answers



# # only used for posthoc evaluation, not when training
# def dirichlet_mixture_loss(labels, predictions, question_index_groups):  # pasted
#     q_losses = []
#     for q_n in range(len(question_index_groups)):
#         q_indices = question_index_groups[q_n]
#         q_start = q_indices[0]
#         q_end = q_indices[1]
#         q_loss = dirichlet_mixture_loss_per_question(labels[:, q_start:q_end+1], predictions[:, q_start:q_end+1])
#         q_losses.append(q_loss)
    
#     total_loss = np.stack(q_losses, axis=1)
#     return total_loss  # leave the reduce_sum to the estimator

# # this works but is very slow
# # def dirichlet_mixture_loss_per_question(labels_q, predictions_q):
# #     n_samples = predictions_q.shape[-1]
# #     total_votes = labels_q.sum(axis=1).squeeze()
# #     log_probs = []
# # #     print(predictions_q.shape,total_votes.shape, n_samples)
# #     for n, galaxy in enumerate(predictions_q):
# #         mixture = acquisition_utils.dirichlet_mixture(np.expand_dims(galaxy, axis=0), total_votes[n], n_samples)
# #         log_probs.append(mixture.log_prob(labels_q[n]))
# #     return -np.squeeze(np.array(log_probs))  # negative log prob

# def dirichlet_mixture_loss_per_question(labels_q, concentrations_q):
#     n_samples = concentrations_q.shape[-1]
#     total_votes = labels_q.sum(axis=1).squeeze()
#     mean_log_probs = DirichletMultinomialEqualMixture(total_votes=total_votes, concentrations=concentrations_q).mean_log_prob(labels_q) 
#     return -np.squeeze(np.array(mean_log_probs))  # negative log prob


# def beta_confidence_interval(concentration0, concentration1, interval_width):

#     dist = tfp.distributions.Beta(concentration0, concentration1, validate_args=True, allow_nan_stats=False)
#     if concentration0 <= 1:
#         mode = 1
#     elif concentration1 <= 1:
#         mode = 0
#     else:
#         mode = dist.mode()
#     mode_cdf = dist.cdf(mode)

#     x = np.linspace(0.001, .999, num=1000)
#     cdf = dist.cdf(x)
#     return confidence_interval_from_cdf(x, cdf, mode_cdf, interval_width)


# def confidence_interval_from_cdf(x, cdf, mode_cdf, interval_width):
#     width_per_half = interval_width / 2

#     lower_index = np.argmin(np.abs(mode_cdf  - cdf - width_per_half))
#     upper_index = np.argmin(np.abs(cdf - mode_cdf - width_per_half))

#     if mode_cdf < width_per_half:  # i.e. if lower interval hits the edge
#         remaining_cdf = mode_cdf  # b
#         upper_index = np.argmin(np.abs(cdf - interval_width))
#     elif 1 - mode_cdf < width_per_half:  # i.e. upper interval hits the edge
#         remaining_cdf = mode_cdf  # b
#         lower_index = np.argmin(np.abs(1 - cdf - interval_width))

#     assert np.allclose(cdf[upper_index] - cdf[lower_index], interval_width, atol=.02)

#     lower = x[lower_index]
#     upper = x[upper_index]
#     return lower, upper
