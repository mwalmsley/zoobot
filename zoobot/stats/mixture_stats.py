# import numpy as np
# from scipy.special import gamma, loggamma, digamma, betaln  # log more numerically stable than gamma
# from scipy.stats import beta  # or could use tfp
# import tensorflow as tf
# import tensorflow_probability as tfp


# def entropy_upper_bound(arr_of_alphas, weights):
#     return entropy_given_components(arr_of_alphas, weights) - distance_term(arr_of_alphas, weights, distance_func=dirichlet_kl_div)


# def entropy_lower_bound(arr_of_alphas, weights):
#     return entropy_given_components(arr_of_alphas, weights) - distance_term(arr_of_alphas, weights, distance_func=lambda x, y: dirichlet_chernoff(x, y, lam=.5))


# def distance_term(answer_model, weights, distance_func):
#     n_dim = len(weights)
#     i_terms = np.zeros(len(weights))
#     for i in range(n_dim):
#         j_terms = np.zeros(len(weights))
#         for j in range(n_dim):
#             # the i=j term is not 0, don't skip it
#             distance = distance_func(answer_model[:, i], answer_model[:, j])
#             j_terms[j] = weights[j] * np.exp(-distance)
#         i_terms[i] = weights[i] * np.log(np.sum(j_terms))
#     return np.sum(i_terms)


# # def mbeta(x):
# #     result = np.prod(gamma(x))/gamma(x.sum())
# #     if np.isnan(result):
# #         raise ValueError(x)
# #     else:
# #         return result

# # def dirichlet_chernoff(a, b, lam):
# #     exp_neg_chern = mbeta(lam*a + (1-lam)*b) / (mbeta(a) ** lam * mbeta(b) ** (1-lam))
# #     chern = -np.log(exp_neg_chern)
# #     return chern

# def mbeta(x):
#     result = np.prod(gamma(x))/gamma(x.sum())
#     if np.isnan(result):
#         raise ValueError(x)
#     else:
#         return result

# def log_mbeta(x):
#     return np.sum(loggamma(x)) - loggamma(np.sum(x))  # nice symmetry

# # def dirichlet_chernoff(a, b, lam):
# #     exp_neg_chern = mbeta(lam*a + (1-lam)*b) / (mbeta(a) ** lam * mbeta(b) ** (1-lam))
# #     chern = -np.log(exp_neg_chern)
# #     return chern

# def dirichlet_chernoff(a, b, lam):
#     neg_chern = log_mbeta(lam * a + (1-lam) * b) - lam*log_mbeta(a) - (1-lam)*log_mbeta(b)
#     return -neg_chern

# # copied from tfp, probably not needed
# # @tf.function
# def dirichlet_kl_div(concentration1, concentration2):
#     # concentration1 = tf.convert_to_tensor(concentration1)
#     # concentration2 = tf.convert_to_tensor(concentration2)
#     digamma_sum_d1 = tf.math.digamma(
#         tf.reduce_sum(concentration1, axis=-1, keepdims=True))
#     digamma_diff = tf.math.digamma(concentration1) - digamma_sum_d1
#     concentration_diff = concentration1 - concentration2

#     return (
#         tf.reduce_sum(concentration_diff * digamma_diff, axis=-1) -
#         tf.math.lbeta(concentration1) + tf.math.lbeta(concentration2))

# # pure python version
# # def dirichlet_kl_div(concentration1, concentration2):
# #     digamma_sum_d1 = digamma(concentration1.sum(axis=-1))
# #     digamma_diff = digamma(concentration1) - digamma_sum_d1
# #     concentration_diff = concentration1 - concentration2
# #     positive_comp = (concentration_diff * digamma_diff).sum(axis=-1)
# #     # negative_comp = betaln(*concentration1) + betaln(*concentration2)
# #     negative_comp = tf.math.lbeta(concentration1) + tf.math.lbeta(concentration2)
# #     return positive_comp - negative_comp.numpy()


# def entropy_given_components(answer_model, weights):
#     list_of_alphas = tf.transpose(answer_model)
#     return np.sum(weights * np.array([tfp.distributions.Dirichlet(alpha).entropy() for alpha in list_of_alphas]))
