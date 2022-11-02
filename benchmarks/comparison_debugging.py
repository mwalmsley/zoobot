
"""
we know from tests/test_loss_equivalence that the losses are mathematically identical prior to aggregation 
(i.e. rows of galaxies, columns of loss-per-question). 
So why do the aggregated losses give different values?
Call the math value x (including the log)

Both benchmarks use batch_size=128
Both use 2 GPUs

Tensorflow loss uses x / batch size because can only SUM within a replica, not MEAN
Presumably should give loss = np.sum(x/batch_size), where sum is over all questions and all galaxies
(uses tf.keras.losses.Reduction.SUM, "scalar sum of weighted losses", https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction)
Adding up across rows and columns and dividing by num. rows is equivalent to adding up across columns and taking an average across (new) rows
"""
import numpy as np

n_galaxies = 20
n_questions = 8
multi_q_loss = np.random.rand(n_galaxies, n_questions)

# LHS is actual TF operations, RHS is intuition (mean loss per galaxy, where per-galaxy loss is summed over questions)
assert np.isclose(np.sum(multi_q_loss/n_galaxies), np.mean(multi_q_loss.sum(axis=1), axis=0))

"""
PyTorch loss just has x, and it's not clear how the loss is aggregated between replicas

Do the losses agree on 1 GPU mode? This will tell if the difference is from per-replica aggregation, or from e.g. averaging over all values not all rows

The difference is approx. 6 - TF is 6x PT, currently.
Perhaps difference of 10 (questions) and factor of 2 (replica aggregation)?

- PT has no 2x factor change in loss with gpu number (good, this is intuitive)
- TF seems to increase by a factor of 2 (11 to 20) when going from 2 to 1 GPUs. Aka, loss is divided by num. GPUs. Likely because batch is divided across gpus? Add explicit GPU factor and restart
- Actual TF/PT factor is therefore 10-ish, not 5-ish. Hmm, same as num questions...(16 / 1.4 =~ 11)

So probably:
- TF is sum of multi-q loss / batch size, divided by num GPUs
- PT was mean of multi-q loss (aka average across questions, not just sum), independent of num GPUs.

Solution:
- TF to be multiplied by num gpus in train_with_keras loss func, counteracting current num gpus divisor (x2 factor, usually)
- PT to sum over questions within loss func, before automatically taking mean across galaxies (x10 factor)
Overall, PT will increase by x5 factor vs TF, should become comparable

"""
