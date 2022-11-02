"""
we know from tests/test_loss_equivalence that the losses are mathematically identical prior to aggregation 
(i.e. rows of galaxies, columns of loss-per-question). 
So why do the aggregated losses give different values?
Call the math value x (including the log)

Both benchmarks use batch_size=128
Both use 2 GPUs

Tensorflow loss uses x / batch size because can only SUM within a replica, not MEAN
Presumably should give loss = sum(x/batch_size), where sum is over all questions and all galaxies

PyTorch loss just has x, and it's not clear how the loss is aggregated between replicas

Do the losses agree on 1 GPU mode? This will tell if the difference is from per-replica aggregation, or from e.g. averaging over all values not all rows

The difference is approx. 6 - TF is 6x PT, currently.
Perhaps difference of 10 (questions) and factor of 2 (replica aggregation)?

"""

