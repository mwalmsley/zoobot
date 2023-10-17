import torch
x = torch.Tensor(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
# let us run the following only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)
else:
    raise AssertionError('CUDA not available')

# TODO DemoRings from galaxy-datasets

from galaxy_datasets import galaxy_mnist

root = '/project/def-bovy/walml/data/roots/galaxy_mnist'

df, label_cols = galaxy_mnist(root, train=True, download=False)


# TODO import zoobot and use something
