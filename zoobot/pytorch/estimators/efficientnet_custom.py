import math  # not torch, just python math!

import torch
from torch import nn, Tensor


def custom_top_dirichlet(input_dim, output_dim):
    """
    Final dense layer used in GZ DECaLS (after global pooling). 
    ``output_dim`` neurons with an activation of ``tf.nn.sigmoid(x) * 100. + 1.``, chosen to ensure 1-100 output range
    This range is suitable for parameters of Dirichlet distribution.

    Args:
        output_dim (int): Dimension of dense layer e.g. 34 for decision tree with 34 answers

    Returns:
        nn.Sequential: nn.Linear followed by 1-101 sigmoid activation
    """
    return nn.Sequential(
        # LinearWithCustomInit(in_features=input_dim, out_features=output_dim),
        nn.Linear(in_features=input_dim, out_features=output_dim),
        ScaledSigmoid()
    )


class LinearWithCustomInit(nn.Linear):

    # fan_in = in_features e.g. 1280


    # identical __init__ etc

    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L103
    def reset_parameters(self) -> None:
        # # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
        # # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # nn.init.uniform_(self.weight, 2., 3.)
        # if self.bias is not None:
        #     # print(self.weight.size(1))
        #     # fan_in = weight.size(1) = input_dim (e.g. 1280)
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     # print(self.bias, bound)
        #     # set bias to have values uniformly drawn between -bound and bound (i.e. 1/sqrt(1280))
        #     nn.init.uniform_(self.bias, -bound, bound)

        # glorot init
        fan_in = self.weight.size(1)
        fan_out = self.weight.size(0)
        weight_bound = math.sqrt(6/(fan_in + fan_out))
        # but with extra factor 10 lower
        nn.init.uniform_(self.weight, -weight_bound/10, weight_bound/10)
        # nn.init.zeros_(self.bias)
        # nn.init.uniform_(self.weight, -0.0101, -0.01)
        # nn.init.zeros_(self.bias)
        nn.init.uniform_(self.bias, -4.5, -3.5)

        # super().reset_parameters()  # sets self.bias, self.weight

class ScaledSigmoid(nn.modules.Sigmoid):
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#ReLU

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): any vector. typically logits from a neural network

        Returns:
            Tensor: input mapped to range (1, 101) via torch.sigmoid
        """
        return torch.sigmoid(input) * 100. + 1.  # could make args if I needed


if __name__ == '__main__':
    # print(nn.Linear(1280, 40).weight)
    # print(LinearWithCustomInit(1280, 40).weight)


    x = torch.rand(1280)

    # print(nn.Linear(1280, 40)(x))
    # print(LinearWithCustomInit(1280, 40)(x))

    # print(torch.sigmoid(nn.Linear(1280, 40)(x)))
    # print(torch.sigmoid(LinearWithCustomInit(1280, 40)(x)))


    lin = nn.Linear(1280, 40)(x)
    lin_cust = LinearWithCustomInit(1280, 40)(x)

    print(ScaledSigmoid()(lin))
    print(ScaledSigmoid()(lin_cust))