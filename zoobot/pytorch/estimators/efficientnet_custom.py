import torch
from torch import nn, Tensor


def custom_top_dirichlet(input_dim, output_dim):
    """
    Final dense layer used in GZ DECaLS (after global pooling). 
    ``output_dim`` neurons with an activation of ``tf.nn.sigmoid(x) * 100. + 1.``, chosen to ensure 1-100 output range
    This range is suitable for parameters of Dirichlet distribution.

    Unlike tf version, NOT inplace

    Args:
        output_dim (int): Dimension of dense layer e.g. 34 for decision tree with 34 answers
    """
    return nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=output_dim),  
        ScaledSigmoid()
    )

class ScaledSigmoid(nn.modules.Sigmoid):
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#ReLU

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input) * 100. + 1.  # could make args if I needed
