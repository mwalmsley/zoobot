# import logging

import torch
from torch import nn, Tensor

# from zoobot.pytorch.estimators import efficientnet_standard


# def define_headless_efficientnet(input_channels=None, get_effnet=efficientnet_standard.efficientnet_b0, use_imagenet_weights=False):
#     """
#     Define efficientnet model to train.
#     Thin wrapper around ``get_effnet``, an efficientnet creation function from ``efficientnet_standard``, that ensures the appropriate args.

#     Additional keyword arguments are passed to ``get_effnet``.

#     Args:
#         input_channels (tuple, optional): Expected input shape e.g. (224, 224, 1). Defaults to None.
#         get_effnet (function, optional): Efficientnet creation function from ``efficientnet_standard``. Defaults to efficientnet_b0.
    
#     Returns:
#         [type]: [description]
#     """
#     logging.info('Building efficientnet to expect input {}, after any preprocessing layers'.format(input_channels))



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
