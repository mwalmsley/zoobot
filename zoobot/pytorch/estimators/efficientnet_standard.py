# https://pytorch.org/vision/main/_modules/torchvision/models/efficientnet.html
# lightly modified from standard pytorch implementation

import logging
import copy
import math
from functools import partial
from typing import Callable, Optional, List, Sequence

import torch
from torch import nn, Tensor

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops import Conv2dNormActivation

from torchvision.models.efficientnet import MBConvConfig, MBConv

class EfficientNet(nn.Module):  # could make lightning, but I think it's clearer to do that one level up
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        include_top: bool,
        input_channels: int = 3,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The dropout probability. Only used for head, and I usually only have headless effnet, so rarely used
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        # _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                # added input_channels as an
                input_channels, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        # pytorch version includes the pooling outside of include_top
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # (1) i.e. equivalent to Global

        if include_top:  # should never be true for Zoobot
            logging.warning('Building EffNet with default top - almost certainly a bad idea for classifying GZ galaxies!')
            self.head = nn.Sequential(
                # TODO replace with PermaDropout via self.training
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(lastconv_output_channels, num_classes),
            )
        else:
            self.head = nn.Identity()  # does nothing

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet(
    arch: str,
    width_mult: float,
    depth_mult: float,
    dropout: float,
    use_imagenet_weights: bool,
    include_top: bool,
    input_channels: int,
    stochastic_depth_prob: float,
    progress: bool
) -> EfficientNet:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    # ratio, kernel, stride, input channels, output channels, layers, width/depth kwargs
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    model = EfficientNet(
        inverted_residual_setting, dropout, include_top, input_channels, stochastic_depth_prob)
        # norm_layer=nn.Identity)
    if use_imagenet_weights:
        assert include_top  # otherwise not sure if weights will load as I've changed code
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def efficientnet_b0(
    input_channels,
    stochastic_depth_prob: float = 0.2,
    use_imagenet_weights: bool = False,
    include_top: bool = True,
    progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        use_imagenet_weights (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # added include_top and input_channels, renamed pretrained to use_imagenet_weights
    return _efficientnet(
        arch="efficientnet_b0",
        width_mult=1.0,
        depth_mult=1.0,
        dropout=0.2,
        stochastic_depth_prob=stochastic_depth_prob,
        use_imagenet_weights=use_imagenet_weights,
        include_top=include_top,
        input_channels=input_channels,
        progress=progress)


def efficientnet_b2(
    input_channels,
    stochastic_depth_prob: float = 0.2,
    use_imagenet_weights: bool = False,
    include_top: bool = True,
    progress: bool = True) -> EfficientNet:
    """
    See efficientnet_b0, identical other than multipliers and dropout
    """
    # added include_top and input_channels, renamed pretrained to use_imagenet_weights
    assert not use_imagenet_weights
    return _efficientnet(
        arch="efficientnet_b2",
        width_mult=1.1,
        depth_mult=1.2,
        dropout=0.3,
        stochastic_depth_prob=stochastic_depth_prob,  # I added as an arg, for extra hparam to search (optionally - default=0.2)
        use_imagenet_weights=use_imagenet_weights,  # will likely fail
        include_top=include_top,
        input_channels=input_channels,
        progress=progress)


def efficientnet_b4(
    input_channels,
    stochastic_depth_prob: float = 0.2,
    use_imagenet_weights: bool = False,
    include_top: bool = True,
    progress: bool = True) -> EfficientNet:
    """
    See efficientnet_b0, identical other than multipliers and dropout
    """
    # added include_top and input_channels, renamed pretrained to use_imagenet_weights
    assert not use_imagenet_weights
    return _efficientnet(
        arch="efficientnet_b4",
        width_mult=1.4,
        depth_mult=1.8,
        dropout=0.4,
        stochastic_depth_prob=stochastic_depth_prob,
        use_imagenet_weights=use_imagenet_weights,  # will likely fail
        include_top=include_top,
        input_channels=input_channels,
        progress=progress)


# TODO efficientnet_v2_*, perhaps?

model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}


if __name__ == '__main__':

    from torchsummary import summary

    summary(efficientnet_b0(input_channels=1, include_top=False), (1, 224, 224), device='cpu')

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 112, 112]             288
       BatchNorm2d-2         [-1, 32, 112, 112]              64
              SiLU-3         [-1, 32, 112, 112]               0
            Conv2d-4         [-1, 32, 112, 112]             288
       BatchNorm2d-5         [-1, 32, 112, 112]              64
              SiLU-6         [-1, 32, 112, 112]               0
 AdaptiveAvgPool2d-7             [-1, 32, 1, 1]               0
            Conv2d-8              [-1, 8, 1, 1]             264
              SiLU-9              [-1, 8, 1, 1]               0
           Conv2d-10             [-1, 32, 1, 1]             288
          Sigmoid-11             [-1, 32, 1, 1]               0
SqueezeExcitation-12         [-1, 32, 112, 112]               0
           Conv2d-13         [-1, 16, 112, 112]             512
      BatchNorm2d-14         [-1, 16, 112, 112]              32
           MBConv-15         [-1, 16, 112, 112]               0
           Conv2d-16         [-1, 96, 112, 112]           1,536
      BatchNorm2d-17         [-1, 96, 112, 112]             192
             SiLU-18         [-1, 96, 112, 112]               0
           Conv2d-19           [-1, 96, 56, 56]             864
      BatchNorm2d-20           [-1, 96, 56, 56]             192
             SiLU-21           [-1, 96, 56, 56]               0
AdaptiveAvgPool2d-22             [-1, 96, 1, 1]               0
           Conv2d-23              [-1, 4, 1, 1]             388
             SiLU-24              [-1, 4, 1, 1]               0
           Conv2d-25             [-1, 96, 1, 1]             480
          Sigmoid-26             [-1, 96, 1, 1]               0
SqueezeExcitation-27           [-1, 96, 56, 56]               0
           Conv2d-28           [-1, 24, 56, 56]           2,304
      BatchNorm2d-29           [-1, 24, 56, 56]              48
           MBConv-30           [-1, 24, 56, 56]               0
           Conv2d-31          [-1, 144, 56, 56]           3,456
      BatchNorm2d-32          [-1, 144, 56, 56]             288
             SiLU-33          [-1, 144, 56, 56]               0
           Conv2d-34          [-1, 144, 56, 56]           1,296
      BatchNorm2d-35          [-1, 144, 56, 56]             288
             SiLU-36          [-1, 144, 56, 56]               0
AdaptiveAvgPool2d-37            [-1, 144, 1, 1]               0
           Conv2d-38              [-1, 6, 1, 1]             870
             SiLU-39              [-1, 6, 1, 1]               0
           Conv2d-40            [-1, 144, 1, 1]           1,008
          Sigmoid-41            [-1, 144, 1, 1]               0
SqueezeExcitation-42          [-1, 144, 56, 56]               0
           Conv2d-43           [-1, 24, 56, 56]           3,456
      BatchNorm2d-44           [-1, 24, 56, 56]              48
  StochasticDepth-45           [-1, 24, 56, 56]               0
           MBConv-46           [-1, 24, 56, 56]               0
           Conv2d-47          [-1, 144, 56, 56]           3,456
      BatchNorm2d-48          [-1, 144, 56, 56]             288
             SiLU-49          [-1, 144, 56, 56]               0
           Conv2d-50          [-1, 144, 28, 28]           3,600
      BatchNorm2d-51          [-1, 144, 28, 28]             288
             SiLU-52          [-1, 144, 28, 28]               0
AdaptiveAvgPool2d-53            [-1, 144, 1, 1]               0
           Conv2d-54              [-1, 6, 1, 1]             870
             SiLU-55              [-1, 6, 1, 1]               0
           Conv2d-56            [-1, 144, 1, 1]           1,008
          Sigmoid-57            [-1, 144, 1, 1]               0
SqueezeExcitation-58          [-1, 144, 28, 28]               0
           Conv2d-59           [-1, 40, 28, 28]           5,760
      BatchNorm2d-60           [-1, 40, 28, 28]              80
           MBConv-61           [-1, 40, 28, 28]               0
           Conv2d-62          [-1, 240, 28, 28]           9,600
      BatchNorm2d-63          [-1, 240, 28, 28]             480
             SiLU-64          [-1, 240, 28, 28]               0
           Conv2d-65          [-1, 240, 28, 28]           6,000
      BatchNorm2d-66          [-1, 240, 28, 28]             480
             SiLU-67          [-1, 240, 28, 28]               0
AdaptiveAvgPool2d-68            [-1, 240, 1, 1]               0
           Conv2d-69             [-1, 10, 1, 1]           2,410
             SiLU-70             [-1, 10, 1, 1]               0
           Conv2d-71            [-1, 240, 1, 1]           2,640
          Sigmoid-72            [-1, 240, 1, 1]               0
SqueezeExcitation-73          [-1, 240, 28, 28]               0
           Conv2d-74           [-1, 40, 28, 28]           9,600
      BatchNorm2d-75           [-1, 40, 28, 28]              80
  StochasticDepth-76           [-1, 40, 28, 28]               0
           MBConv-77           [-1, 40, 28, 28]               0
           Conv2d-78          [-1, 240, 28, 28]           9,600
      BatchNorm2d-79          [-1, 240, 28, 28]             480
             SiLU-80          [-1, 240, 28, 28]               0
           Conv2d-81          [-1, 240, 14, 14]           2,160
      BatchNorm2d-82          [-1, 240, 14, 14]             480
             SiLU-83          [-1, 240, 14, 14]               0
AdaptiveAvgPool2d-84            [-1, 240, 1, 1]               0
           Conv2d-85             [-1, 10, 1, 1]           2,410
             SiLU-86             [-1, 10, 1, 1]               0
           Conv2d-87            [-1, 240, 1, 1]           2,640
          Sigmoid-88            [-1, 240, 1, 1]               0
SqueezeExcitation-89          [-1, 240, 14, 14]               0
           Conv2d-90           [-1, 80, 14, 14]          19,200
      BatchNorm2d-91           [-1, 80, 14, 14]             160
           MBConv-92           [-1, 80, 14, 14]               0
           Conv2d-93          [-1, 480, 14, 14]          38,400
      BatchNorm2d-94          [-1, 480, 14, 14]             960
             SiLU-95          [-1, 480, 14, 14]               0
           Conv2d-96          [-1, 480, 14, 14]           4,320
      BatchNorm2d-97          [-1, 480, 14, 14]             960
             SiLU-98          [-1, 480, 14, 14]               0
AdaptiveAvgPool2d-99            [-1, 480, 1, 1]               0
          Conv2d-100             [-1, 20, 1, 1]           9,620
            SiLU-101             [-1, 20, 1, 1]               0
          Conv2d-102            [-1, 480, 1, 1]          10,080
         Sigmoid-103            [-1, 480, 1, 1]               0
SqueezeExcitation-104          [-1, 480, 14, 14]               0
          Conv2d-105           [-1, 80, 14, 14]          38,400
     BatchNorm2d-106           [-1, 80, 14, 14]             160
 StochasticDepth-107           [-1, 80, 14, 14]               0
          MBConv-108           [-1, 80, 14, 14]               0
          Conv2d-109          [-1, 480, 14, 14]          38,400
     BatchNorm2d-110          [-1, 480, 14, 14]             960
            SiLU-111          [-1, 480, 14, 14]               0
          Conv2d-112          [-1, 480, 14, 14]           4,320
     BatchNorm2d-113          [-1, 480, 14, 14]             960
            SiLU-114          [-1, 480, 14, 14]               0
AdaptiveAvgPool2d-115            [-1, 480, 1, 1]               0
          Conv2d-116             [-1, 20, 1, 1]           9,620
            SiLU-117             [-1, 20, 1, 1]               0
          Conv2d-118            [-1, 480, 1, 1]          10,080
         Sigmoid-119            [-1, 480, 1, 1]               0
SqueezeExcitation-120          [-1, 480, 14, 14]               0
          Conv2d-121           [-1, 80, 14, 14]          38,400
     BatchNorm2d-122           [-1, 80, 14, 14]             160
 StochasticDepth-123           [-1, 80, 14, 14]               0
          MBConv-124           [-1, 80, 14, 14]               0
          Conv2d-125          [-1, 480, 14, 14]          38,400
     BatchNorm2d-126          [-1, 480, 14, 14]             960
            SiLU-127          [-1, 480, 14, 14]               0
          Conv2d-128          [-1, 480, 14, 14]          12,000
     BatchNorm2d-129          [-1, 480, 14, 14]             960
            SiLU-130          [-1, 480, 14, 14]               0
AdaptiveAvgPool2d-131            [-1, 480, 1, 1]               0
          Conv2d-132             [-1, 20, 1, 1]           9,620
            SiLU-133             [-1, 20, 1, 1]               0
          Conv2d-134            [-1, 480, 1, 1]          10,080
         Sigmoid-135            [-1, 480, 1, 1]               0
SqueezeExcitation-136          [-1, 480, 14, 14]               0
          Conv2d-137          [-1, 112, 14, 14]          53,760
     BatchNorm2d-138          [-1, 112, 14, 14]             224
          MBConv-139          [-1, 112, 14, 14]               0
          Conv2d-140          [-1, 672, 14, 14]          75,264
     BatchNorm2d-141          [-1, 672, 14, 14]           1,344
            SiLU-142          [-1, 672, 14, 14]               0
          Conv2d-143          [-1, 672, 14, 14]          16,800
     BatchNorm2d-144          [-1, 672, 14, 14]           1,344
            SiLU-145          [-1, 672, 14, 14]               0
AdaptiveAvgPool2d-146            [-1, 672, 1, 1]               0
          Conv2d-147             [-1, 28, 1, 1]          18,844
            SiLU-148             [-1, 28, 1, 1]               0
          Conv2d-149            [-1, 672, 1, 1]          19,488
         Sigmoid-150            [-1, 672, 1, 1]               0
SqueezeExcitation-151          [-1, 672, 14, 14]               0
          Conv2d-152          [-1, 112, 14, 14]          75,264
     BatchNorm2d-153          [-1, 112, 14, 14]             224
 StochasticDepth-154          [-1, 112, 14, 14]               0
          MBConv-155          [-1, 112, 14, 14]               0
          Conv2d-156          [-1, 672, 14, 14]          75,264
     BatchNorm2d-157          [-1, 672, 14, 14]           1,344
            SiLU-158          [-1, 672, 14, 14]               0
          Conv2d-159          [-1, 672, 14, 14]          16,800
     BatchNorm2d-160          [-1, 672, 14, 14]           1,344
            SiLU-161          [-1, 672, 14, 14]               0
AdaptiveAvgPool2d-162            [-1, 672, 1, 1]               0
          Conv2d-163             [-1, 28, 1, 1]          18,844
            SiLU-164             [-1, 28, 1, 1]               0
          Conv2d-165            [-1, 672, 1, 1]          19,488
         Sigmoid-166            [-1, 672, 1, 1]               0
SqueezeExcitation-167          [-1, 672, 14, 14]               0
          Conv2d-168          [-1, 112, 14, 14]          75,264
     BatchNorm2d-169          [-1, 112, 14, 14]             224
 StochasticDepth-170          [-1, 112, 14, 14]               0
          MBConv-171          [-1, 112, 14, 14]               0
          Conv2d-172          [-1, 672, 14, 14]          75,264
     BatchNorm2d-173          [-1, 672, 14, 14]           1,344
            SiLU-174          [-1, 672, 14, 14]               0
          Conv2d-175            [-1, 672, 7, 7]          16,800
     BatchNorm2d-176            [-1, 672, 7, 7]           1,344
            SiLU-177            [-1, 672, 7, 7]               0
AdaptiveAvgPool2d-178            [-1, 672, 1, 1]               0
          Conv2d-179             [-1, 28, 1, 1]          18,844
            SiLU-180             [-1, 28, 1, 1]               0
          Conv2d-181            [-1, 672, 1, 1]          19,488
         Sigmoid-182            [-1, 672, 1, 1]               0
SqueezeExcitation-183            [-1, 672, 7, 7]               0
          Conv2d-184            [-1, 192, 7, 7]         129,024
     BatchNorm2d-185            [-1, 192, 7, 7]             384
          MBConv-186            [-1, 192, 7, 7]               0
          Conv2d-187           [-1, 1152, 7, 7]         221,184
     BatchNorm2d-188           [-1, 1152, 7, 7]           2,304
            SiLU-189           [-1, 1152, 7, 7]               0
          Conv2d-190           [-1, 1152, 7, 7]          28,800
     BatchNorm2d-191           [-1, 1152, 7, 7]           2,304
            SiLU-192           [-1, 1152, 7, 7]               0
AdaptiveAvgPool2d-193           [-1, 1152, 1, 1]               0
          Conv2d-194             [-1, 48, 1, 1]          55,344
            SiLU-195             [-1, 48, 1, 1]               0
          Conv2d-196           [-1, 1152, 1, 1]          56,448
         Sigmoid-197           [-1, 1152, 1, 1]               0
SqueezeExcitation-198           [-1, 1152, 7, 7]               0
          Conv2d-199            [-1, 192, 7, 7]         221,184
     BatchNorm2d-200            [-1, 192, 7, 7]             384
 StochasticDepth-201            [-1, 192, 7, 7]               0
          MBConv-202            [-1, 192, 7, 7]               0
          Conv2d-203           [-1, 1152, 7, 7]         221,184
     BatchNorm2d-204           [-1, 1152, 7, 7]           2,304
            SiLU-205           [-1, 1152, 7, 7]               0
          Conv2d-206           [-1, 1152, 7, 7]          28,800
     BatchNorm2d-207           [-1, 1152, 7, 7]           2,304
            SiLU-208           [-1, 1152, 7, 7]               0
AdaptiveAvgPool2d-209           [-1, 1152, 1, 1]               0
          Conv2d-210             [-1, 48, 1, 1]          55,344
            SiLU-211             [-1, 48, 1, 1]               0
          Conv2d-212           [-1, 1152, 1, 1]          56,448
         Sigmoid-213           [-1, 1152, 1, 1]               0
SqueezeExcitation-214           [-1, 1152, 7, 7]               0
          Conv2d-215            [-1, 192, 7, 7]         221,184
     BatchNorm2d-216            [-1, 192, 7, 7]             384
 StochasticDepth-217            [-1, 192, 7, 7]               0
          MBConv-218            [-1, 192, 7, 7]               0
          Conv2d-219           [-1, 1152, 7, 7]         221,184
     BatchNorm2d-220           [-1, 1152, 7, 7]           2,304
            SiLU-221           [-1, 1152, 7, 7]               0
          Conv2d-222           [-1, 1152, 7, 7]          28,800
     BatchNorm2d-223           [-1, 1152, 7, 7]           2,304
            SiLU-224           [-1, 1152, 7, 7]               0
AdaptiveAvgPool2d-225           [-1, 1152, 1, 1]               0
          Conv2d-226             [-1, 48, 1, 1]          55,344
            SiLU-227             [-1, 48, 1, 1]               0
          Conv2d-228           [-1, 1152, 1, 1]          56,448
         Sigmoid-229           [-1, 1152, 1, 1]               0
SqueezeExcitation-230           [-1, 1152, 7, 7]               0
          Conv2d-231            [-1, 192, 7, 7]         221,184
     BatchNorm2d-232            [-1, 192, 7, 7]             384
 StochasticDepth-233            [-1, 192, 7, 7]               0
          MBConv-234            [-1, 192, 7, 7]               0
          Conv2d-235           [-1, 1152, 7, 7]         221,184
     BatchNorm2d-236           [-1, 1152, 7, 7]           2,304
            SiLU-237           [-1, 1152, 7, 7]               0
          Conv2d-238           [-1, 1152, 7, 7]          10,368
     BatchNorm2d-239           [-1, 1152, 7, 7]           2,304
            SiLU-240           [-1, 1152, 7, 7]               0
AdaptiveAvgPool2d-241           [-1, 1152, 1, 1]               0
          Conv2d-242             [-1, 48, 1, 1]          55,344
            SiLU-243             [-1, 48, 1, 1]               0
          Conv2d-244           [-1, 1152, 1, 1]          56,448
         Sigmoid-245           [-1, 1152, 1, 1]               0
SqueezeExcitation-246           [-1, 1152, 7, 7]               0
          Conv2d-247            [-1, 320, 7, 7]         368,640
     BatchNorm2d-248            [-1, 320, 7, 7]             640
          MBConv-249            [-1, 320, 7, 7]               0
          Conv2d-250           [-1, 1280, 7, 7]         409,600
     BatchNorm2d-251           [-1, 1280, 7, 7]           2,560
            SiLU-252           [-1, 1280, 7, 7]               0
AdaptiveAvgPool2d-253           [-1, 1280, 1, 1]               0
        Identity-254                 [-1, 1280]               0
================================================================
Total params: 4,006,972
Trainable params: 4,006,972
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 173.64
Params size (MB): 15.29
Estimated Total Size (MB): 189.12

# Without batchnorm, params match TF exactly
Total params: 3,964,956
Trainable params: 3,964,956
Non-trainable params: 0
"""

# pytorch does vastly worse without batchnorm (doesn't train properly
# tensorflow still does exactly the same