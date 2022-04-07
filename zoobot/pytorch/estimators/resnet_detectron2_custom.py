import torch
from detectron2.config.defaults import _C as default_config
from detectron2.layers import shape_spec

from zoobot.pytorch.estimators.resnet_detectron2_standard import build_resnet_backbone


def get_resnet(
        input_channels,
        use_imagenet_weights=False,
        include_top=False
    ):

    input_shape = shape_spec.ShapeSpec(height=None, width=None, channels=input_channels, stride=None)

    if use_imagenet_weights:
        raise NotImplementedError  # not implemented yet, could load the detectron2 pretrained imagenet weights if we wanted

    if include_top:
        raise NotImplementedError  # detectron2's version has no top

    base_resnet = build_resnet_backbone(default_config, input_shape)  # exactly matching detectron2's version
    # output is from final stage before head aka res5, 1024

    base_resnet_with_pooling = torch.nn.Sequential(
        base_resnet,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(1)  # adaptive pool leaves NC11, want NC before linear layer so flatten after dim=1
    )

    return base_resnet_with_pooling


if __name__ == '__main__':

    # debugging only

    import numpy as np

    channels = 3
    input_shape = shape_spec.ShapeSpec(height=None, width=None, channels=channels, stride=None)

    # model = build_resnet_backbone(default_config, input_shape)  # exactly matching detectron2's version
    # print(model)

    # model = get_resnet(input_channels=1)

    from zoobot.pytorch.training import losses
    from zoobot.shared import label_metadata, schemas
    from zoobot.pytorch.estimators import define_model

    question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)

    loss_func = losses.calculate_multiquestion_loss

    model = define_model.ZoobotModel(schema=schema, loss=loss_func, channels=channels, get_architecture=get_resnet)
    # print(model)

    x = torch.from_numpy(np.random.rand(16, channels, 224, 224)).float()
    print(model(x).shape)
