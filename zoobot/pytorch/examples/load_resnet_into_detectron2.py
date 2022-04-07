
import os, cv2  # pip install opencv-python
import matplotlib.pyplot as plt

import torch
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def rename_ckpt_keys(ckpt_to_load):

    renamed_ckpt_loc = os.path.basename(ckpt_to_load).replace('.ckpt', 'renamed_for_detectron2.ckpt')

    x = torch.load(
        ckpt_to_load, 
        map_location=torch.device('cpu')
    )
    model_state_dict = x['state_dict']

    backbone_state_dict = {}
    """
    detectron2 uses slightly different names for the resnet once loaded in faster-rcnn form.
    In faster-rcnn, res1-4 are named "backbone" not "model".
    We therefore need to rename the state_dict keys as "backbone" not "model".
    For example: model.0.0.res2.0.conv1.norm. -> backbone.res2.0.conv1.norm
    
    res5 is replaced by the region-of-interest proposal layers. You could load res5 onto those layers, perhaps, but best to freeze the model and retrain res5 from scratch I think.
    
    The "num_batches" tracked" attribute of the batch_norm layers is not included in faster-rcnn's frozenBN layers and does not affect outputs, so delete that

    model.2.0 is the linear head used to predict dirichlet concentrations. This is not used in faster-rcnn, which only wants the features (from res4), so delete that
    """

    for k, v in model_state_dict.items():
        if ('num_batches_tracked' not in k) and ('model.2.0' not in k):
            backbone_state_dict[k.replace('model.0.0', 'backbone').replace('backbone.res5', 'not_needed.res5')] = v

    torch.save(backbone_state_dict, renamed_ckpt_loc)

    return renamed_ckpt_loc


def load_renamed_ckpt_to_detectron2(renamed_ckpt_loc):

    setup_logger()

    im = cv2.imread("data/example_images/basic/J000411.19+020924.0.png") 

    cfg = get_cfg()
    
    # model type and training configuration (training config not used here)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    assert os.path.isfile(renamed_ckpt_loc)
    cfg.MODEL.WEIGHTS = renamed_ckpt_loc

    cfg.MODEL.DEVICE='cpu'  # model was trained on gpu, but you might not have one - simplest to load on cpu

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)  # TODO change metadata catalog
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
    # will probably be blank (or silly) because we didn't load the ROI layers - but that's okay, just checking the math works
    # the next step would be training the detectron2 ROI layers on a dataset with segmentation labels, while keeping the backbone (res4 and below) frozen


if __name__ == '__main__':

     # TODO replace with your trained checkpoint. Must be the detectron2 version of resnet50 (resnet_detectron2_custom), not the torchvision version.
    ckpt_to_load = '/home/walml/repos/resnet_det_1gpu_b256/checkpoints/epoch=42-step=44032.ckpt'

    renamed_ckpt_loc = rename_ckpt_keys(ckpt_to_load)

    load_renamed_ckpt_to_detectron2(renamed_ckpt_loc)
