import pytest

import timm
import torch


def test_get_encoder():
    model = timm.create_model("hf_hub:mwalmsley/zoobot-encoder-efficientnet_b0", pretrained=True)
    assert model(torch.rand(1, 3, 224, 224)).shape == (1, 1280)


def test_get_finetuned():
    # checkpoint_loc = 'https://huggingface.co/mwalmsley/zoobot-finetuned-is_tidal/resolve/main/3.ckpt' pickle problem via lightning
    # checkpoint_loc = '/home/walml/Downloads/3.ckpt'  # works when downloaded manually

    from huggingface_hub import hf_hub_download

    REPO_ID = "mwalmsley/zoobot-finetuned-is_tidal"
    FILENAME = "FinetuneableZoobotClassifier.ckpt"

    downloaded_loc = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
    )
    from zoobot.pytorch.training import finetune
    model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(downloaded_loc, map_location='cpu') #  hub_name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
    assert model(torch.rand(1, 3, 224, 224)).shape == (1, 2)

def test_get_finetuned_class_method():

    from zoobot.pytorch.training import finetune

    model = finetune.FinetuneableZoobotClassifier.load_from_name('mwalmsley/zoobot-finetuned-is_tidal', map_location='cpu')
    assert model(torch.rand(1, 3, 224, 224)).shape == (1, 2)

# def test_get_finetuned_from_local():
#     # checkpoint_loc = '/home/walml/repos/zoobot/tests/convnext_nano_finetuned_linear_is-lsb.ckpt'
#     checkpoint_loc = '/home/walml/repos/zoobot-foundation/results/finetune/is-lsb/debug/checkpoints/4.ckpt'
    
#     from zoobot.pytorch.training import finetune
#     # if originally trained with a direct in-memory checkpoint, must specify the hub name manually. otherwise it's saved as an hparam.
#     model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(checkpoint_loc, map_location='cpu') # hub_name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', )
#     assert model(torch.rand(1, 3, 224, 224)).shape == (1, 2)