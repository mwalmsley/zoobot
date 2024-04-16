import logging
import os

import timm

from galaxy_datasets import demo_rings

from zoobot.pytorch.training import finetune, representations
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.pytorch.training import finetune
from zoobot.shared import load_predictions, schemas


def main(catalog, save_dir, name="hf_hub:mwalmsley/zoobot-encoder-convnext_nano"):

    assert all([os.path.isfile(x) for x in catalog['file_loc']])
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load the encoder

    # OPTION 1
    # Load a pretrained model from HuggingFace, with no finetuning, only as published
    model = representations.ZoobotEncoder.load_from_name(name)
    # or equivalently (the above is just a wrapper for these two lines below)
    # encoder = timm.create_model(model_name=name, pretrained=True)
    # model = representations.ZoobotEncoder(encoder=encoder)

    """
    # OPTION 2

    # Load a model that has been finetuned on your own data
    # (...do your usual finetuning..., or load a finetuned model with finetune.FinetuneableZoobotClassifier(checkpoint_loc=....ckpt)
    encoder = finetuned_model.encoder
    # and then convert to simple pytorch lightning model. You can use any pytorch model here.
    model = representations.ZoobotEncoder(encoder=encoder)
    """

    encoder_dim = define_model.get_encoder_dim(model.encoder)
    label_cols = [f'feat_{n}' for n in range(encoder_dim)]
    save_loc = os.path.join(save_dir, 'representations.hdf5')

    accelerator = 'cpu'  # or 'gpu' if available
    batch_size = 32
    resize_after_crop = 224

    datamodule_kwargs = {'batch_size': batch_size, 'resize_after_crop': resize_after_crop}
    trainer_kwargs = {'devices': 1, 'accelerator': accelerator}
    predict_on_catalog.predict(
        catalog,
        model,
        n_samples=1,
        label_cols=label_cols,
        save_loc=save_loc,
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs
    )

    return save_loc


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)

    # use this demo dataset
    # TODO change this to wherever you'd like, it will auto-download
    data_dir = '/Users/user/repos/galaxy-datasets/roots/demo_rings'
    catalog, _ = demo_rings(root=data_dir, download=True, train=True)
    print(catalog.head())
    # zoobot expects id_str and file_loc columns, so add these if needed

    # save the representations here
    # TODO change this to wherever you'd like
    save_dir = os.path.join('/Users/user/repos/zoobot/results/pytorch/representations/example')

    representations_loc = main(catalog, save_dir)
    rep_df = load_predictions.single_forward_pass_hdf5s_to_df(representations_loc)
    print(rep_df)
