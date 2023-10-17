import logging
from functools import partial
from typing import List

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import timm

from zoobot.pytorch.estimators import efficientnet_custom, custom_layers
from zoobot.pytorch.training import losses

# overall strategy
# timm for defining complicated pytorch modules
# manual modules for simple things (e.g. heads)
# together, this give pure pytorch self.encoder and self.head
# lightning modules with training instructions built out of those blocks (Zoobot, FinetuneableZooot)
# FinetuneableZoobot could subclass Zoobot to have different optimisers/init args - first has arch stuff, second has optim stuff

# to create from-scratch zoobot model:
# ZoobotTree(pytorch_encoder_args, head_args)   (needs hparam args to be restoreable)
# and within that init:
# pytorch_encoder = get_pytorch_encoder(timm model and model args)
# pytorch_dirichlet_head = get_pytorch_dirichlet_head...
# and then train from scratch

# to finetune:
# load ZoobotTree from checkpoint and keep only ZoobotEncoder
# encoder = ZoobotTree.load_from_checkpoint().encoder   (could wrap in load_pretrained_encoder)
# FinetuneableZoobotClassifier(pretrained_model.encoder, optim_args, task_args)
# (same approach for FinetuneableZoobotTree)

# to use just the encoder later: 
# encoder = load_pretrained_encoder(pyramid=False)
# when pyramid=True, reset the timm model to pull lightning features (TODO)

# timm gives regular pytorch models (with .forward_features argument available)
# for both training and finetuning, we also use some custom torch classes as heads
# DirichletHead(schema, dropout)
# LinearClassifier(output_dim, dropout)



class GenericLightningModule(pl.LightningModule):
    """
    All Zoobot models use the lightningmodule API and so share this structure
    super generic, just to outline the structure. nothing specific to dirichlet, gz, etc
    only assumes an encoder and a head
    """

    def __init__(
        self,
        *args,  # to be saved as hparams
        ):
        super().__init__()
        self.save_hyperparameters()  # saves all args by default
        self.setup_metrics()


    def setup_metrics(self):
        # these are ignored unless output dim = 2
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        # self.log_on_step = False
        # self.log_on_step is useful for debugging, but slower - best when log_every_n_steps is fairly large


    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)
    
    def make_step(self, batch, batch_idx, step_name):
        x, labels = batch
        predictions = self(x)  # by default, these are Dirichlet concentrations
        loss = self.calculate_and_log_loss(predictions, labels, step_name)      
        return {'loss': loss, 'predictions': predictions, 'labels': labels}

    def calculate_and_log_loss(self, predictions, labels, step_name):
        raise NotImplementedError('Must be subclassed')

    def configure_optimizers(self):
        raise NotImplementedError('Must be subclassed')

    def training_step(self, batch, batch_idx):
        return self.make_step(batch, batch_idx, step_name='train')

    def on_train_batch_end(self, outputs, *args):
        self.log_outputs(outputs, step_name='train')

    def validation_step(self, batch, batch_idx):
        return self.make_step(batch, batch_idx, step_name='validation')

    def on_validation_batch_end(self, outputs, *args):
        self.log_outputs(outputs, step_name='validation')

    def log_outputs(self, outputs, step_name):
        raise NotImplementedError('Must be subclassed')

    def test_step(self, batch, batch_idx):
        return self.make_step(batch, batch_idx, step_name='test')

    def on_test_batch_end(self, outputs, *args):
         self.log_outputs(outputs, step_name='test')

    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#inference
        # this calls forward, while avoiding the need for e.g. model.eval(), torch.no_grad()
        # x, y = batch  # would be usual format, but here, batch does not include labels
        return self(batch)


class ZoobotTree(GenericLightningModule):
    """
    
    The Zoobot model. Train from scratch using :func:`zoobot.pytorch.training.train_with_pytorch_lightning.train_default_zoobot_from_scratch`.

    PyTorch LightningModule describing how to train the encoder and head (described below).
    Trains using Dirichlet loss. Labels should be num. volunteers giving each answer to each question. 

    See the code for exact training step, logging, etc - there's a lot of detail.

    Args:
        output_dim (int): Output dimension of model's head e.g. 34 for predicting a 34-answer decision tree.
        question_index_groups (List): Mapping of which label indices are part of the same question. See :ref:`training_on_vote_counts`.
        architecture_name (str, optional): Architecture to use. Passed to timm. Must be in timm.list_models(). Defaults to "efficientnet_b0".
        channels (int, optional): Num. input channels. Probably 3 or 1. Defaults to 1.
        use_imagenet_weights (bool, optional): Load weights pretrained on ImageNet (NOT galaxies!). Defaults to False.
        test_time_dropout (bool, optional): Apply dropout at test time, to pretend to be Bayesian. Defaults to True.
        timm_kwargs (dict, optional): passed to timm.create_model e.g. drop_path_rate=0.2 for effnet. Defaults to {}.
        learning_rate (float, optional): AdamW learning rate. Defaults to 1e-3.
    """

    # lightning only supports checkpoint loading / hparams which are not fancy classes
    # therefore, can't nicely wrap these arguments. So it goes.
    # override GenericLightningModule above, only this init
    def __init__(
        self,
        output_dim: int,
        question_index_groups: List,
        # encoder args
        architecture_name="efficientnet_b0",
        channels=1,
        use_imagenet_weights=False,
        test_time_dropout=True,
        timm_kwargs={},  # passed to timm.create_model e.g. drop_path_rate=0.2 for effnet
        # head args
        dropout_rate=0.2,
        learning_rate=1e-3,  # PyTorch default
        # optim args
        betas=(0.9, 0.999),  # PyTorch default
        weight_decay=0.01,  # AdamW PyTorch default
        scheduler_params={}  # no scheduler by default
        ):

        # now, finally, can pass only standard variables as hparams to save
        # will still need to actually use these variables later, this super init only saves them
        super().__init__(
            output_dim,
            question_index_groups,
            architecture_name,
            channels,
            timm_kwargs,
            test_time_dropout,
            dropout_rate,
            learning_rate,
            betas,
            weight_decay,
            scheduler_params
        )

        logging.info('Generic __init__ complete - moving to Zoobot __init__')

        # set attributes for learning rate, betas, used by self.configure_optimizers()
        # TODO refactor to optimizer params
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params

        self.encoder = get_pytorch_encoder(
            architecture_name,
            channels,
            use_imagenet_weights=use_imagenet_weights,
            **timm_kwargs
        )
        # bit lazy assuming 224 input size
        self.encoder_dim = get_encoder_dim(self.encoder, input_size=224, channels=channels)
        # typically encoder_dim=1280 for effnetb0
        logging.info('encoder dim: {}'.format(self.encoder_dim))


        self.head = get_pytorch_dirichlet_head(
            self.encoder_dim,
            output_dim=output_dim,
            test_time_dropout=test_time_dropout,
            dropout_rate=dropout_rate
        )

        self.loss_func = get_dirichlet_loss_func(question_index_groups)

        logging.info('Zoobot __init__ complete')


    def calculate_and_log_loss(self, predictions, labels, step_name):
        # self.loss_func returns shape of (galaxy, question), mean to ()
        multiq_loss = self.loss_func(predictions, labels, sum_over_questions=False)
        # if hasattr(self, 'schema'):
        self.log_loss_per_question(multiq_loss, prefix=step_name)
        # sum over questions and take a per-device mean
        # for DDP strategy, batch size is constant (batches are not divided, data pool is divided)
        # so this will be the global per-example mean
        loss = torch.mean(torch.sum(multiq_loss, axis=1))
        return loss


    def configure_optimizers(self):
        # designed for training from scratch
        # parameters = list(self.head.parameters()) + list(self.encoder.parameters()) TODO should happen automatically?
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay
        )  
        if self.scheduler_params.get('name', None) == 'plateau':
            logging.info(f'Using Plateau scheduler with {self.scheduler_params}')
            # TODO could generalise this if needed
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                min_lr=1e-6,
                patience=self.scheduler_params.get('patience', 5)
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'validation/epoch_loss'}
        else:
            logging.info('No scheduler used')
            return optimizer  # no scheduler


    def log_outputs(self, outputs, step_name):
        self.log("{}/epoch_loss".format(step_name), outputs['loss'], on_epoch=True, on_step=False,prog_bar=True, logger=True, sync_dist=True)
        # if self.log_on_step:
        #     # seperate call to allow for different name, to allow for consistency with TF.keras auto-names
        #     self.log(
        #         "{}/step_loss".format(step_name), outputs['loss'], on_epoch=False, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        if outputs['predictions'].shape[1] == 2:  # will only do for binary classifications
            # logging.info(predictions.shape, labels.shape)
            self.log(
                "{}_accuracy".format(step_name), self.train_accuracy(outputs['predictions'], torch.argmax(outputs['labels'], dim=1, keepdim=False)), prog_bar=True, sync_dist=True)


    def log_loss_per_question(self, multiq_loss, prefix):
        # log questions individually
        # TODO need schema attribute or similar to have access to question names, this will do for now
        # unlike Finetuneable..., does not use TorchMetrics, simply logs directly
        # TODO could use TorchMetrics and for q in schema, self.q_metric loop
        for question_n in range(multiq_loss.shape[1]):
            self.log(f'{prefix}/epoch_questions/question_{question_n}_loss:0', torch.mean(multiq_loss[:, question_n]), on_epoch=True, on_step=False, sync_dist=True)


    

def get_dirichlet_loss_func(question_index_groups):
    # This just adds schema.question_index_groups as an arg to the usual (labels, preds) loss arg format
    # Would use lambda but multi-gpu doesn't support as lambda can't be pickled
    return partial(dirichlet_loss, question_index_groups=question_index_groups)


    # accept (labels, preds), return losses of shape (batch, question)
def dirichlet_loss(preds, labels, question_index_groups, sum_over_questions=False):
    # pytorch convention is preds, labels for loss func
    # my and sklearn convention is labels, preds for loss func

    # multiquestion_loss returns loss of shape (batch, question)
    # torch.sum(multiquestion_loss, axis=1) gives loss of shape (batch). Equiv. to non-log product of question likelihoods.
    multiq_loss = losses.calculate_multiquestion_loss(labels, preds, question_index_groups)
    if sum_over_questions:
        return torch.sum(multiq_loss, axis=1)
    else:
        return multiq_loss


def get_encoder_dim(encoder, input_size, channels):
    x = torch.randn(1, channels, input_size, input_size)  # batch size of 1
    return encoder(x).shape[-1]


def get_pytorch_encoder(
    architecture_name='efficientnet_b0',
    channels=1,
    use_imagenet_weights=False,
    **timm_kwargs
    ) -> nn.Module:
    """
    Create a trainable efficientnet model.
    First layers are galaxy-appropriate augmentation layers - see :meth:`zoobot.estimators.define_model.add_augmentation_layers`.
    Expects single channel image e.g. (300, 300, 1), likely with leading batch dimension.

    Optionally (by default) include the head (output layers) used for GZ DECaLS.
    Specifically, global average pooling followed by a dense layer suitable for predicting dirichlet parameters.
    See ``efficientnet_custom.custom_top_dirichlet``

    Args:
        output_dim (int): Dimension of head dense layer. No effect when include_top=False.
        input_size (int): Length of initial image e.g. 300 (asmeaned square)
        crop_size (int): Length to randomly crop image. See :meth:`zoobot.estimators.define_model.add_augmentation_layers`.
        resize_size (int): Length to resize image. See :meth:`zoobot.estimators.define_model.add_augmentation_layers`.
        weights_loc (str, optional): If str, load weights from efficientnet checkpoint at this location. Defaults to None.
        include_top (bool, optional): If True, include head used for GZ DECaLS: global pooling and dense layer. Defaults to True.
        expect_partial (bool, optional): If True, do not raise partial match error when loading weights (likely for optimizer state). Defaults to False.
        channels (int, default 1): Number of channels i.e. C in NHWC-dimension inputs. 

    Returns:
        torch.nn.Sequential: trainable efficientnet model including augmentations and optional head
    """
    # num_classes=0 gives pooled encoder
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/efficientnet.py
    
    # support older code that didn't specify effnet version
    if architecture_name == 'efficientnet':
        logging.warning('efficientnet variant not specified - please set architecture_name=efficientnet_b0 (or similar)')
        architecture_name = 'efficientnet_b0'
    return timm.create_model(architecture_name, in_chans=channels, num_classes=0, pretrained=use_imagenet_weights, **timm_kwargs)


def get_pytorch_dirichlet_head(encoder_dim: int, output_dim: int, test_time_dropout: bool, dropout_rate: float) -> torch.nn.Sequential:
    """
    Head to combine with encoder (above) when predicting Galaxy Zoo decision tree answers.
    Pytorch Sequential model.
    Predicts Dirichlet concentration parameters.
    
    Also used when finetuning on a new decision tree - see :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotTree`.

    Args:
        encoder_dim (int): dimensions of preceding encoder i.e. the input size expected by this submodel.
        output_dim (int): output dimensions of this head e.g. 34 to predict 34 answers.
        test_time_dropout (bool): Use dropout at test time. 
        dropout_rate (float): P of dropout. See torch.nn.Dropout docs.

    Returns:
        torch.nn.Sequential: pytorch model expecting `encoder_dim` vector and predicting `output_dim` decision tree answers.
    """

    modules_to_use = []

    assert output_dim is not None
    # no AdaptiveAvgPool2d, encoder assumed to pool already  
    if test_time_dropout:
        logging.info('Using test-time dropout')
        dropout_layer = custom_layers.PermaDropout
    else:
        logging.info('Not using test-time dropout')
        dropout_layer = torch.nn.Dropout
    modules_to_use.append(dropout_layer(dropout_rate))
    # TODO could optionally add a bottleneck layer here
    modules_to_use.append(efficientnet_custom.custom_top_dirichlet(encoder_dim, output_dim))

    return nn.Sequential(*modules_to_use)
