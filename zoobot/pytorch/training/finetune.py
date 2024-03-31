import logging
import os
from typing import Any, Union, Optional
import warnings
from functools import partial

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import torch
import torch.nn.functional as F
import torchmetrics as tm
import timm

from zoobot.pytorch.training import losses, schedulers
from zoobot.pytorch.estimators import define_model
from zoobot.shared import schemas

# https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/7
# I do this recursively and only for BatchNorm2d (not dropout, which I still want active)


def freeze_batchnorm_layers(model):
    for name, child in (model.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            logging.debug('Freezing {} {}'.format(child, name))
            child.eval()  # no grads, no param updates, no statistic updates
        else:
            freeze_batchnorm_layers(child)  # recurse


class FinetuneableZoobotAbstract(pl.LightningModule):
    """
    Parent class of :class:`FinetuneableZoobotClassifier`, :class:`FinetuneableZoobotRegressor`, :class:`FinetuneableZoobotTree`.
    You cannot use this class directly - you must use the child classes above instead.

    This class defines the shared finetuning args and methods used by those child classes.
    For example: 
    - When provided `name`, it will load the HuggingFace encoder with that name (see below for more).
    - When provided `learning_rate` it will set the optimizer to use that learning rate.

    Any FinetuneableZoobot model can be loaded in one of three ways:
        - HuggingFace name e.g. FinetuneableZoobotX(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...). Recommended.
        - Any PyTorch model in memory e.g. FinetuneableZoobotX(encoder=some_model, ...)
        - ZoobotTree checkpoint e.g. FinetuneableZoobotX(zoobot_checkpoint_loc='path/to/zoobot_tree.ckpt', ...)

    You could subclass this class to solve new finetuning tasks - see :ref:`advanced_finetuning`.

    Args:
        name (str, optional): Name of a model on HuggingFace Hub e.g.'hf_hub:mwalmsley/zoobot-encoder-convnext_nano'. Defaults to None.
        encoder (torch.nn.Module, optional): A PyTorch model already loaded in memory
        zoobot_checkpoint_loc (str, optional): Path to ZoobotTree lightning checkpoint to load. Loads with Load with :func:`zoobot.pytorch.training.finetune.load_pretrained_encoder`. Defaults to None.
        
        n_blocks (int, optional): 
        lr_decay (float, optional): For each layer i below the head, reduce the learning rate by lr_decay ^ i. Defaults to 0.75.
        weight_decay (float, optional): AdamW weight decay arg (i.e. L2 penalty). Defaults to 0.05.
        learning_rate (float, optional): AdamW learning rate arg. Defaults to 1e-4.
        dropout_prob (float, optional): P of dropout before final output layer. Defaults to 0.5.
        always_train_batchnorm (bool, optional): Temporarily deprecated. Previously, if True, do not update batchnorm stats during finetuning. Defaults to True.
        cosine_schedule (bool, optional): Reduce the learning rate each epoch according to a cosine schedule, after warmup_epochs. Defaults to False.
        warmup_epochs (int, optional): Linearly increase the learning rate from 0 to `learning_rate` over the first `warmup_epochs` epochs, before applying cosine schedule. No effect if cosine_schedule=False.
        max_cosine_epochs (int, optional): Epochs for the scheduled learning rate to decay to final learning rate (below). Warmup epochs don't count. No effect if `cosine_schedule=False`.
        max_learning_rate_reduction_factor (float, optional): Set final learning rate as `learning_rate` * `max_learning_rate_reduction_factor`. No effect if `cosine_schedule=False`.
        from_scratch (bool, optional): Ignore all settings above and train from scratch at `learning_rate` for all layers. Useful for a quick baseline. Defaults to False.
        prog_bar (bool, optional): Print progress bar during finetuning. Defaults to True.
        visualize_images (bool, optional): Upload example images to WandB. Good for debugging but slow. Defaults to False.
        seed (int, optional): random seed to use. Defaults to 42.
    """

    def __init__(
        self,

        # load a pretrained timm encoder saved on huggingface hub
        # (aimed at most users, easiest way to load published models)
        name=None,

        # ...or directly pass any model to use as encoder (if you do this, you will need to keep it around for later)
        # (aimed at tinkering with new architectures e.g. SSL)
        encoder=None,  # use any torch model already loaded in memory (must have .forward() method)

        # load a pretrained zoobottree model and grab the encoder (a timm model)
        # requires the exact same zoobot version used for training, not very portable
        # (aimed at supervised experiments)
        zoobot_checkpoint_loc=None,  

        # finetuning settings
        n_blocks=0,  # how many layers deep to FT
        lr_decay=0.75,
        weight_decay=0.05,
        learning_rate=1e-4,  # 10x lower than typical, you may like to experiment
        dropout_prob=0.5,
        always_train_batchnorm=False,  # temporarily deprecated
        # n_layers=0,  # for backward compat., n_blocks preferred. Now removed in v2.
        # these args are for the optional learning rate scheduler, best not to use unless you've tuned everything else already
        cosine_schedule=False,
        warmup_epochs=0,
        max_cosine_epochs=100,
        max_learning_rate_reduction_factor=0.01,
        # escape hatch for 'from scratch' baselines
        from_scratch=False,
        # debugging utils
        prog_bar=True,
        visualize_images=False,  # upload examples to wandb, good for debugging
        seed=42
    ):
        super().__init__()

        # adds every __init__ arg to model.hparams
        # will also add to wandb if using logging=wandb, I think
        # necessary if you want to reload!
        # with warnings.catch_warnings():
            # warnings.simplefilter("ignore")
            # this raises a warning that encoder is already a Module hence saved in checkpoint hence no need to save as hparam
            # true - except we need it to instantiate this class, so it's really handy to have saved as well
            # therefore ignore the warning
        self.save_hyperparameters(ignore=['encoder']) # never serialise the encoder, way too heavy
            # if you need the encoder to recreate, pass when loading checkpoint e.g. 
            # FinetuneableZoobotTree.load_from_checkpoint(loc, encoder=encoder)
        
        if name is not None:
            assert encoder is None, 'Cannot pass both name and encoder to use'
            self.encoder = timm.create_model(name, num_classes=0, pretrained=True)
            self.encoder_dim = self.encoder.num_features

        elif zoobot_checkpoint_loc is not None:
            assert encoder is None, 'Cannot pass both checkpoint to load and encoder to use'
            self.encoder = load_pretrained_zoobot(zoobot_checkpoint_loc)  # extracts the timm encoder
            self.encoder_dim = self.encoder.num_features
        else:
            assert zoobot_checkpoint_loc is None, 'Cannot pass both checkpoint to load and encoder to use'
            assert encoder is not None, 'Must pass either checkpoint to load or encoder to use'
            self.encoder = encoder
            # work out encoder dim 'manually'
            self.encoder_dim = define_model.get_encoder_dim(self.encoder)

        self.n_blocks = n_blocks

        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob

        self.cosine_schedule = cosine_schedule
        self.warmup_epochs = warmup_epochs
        self.max_cosine_epochs = max_cosine_epochs
        self.max_learning_rate_reduction_factor = max_learning_rate_reduction_factor

        self.from_scratch = from_scratch

        self.always_train_batchnorm = always_train_batchnorm
        if self.always_train_batchnorm:
            raise NotImplementedError('Temporarily deprecated, always_train_batchnorm=True not supported')
            # logging.info('always_train_batchnorm=True, so all batch norm layers will be finetuned')

        self.train_loss_metric = tm.MeanMetric()
        self.val_loss_metric = tm.MeanMetric()
        self.test_loss_metric = tm.MeanMetric()

        self.seed = seed
        self.prog_bar = prog_bar
        self.visualize_images = visualize_images

    def configure_optimizers(self):  
        """
        This controls which parameters get optimized

        self.head is always optimized, with no learning rate decay
        when self.n_blocks == 0, only self.head is optimized (i.e. frozen* encoder)
        
        for self.encoder, we enumerate the blocks (groups of layers) to potentially finetune
        and then pick the top self.n_blocks to finetune
        
        weight_decay is applied to both the head and (if relevant) the encoder
        learning rate decay is applied to the encoder only: lr * (lr_decay**block_n), ignoring the head (block 0)

        What counts as a "block" is a bit fuzzy, but I generally use the self.encoder.stages from timm. I also count the stem as a block.

        *batch norm layers may optionally still have updated statistics using always_train_batchnorm
        """

        lr = self.learning_rate
        params = [{"params": self.head.parameters(), "lr": lr}]

        logging.info(f'Encoder architecture to finetune: {type(self.encoder)}')

        if self.from_scratch:
            logging.warning('self.from_scratch is True, training everything and ignoring all settings')
            params += [{"params": self.encoder.parameters(), "lr": lr}]
            return torch.optim.AdamW(params, weight_decay=self.weight_decay)

        if isinstance(self.encoder, timm.models.EfficientNet): # includes v2
            # TODO for now, these count as separate layers, not ideal
            early_tuneable_layers = [self.encoder.conv_stem, self.encoder.bn1]
            encoder_blocks = list(self.encoder.blocks)
            tuneable_blocks = early_tuneable_layers + encoder_blocks
        elif isinstance(self.encoder, timm.models.ResNet):
            # all timm resnets seem to have this structure
            tuneable_blocks = [
                # similarly
                self.encoder.conv1,
                self.encoder.bn1,
                self.encoder.layer1,
                self.encoder.layer2,
                self.encoder.layer3,
                self.encoder.layer4
            ]
        elif isinstance(self.encoder, timm.models.MaxxVit):
            tuneable_blocks = [self.encoder.stem] + [stage for stage in self.encoder.stages]
        elif isinstance(self.encoder, timm.models.ConvNeXt):  # stem + 4 blocks, for all sizes
            # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py#L264
            tuneable_blocks = [self.encoder.stem] + [stage for stage in self.encoder.stages]
        else:
            raise ValueError(f'Encoder architecture not automatically recognised: {type(self.encoder)}')
            
        assert self.n_blocks <= len(
            tuneable_blocks
        ), f"Network only has {len(tuneable_blocks)} tuneable blocks, {self.n_blocks} specified for finetuning"

        
        # take n blocks, ordered highest layer to lowest layer
        tuneable_blocks.reverse()
        logging.info('possible blocks to tune: {}'.format(len(tuneable_blocks)))
        # will finetune all params in first N
        logging.info('blocks that will be tuned: {}'.format(self.n_blocks))
        blocks_to_tune = tuneable_blocks[:self.n_blocks]
        # optionally, can finetune batchnorm params in remaining layers
        remaining_blocks = tuneable_blocks[self.n_blocks:]
        logging.info('Remaining blocks: {}'.format(len(remaining_blocks)))
        assert not any([block in remaining_blocks for block in blocks_to_tune]), 'Some blocks are in both tuneable and remaining'

        # Append parameters of layers for finetuning along with decayed learning rate
        for i, block in enumerate(blocks_to_tune):  # _ is the block name e.g. '3'
            params.append({
                    "params": block.parameters(),
                    "lr": lr * (self.lr_decay**i)
                })

        # optionally, for the remaining layers (not otherwise finetuned) you can choose to still FT the batchnorm layers
        for i, block in enumerate(remaining_blocks):
            if self.always_train_batchnorm:
                raise NotImplementedError
                # _, block_batch_norm_params = get_batch_norm_params_lighting(block)
                # params.append({
                #     "params": block_batch_norm_params,
                #     "lr": lr * (self.lr_decay**i)
                # })


        logging.info('param groups: {}'.format(len(params)))

        # because it iterates through the generators, THIS BREAKS TRAINING so only uncomment to debug params
        # for param_group_n, param_group in enumerate(params):
        #     shapes_within_param_group = [p.shape for p in list(param_group['params'])]
        #     logging.debug('param group {}: {}'.format(param_group_n, shapes_within_param_group))
        # print('head params to optimize', [p.shape for p in params[0]['params']])  # head only
        # print(list(param_group['params']) for param_group in params)
        # exit()
        # Initialize AdamW optimizer

        opt = torch.optim.AdamW(params, weight_decay=self.weight_decay)  # lr included in params dict
        logging.info('Optimizer ready, configuring scheduler')

        if self.cosine_schedule:
            logging.info('Using lightly cosine schedule, warmup for {} epochs, max for {} epochs'.format(self.warmup_epochs, self.max_cosine_epochs))
            # from lightly.utils.scheduler import CosineWarmupScheduler  #copied from here to avoid dependency
            # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
            # Dictionary, with an "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_scheduler_config.
            lr_scheduler = schedulers.CosineWarmupScheduler(
                optimizer=opt,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_cosine_epochs,
                start_value=self.learning_rate,
                end_value=self.learning_rate * self.max_learning_rate_reduction_factor,
            )

            # logging.info('Using CosineAnnealingLR schedule, warmup not supported, max for {} epochs'.format(self.max_cosine_epochs))
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer=opt,
            #     T_max=self.max_cosine_epochs,
            #     eta_min=self.learning_rate * self.max_learning_rate_reduction_factor
            # )

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    'scheduler': lr_scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            logging.info('Learning rate scheduler not used')
        return opt
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.head(x)
        # TODO encoder output shape changes with input shape (of course) so need to specify explicitly or skip
        return x

    def make_step(self, batch):
        y, y_pred, loss = self.run_step_through_model(batch)
        return self.step_to_dict(y, y_pred, loss)

    def run_step_through_model(self, batch):
      # part of training/val/test for all subclasses
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)  # must be subclasses and specified
        return y, y_pred, loss

    def step_to_dict(self, y, y_pred, loss):
        return {'loss': loss.mean(), 'predictions': y_pred, 'labels': y}

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self.make_step(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.make_step(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.make_step(batch)
    
    def predict_step(self, batch, batch_idx) -> Any:
        # I can't work out how to get webdataset to return a single item im, not a tuple (im,).
        # this is fine for training but annoying for predict
        # help welcome. meanwhile, this works around it
        if isinstance(batch, list) and len(batch) == 1:
            return self(batch[0])
        return self(batch)

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx=0):
        # v2 docs currently do not show dataloader_idx as train argument so unclear if this will value be updated properly
        # arg is shown for val/test equivalents
        # currently does nothing in Zoobot so inconsequential
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-train-batch-end
        self.train_loss_metric(outputs['loss'])
        self.log(
            "finetuning/train_loss", 
            self.train_loss_metric, 
            prog_bar=self.prog_bar, 
            on_step=False,
            on_epoch=True
        )

    def on_validation_batch_end(self, outputs: dict, batch, batch_idx: int, dataloader_idx=0):
        self.val_loss_metric(outputs['loss'])
        self.log(
            "finetuning/val_loss", 
            self.val_loss_metric, 
            prog_bar=self.prog_bar, 
            on_step=False,
            on_epoch=True
        )
        # unique to val batch end
        if self.visualize_images:
          self.upload_images_to_wandb(outputs, batch, batch_idx)

    def on_test_batch_end(self, outputs: dict, batch, batch_idx: int, dataloader_idx=0):
        self.test_loss_metric(outputs['loss'])
        self.log(
            "finetuning/test_loss", 
            self.test_loss_metric, 
            prog_bar=self.prog_bar, 
            on_step=False,
            on_epoch=True
        )

# lighting v2. removed validation_epoch_end(self, outputs)
# now only has *on_*validation_epoch_end(self)
# replacing by using explicit torchmetric for loss
# https://github.com/Lightning-AI/lightning/releases/tag/2.0.0

    def upload_images_to_wandb(self, outputs, batch, batch_idx):
      raise NotImplementedError('Must be subclassed')
    
    @classmethod
    def load_from_name(cls, name: str, **kwargs):
        downloaded_loc = download_from_name(cls.__name__, name, **kwargs)
        return cls.load_from_checkpoint(downloaded_loc, **kwargs)  # trained on GPU, may need map_location='cpu' if you get a device error





class FinetuneableZoobotClassifier(FinetuneableZoobotAbstract):
    """
    Pretrained Zoobot model intended for finetuning on a classification problem.

    Any args not listed below are passed to :class:``FinetuneableZoobotAbstract`` (for example, `learning_rate`).
    These are shared between classifier, regressor, and tree models.
    See the docstring of :class:``FinetuneableZoobotAbstract`` for more.

    Models can be loaded in one of three ways:
    - HuggingFace name e.g. FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...). Recommended.
    - Any PyTorch model in memory e.g. FinetuneableZoobotClassifier(encoder=some_model, ...)
    - ZoobotTree checkpoint e.g. FinetuneableZoobotClassifier(zoobot_checkpoint_loc='path/to/zoobot_tree.ckpt', ...)

    Args:
        num_classes (int): num. of target classes (e.g. 2 for binary classification).
        label_smoothing (float, optional): See torch cross_entropy_loss docs. Defaults to 0.
        class_weights (arraylike, optional): See torch cross_entropy_loss docs. Defaults to None.
        
    """

    def __init__(
            self,
            num_classes: int,
            label_smoothing=0.,
            class_weights=None,
            **super_kwargs) -> None:

        super().__init__(**super_kwargs)

        logging.info('Using classification head and cross-entropy loss')
        self.head = LinearHead(
            input_dim=self.encoder_dim,
            output_dim=num_classes,
            dropout_prob=self.dropout_prob
        )
        self.label_smoothing = label_smoothing
        self.loss = partial(cross_entropy_loss,
                            weight=class_weights,
                            label_smoothing=self.label_smoothing)
        logging.info(f'num_classes: {num_classes}')
        if num_classes == 2:
            logging.info('Using binary classification')
            task = 'binary'
        else:
            logging.info('Using multi-class classification')
            task = 'multiclass'
        self.train_acc = tm.Accuracy(task=task, average="micro", num_classes=num_classes)
        self.val_acc = tm.Accuracy(task=task, average="micro", num_classes=num_classes)
        self.test_acc = tm.Accuracy(task=task, average="micro", num_classes=num_classes)
        
    def step_to_dict(self, y, y_pred, loss):
        y_class_preds = torch.argmax(y_pred, axis=1) # type: ignore
        return {'loss': loss.mean(), 'predictions': y_pred, 'labels': y, 'class_predictions': y_class_preds}

    def on_train_batch_end(self, step_output, *args):
        super().on_train_batch_end(step_output, *args)

        self.train_acc(step_output['class_predictions'], step_output['labels'])
        self.log(
            'finetuning/train_acc',
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar
        )
    
    def on_validation_batch_end(self, step_output, *args):
        super().on_validation_batch_end(step_output, *args)

        self.val_acc(step_output['class_predictions'], step_output['labels'])
        self.log(
            'finetuning/val_acc',
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar
        )

    def on_test_batch_end(self, step_output, *args) -> None:
        super().on_test_batch_end(step_output, *args)

        self.test_acc(step_output['class_predictions'], step_output['labels'])
        self.log(
            "finetuning/test_acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar
        )

    
    def predict_step(self, x: Union[list[torch.Tensor], torch.Tensor], batch_idx):
        # see Abstract version
        if isinstance(x, list) and len(x) == 1:
            return self(x[0])
        x = self.forward(x)  # type: ignore # logits from LinearHead
        # then applies softmax
        return F.softmax(x, dim=1)


    def upload_images_to_wandb(self, outputs, batch, batch_idx):
      # self.logger is set by pl.Trainer(logger=) argument
        if (self.logger is not None) and (batch_idx == 0):
            x, y = batch
            y_pred_softmax = F.softmax(outputs['predictions'], dim=1)
            n_images = 5
            images = [img for img in x[:n_images]]
            captions = [f'Ground Truth: {y_i} \nPrediction: {y_p_i}' for y_i, y_p_i in zip(
                y[:n_images], y_pred_softmax[:n_images])]
            self.logger.log_image( # type: ignore
                key='val_images',
                images=images,
                caption=captions)



class FinetuneableZoobotRegressor(FinetuneableZoobotAbstract):
    """
    Pretrained Zoobot model intended for finetuning on a regression problem.    

    Any args not listed below are passed to :class:``FinetuneableZoobotAbstract`` (for example, `learning_rate`).
    These are shared between classifier, regressor, and tree models.
    See the docstring of :class:``FinetuneableZoobotAbstract`` for more.

    Models can be loaded in one of three ways:
    - HuggingFace name e.g. FinetuneableZoobotRegressor(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...). Recommended.
    - Any PyTorch model in memory e.g. FinetuneableZoobotRegressor(encoder=some_model, ...)
    - ZoobotTree checkpoint e.g. FinetuneableZoobotRegressor(zoobot_checkpoint_loc='path/to/zoobot_tree.ckpt', ...)


    Args:
        loss (str, optional): Loss function to use. Must be one of 'mse', 'mae'. Defaults to 'mse'.
        unit_interval (bool, optional): If True, use sigmoid activation for the final layer, ensuring predictions between 0 and 1. Defaults to False.
        
    """

    def __init__(
            self,
            loss:str='mse',
            unit_interval:bool=False, 
            **super_kwargs) -> None:

        super().__init__(**super_kwargs)

        self.unit_interval = unit_interval
        if self.unit_interval:
            logging.info('unit_interval=True, using sigmoid activation for finetunng head')
            head_activation = torch.nn.functional.sigmoid
        else:
            head_activation = None
    
        logging.info('Using classification head and cross-entropy loss')
        self.head = LinearHead(
            input_dim=self.encoder_dim,
            output_dim=1,
            dropout_prob=self.dropout_prob,
            activation=head_activation
        )
        if loss in ['mse', 'mean_squared_error']:
            self.loss = mse_loss
        elif loss in ['mae', 'mean_absolute_error', 'l1', 'l1_loss']:
            self.loss = l1_loss
        else:
            raise ValueError(f'Loss {loss} not recognised. Must be one of mse, mae')

        # rmse metrics. loss is mse already.
        self.train_rmse = tm.MeanSquaredError(squared=False)
        self.val_rmse = tm.MeanSquaredError(squared=False)
        self.test_rmse = tm.MeanSquaredError(squared=False)
        
    def step_to_dict(self, y, y_pred, loss):
        return {'loss': loss.mean(), 'predictions': y_pred, 'labels': y}

    def on_train_batch_end(self, step_output, *args):
        super().on_train_batch_end(step_output, *args)

        self.train_rmse(step_output['predictions'], step_output['labels'])
        self.log(
            'finetuning/train_rmse',
            self.train_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar
        )
    
    def on_validation_batch_end(self, step_output, *args):
        super().on_validation_batch_end(step_output, *args)

        self.val_rmse(step_output['predictions'], step_output['labels'])
        self.log(
            'finetuning/val_rmse',
            self.val_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar
        )

    def on_test_batch_end(self, step_output, *args) -> None:
        super().on_test_batch_end(step_output, *args)

        self.test_rmse(step_output['predictions'], step_output['labels'])
        self.log(
            "finetuning/test_rmse",
            self.test_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar
        )

    
    def predict_step(self, x: Union[list[torch.Tensor], torch.Tensor], batch_idx):
        # see Abstract version
        if isinstance(x, list) and len(x) == 1:
            return self(x[0])
        return self.forward(x)


class FinetuneableZoobotTree(FinetuneableZoobotAbstract):
    """
    Pretrained Zoobot model intended for finetuning on a decision tree (i.e. GZ-like) problem. 
    Uses Dirichlet-Multinomial loss introduced in GZ DECaLS.
    Briefly: predicts a Dirichlet distribution for the probability of a typical volunteer giving each answer, 
    and uses the Dirichlet-Multinomial loss to compare the predicted distribution of votes (given k volunteers were asked) to the true distribution.

    Does not produce accuracy or MSE metrics, as these are not relevant for this task. Loss logging only.

    If you're using this, you're probably working on a Galaxy Zoo catalog, and you should Slack Mike!

    Any args not listed below are passed to :class:``FinetuneableZoobotAbstract`` (for example, `learning_rate`).
    These are shared between classifier, regressor, and tree models.
    See the docstring of :class:``FinetuneableZoobotAbstract`` for more.

    Models can be loaded in one of three ways:
    - HuggingFace name e.g. FinetuneableZoobotRegressor(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...). Recommended.
    - Any PyTorch model in memory e.g. FinetuneableZoobotRegressor(encoder=some_model, ...)
    - ZoobotTree checkpoint e.g. FinetuneableZoobotRegressor(zoobot_checkpoint_loc='path/to/zoobot_tree.ckpt', ...)

    Args:
        schema (schemas.Schema): description of the layout of the decision tree. See :class:`zoobot.shared.schemas.Schema`.
    """

    def __init__(
        self,
        schema: schemas.Schema,
        **super_kwargs
    ):

        super().__init__(**super_kwargs)

        logging.info('Using dropout+dirichlet head and dirichlet (count) loss')

        self.schema = schema
        self.output_dim = len(self.schema.label_cols)

        self.head = define_model.get_pytorch_dirichlet_head(
            encoder_dim=self.encoder_dim,
            output_dim=self.output_dim,
            test_time_dropout=False,
            dropout_rate=self.dropout_prob
        )
      
        self.loss = define_model.get_dirichlet_loss_func(self.schema.question_index_groups)

    def upload_images_to_wandb(self, outputs, batch, batch_idx):
      raise NotImplementedError

    # other functions are simply inherited from FinetunedZoobotAbstract

class LinearHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_prob=0.5, activation=None):
        """
        Small utility class for a linear head with dropout and optional choice of activation.

        - Apply dropout to features before the final linear layer.
        - Apply a final linear layer
        - Optionally, apply `activation` callable

        Args:
            input_dim (int): input dim of the linear layer (i.e. the encoder output dimension)
            output_dim (int): output dim of the linear layer (often e.g. N for N classes, or 1 for regression)
            dropout_prob (float, optional): Dropout probability. Defaults to 0.5.
            activation (callable, optional): callable expecting tensor e.g. torch softmax. Defaults to None.
        """
        # input dim is representation dim, output_dim is num classes
        super(LinearHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        # returns logits, as recommended for CrossEntropy loss
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.output_dim == 1:
            return x.squeeze()
        else:
            return x



def cross_entropy_loss(y_pred: torch.Tensor, y: torch.Tensor, label_smoothing: float=0., weight=None):
    """
    Calculate cross-entropy loss with optional label smoothing and class weights. No aggregation applied.
    Trivial wrapper of torch.nn.functional.cross_entropy with reduction='none'.

    Args:
        y_pred (torch.Tensor): ints of shape (batch)
        y (torch.Tensor): predictions of shape (batch, classes)
        label_smoothing (float, optional): See docstring of torch.nn.functional.cross_entropy. Defaults to 0..
        weight (arraylike, optional): See docstring of torch.nn.functional.cross_entropy. Defaults to None.

    Returns:
        torch.Tensor: unreduced cross-entropy loss
    """
    return F.cross_entropy(y_pred, y.long(), label_smoothing=label_smoothing, weight=weight, reduction='none')


def mse_loss(y_pred, y):
    """
    Trivial wrapper of torch.nn.functional.mse_loss with reduction='none'.

    Args:
        y_pred (torch.Tensor): See docstring of torch.nn.functional.mse_loss.
        y (torch.Tensor): See docstring of torch.nn.functional.mse_loss.

    Returns:
        torch.Tensor: See docstring of torch.nn.functional.mse_loss.
    """
    return F.mse_loss(y_pred, y, reduction='none')

def l1_loss(y_pred, y):
    """
    Trivial wrapper of torch.nn.functional.l1_loss with reduction='none'.

    Args:
        y_pred (torch.Tensor): See docstring of torch.nn.functional.l1_loss.
        y (torch.Tensor): See docstring of torch.nn.functional.l1_loss.

    Returns:
        torch.Tensor: See docstring of torch.nn.functional.l1_loss.
    """
    return F.l1_loss(y_pred, y, reduction='none')


def dirichlet_loss(y_pred: torch.Tensor, y: torch.Tensor, question_index_groups):
    """
    Calculate Dirichlet-Multinomial loss for a batch of predictions and labels.
    Returns a scalar loss (ready for gradient descent) by summing across answers and taking a mean across the batch.
    Reduction equivalent to sum(axis=1).mean(), but with fewer operations.

    Args:
        y_pred (torch.Tensor): Predicted dirichlet distribution, of shape (batch, answers)
        y (torch.Tensor): Count of volunteer votes for each answer, of shape (batch, answers)
        question_index_groups (list): Answer indices for each question i.e. [(question.start_index, question.end_index), ...] for all questions. Useful for slicing model predictions by question. See :ref:`schemas`.

    Returns:
        torch.Tensor: Dirichlet-Multinomial loss. Scalar, summing across answers and taking a mean across the batch i.e. sum(axis=1).mean())
    """
    # my func uses sklearn convention y, y_pred
    return losses.calculate_multiquestion_loss(y, y_pred, question_index_groups).mean()*len(question_index_groups)



def load_pretrained_zoobot(checkpoint_loc: str) -> torch.nn.Module:
    """
    Args:
        checkpoint_loc (str): path to saved LightningModule checkpoint, likely of :class:`ZoobotTree`, :class:`FinetuneableZoobotClassifier`, or :class:`FinetunabelZoobotTree`. Must have .zoobot attribute.

    Returns:
        torch.nn.Module: pretrained PyTorch encoder within that LightningModule.
    """
    if torch.cuda.is_available():
        map_location = None
    else:
        # necessary to load gpu-trained model on cpu
        map_location = torch.device('cpu')
    return define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc, map_location=map_location).encoder # type: ignore
    

def get_trainer(
    save_dir: str,
    file_template="{epoch}",
    save_top_k=1,
    max_epochs=100,
    patience=10,
    devices='auto',
    accelerator='auto',
    logger=None,
    **trainer_kwargs
) -> pl.Trainer:
    """
    Convenience wrapper to create a PyTorch Lightning Trainer that carries out the finetuning process.
    Use like so: trainer.fit(model, datamodule)

    `get_trainer` args are for common Trainer settings e.g. early stopping checkpointing, etc. By default:
    - Saves the top-k models based on validation loss
    - Uses early stopping with `patience` i.e. end training if validation loss does not improve after `patience` epochs.
    - Monitors the learning rate (useful when using a learning rate scheduler)

    Any extra args not listed below are passed directly to the PyTorch Lightning Trainer.
    Use this to add any custom configuration not covered by the `get_trainer` args.
    See https://lightning.ai/docs/pytorch/stable/common/trainer.html

    Args:
        save_dir (str): folder in which to save checkpoints and logs.
        file_template (str, optional): custom naming for checkpoint files. See Lightning docs. Defaults to "{epoch}".
        save_top_k (int, optional): save the top k checkpoints only. Defaults to 1.
        max_epochs (int, optional): train for up to this many epochs. Defaults to 100.
        patience (int, optional): wait up to this many epochs for decreasing loss before ending training. Defaults to 10.
        devices (str, optional): number of devices for training (typically, num. GPUs). Defaults to 'auto'.
        accelerator (str, optional): which device to use (typically 'gpu' or 'cpu'). Defaults to 'auto'.
        logger (pl.loggers.wandb_logger, optional): If pl.loggers.wandb_logger, track experiment on Weights and Biases. Defaults to None.

    Returns:
        pl.Trainer: PyTorch Lightning trainer object for finetuning a model on a GalaxyDataModule.
    """

    checkpoint_callback = ModelCheckpoint(
        monitor='finetuning/val_loss',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=os.path.join(save_dir, 'checkpoints'),
        filename=file_template,
        save_weights_only=True,
        save_top_k=save_top_k
    )

    early_stopping_callback = EarlyStopping(
        monitor='finetuning/val_loss',
        mode='min',
        patience=patience
    )

    learning_rate_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    # Initialise pytorch lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, learning_rate_monitor_callback],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        **trainer_kwargs,
    )

    return trainer


def download_from_name(class_name: str, hub_name: str):
    """
    Download a finetuned model from the HuggingFace Hub by name.
    Used to load pretrained Zoobot models by name, e.g. FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...).

    Downloaded models are saved to the HuggingFace cache directory for later use (typically ~/.cache/huggingface).

    You shouldn't need to call this; it's used internally by the FinetuneableZoobot classes.

    Args:
        class_name (str): one of FinetuneableZoobotClassifier, FinetuneableZoobotRegressor, FinetuneableZoobotTree
        hub_name (str): e.g. mwalmsley/zoobot-encoder-convnext_nano

    Returns:
        str: path to downloaded model (in HuggingFace cache directory). Likely then loaded by Lightning.
    """
    from huggingface_hub import hf_hub_download

    if hub_name.startswith('hf_hub:'):
        logging.info('Passed name with hf_hub: prefix, dropping prefix')
        repo_id = hub_name.split('hf_hub:')[1]
    else:
        repo_id = hub_name
    downloaded_loc = hf_hub_download(
        repo_id=repo_id,
        filename=f"{class_name}.ckpt"
    )
    return downloaded_loc




def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int] = None,
) -> float:
    """
    Use cosine decay to gradually modify start_value to reach target end_value during
    iterations.
    Copied from lightly library (thank you for open sourcing)

    Args:
        step:
            Current step number.
        max_steps:
            Total number of steps.
        start_value:
            Starting value.
        end_value:
            Target value.
        period (optional):
            The number of steps over which the cosine function completes a full cycle.
            If not provided, it defaults to max_steps.

    Returns:
        Cosine decay value.

    """
    if step < 0:
        raise ValueError("Current step number can't be negative")
    if max_steps < 1:
        raise ValueError("Total step number must be >= 1")
    if period is None and step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )
    if period is not None and period <= 0:
        raise ValueError("Period must be >= 1")

    decay: float
    if period is not None:  # "cycle" based on period, if provided
        decay = (
            end_value
            - (end_value - start_value) * (np.cos(2 * np.pi * step / period) + 1) / 2
        )
    elif max_steps == 1:
        # Avoid division by zero
        decay = end_value
    elif step == max_steps:
        # Special case for Pytorch Lightning which updates LR scheduler also for epoch
        # after last training epoch.
        decay = end_value
    else:
        decay = (
            end_value
            - (end_value - start_value)
            * (np.cos(np.pi * step / (max_steps - 1)) + 1)
            / 2
        )
    return decay

