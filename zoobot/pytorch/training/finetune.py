# Based on Inigo's BYOL FT step
# https://github.com/inigoval/finetune/blob/main/finetune.py
import logging
import os
from typing import Any, Union
import warnings
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import torch
import torch.nn.functional as F
import torchmetrics as tm
import timm


from foundation.models.mae_lightly import CustomMAEEncoder

from zoobot.pytorch.training import losses
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
    Parent class of :class:`FinetuneableZoobotClassifier` and :class:`FinetuneableZoobotTree`.
    You cannot use this class directly - you must use the child classes above instead.

    This class defines the finetuning methods that those child classes both use.
    For example: when provided `checkpoint_loc`, it will load the encoder from that checkpoint.
    Both :class:`FinetuneableZoobotClassifier` and :class:`FinetuneableZoobotTree`
    can (and should) be passed any of these arguments to customise finetuning.

    You could subclass this class to solve new finetuning tasks (like regression) - see :ref:`advanced_finetuning`.

    Args:
        checkpoint_loc (str, optional): Path to encoder checkpoint to load (likely a saved ZoobotTree). Defaults to None.
        encoder (pl.LightningModule, optional): Alternatively, pass an encoder directly. Load with :func:`zoobot.pytorch.training.finetune.load_pretrained_encoder`.
        encoder_dim (int, optional): Output dimension of encoder. Defaults to 1280 (EfficientNetB0's encoder dim).
        lr_decay (float, optional): For each layer i below the head, reduce the learning rate by lr_decay ^ i. Defaults to 0.75.
        weight_decay (float, optional): AdamW weight decay arg (i.e. L2 penalty). Defaults to 0.05.
        learning_rate (float, optional): AdamW learning rate arg. Defaults to 1e-4.
        dropout_prob (float, optional): P of dropout before final output layer. Defaults to 0.5.
        always_train_batchnorm (bool, optional): If True, do not update batchnorm stats during finetuning. Defaults to True.
        prog_bar (bool, optional): Print progress bar during finetuning. Defaults to True.
        visualize_images (bool, optional): Upload example images to WandB. Good for debugging but slow. Defaults to False.
        seed (int, optional): random seed to use. Defaults to 42.
    """

    def __init__(
        self,
        # can provide either zoobot_checkpoint_loc, and will load this model as encoder...
        zoobot_checkpoint_loc=None,
        # ...or directly pass any model to use as encoder (if you do this, you will need to keep it around for later)
        encoder=None,
        n_blocks=0,  # how many layers deep to FT
        lr_decay=0.75,
        weight_decay=0.05,
        learning_rate=1e-4,  # 10x lower than typical, you may like to experiment
        dropout_prob=0.5,
        always_train_batchnorm=True,
        prog_bar=True,
        visualize_images=False,  # upload examples to wandb, good for debugging
        seed=42,
        n_layers=0,  # for backward compat., n_blocks preferred
        # these args are for the optional learning rate scheduler, best not to use unless you've tuned everything else already
        cosine_schedule=False,
        warmup_epochs=10,
        max_cosine_epochs=100,
        max_learning_rate_reduction_factor=0.01
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

        if zoobot_checkpoint_loc is not None:
          assert encoder is None, 'Cannot pass both checkpoint to load and encoder to use'
          self.encoder = load_pretrained_zoobot(zoobot_checkpoint_loc)
        else:
          assert zoobot_checkpoint_loc is None, 'Cannot pass both checkpoint to load and encoder to use'
          assert encoder is not None, 'Must pass either checkpoint to load or encoder to use'
          self.encoder = encoder

        # TODO read as encoder property
        if isinstance(self.encoder, CustomMAEEncoder):
            self.encoder_dim = 256  # vit hidden dim, assuming average pool over seq dim
            # self.encoder_dim = 9216
        else:
            self.encoder_dim = define_model.get_encoder_dim(self.encoder)
        self.n_blocks = n_blocks
        logging.info('Blocks to finetune: {}'.format(n_layers))

        # for backwards compat.
        if n_layers:
            logging.warning('FinetuneableZoobot(n_layers) is now renamed to n_blocks, please update to pass n_blocks instead! For now, setting n_blocks=n_layers')
            self.n_blocks = n_layers
            logging.info('Layers to finetune: {}'.format(n_layers))

        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob

        self.cosine_schedule = cosine_schedule
        self.warmup_epochs = warmup_epochs
        self.max_cosine_epochs = max_cosine_epochs
        self.max_learning_rate_reduction_factor = max_learning_rate_reduction_factor

        self.always_train_batchnorm = always_train_batchnorm
        if self.always_train_batchnorm:
            logging.info('always_train_batchnorm=True, so all batch norm layers will be finetuned')

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

        if isinstance(self.encoder, CustomMAEEncoder):
            logging.info('Using custom optimizer for MAE encoder')
            # equivalent to usual, but in param_group format
            head_param_groups = [
                {'params': self.head.parameters(),
                 'weight_decay': self.weight_decay,
                 'lr_scale': 1.  # no lr decay on head
                }
            ]
            # now custom bit for the encoder
            encoder_param_groups = self.encoder.get_param_groups(self.weight_decay, self.lr_decay)
            n_param_groups_to_tune = self.n_blocks * 2 # finetune top N. First layer is pos embedding, then pairs of decay/no decay, 18 pairs by default
            if n_param_groups_to_tune > len(encoder_param_groups):
                logging.warning('more param groups (blocks*2) specified to finetune than available')
            encoder_param_groups_to_tune = encoder_param_groups[-n_param_groups_to_tune:]  
            param_groups = encoder_param_groups_to_tune + head_param_groups
            return torch.optim.AdamW(param_groups, lr=self.learning_rate)

        lr = self.learning_rate
        params = [{"params": self.head.parameters(), "lr": lr}]

        logging.info(f'Encoder architecture to finetune: {type(self.encoder)}')

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
        for param_group_n, param_group in enumerate(params):
            shapes_within_param_group = [p.shape for p in list(param_group['params'])]
            logging.info('param group {}: {}'.format(param_group_n, shapes_within_param_group))
        # print('head params to optimize', [p.shape for p in params[0]['params']])  # head only
        # print(list(param_group['params']) for param_group in params)
        # exit()
        # Initialize AdamW optimizer
        opt = torch.optim.AdamW(params, weight_decay=self.weight_decay)  # lr included in params dict

        if self.cosine_schedule:
            logging.info('Using cosine schedule, warmup for {} epochs, max for {} epochs'.format(self.warmup_epochs, self.max_cosine_epochs))
            from lightly.utils.scheduler import CosineWarmupScheduler  # new dependency for zoobot, TBD - maybe just copy
            # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
            # Dictionary, with an "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_scheduler_config.
            lr_scheduler = CosineWarmupScheduler(
                optimizer=opt,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_cosine_epochs,
                start_value=self.learning_rate,
                end_value=self.learning_rate * self.max_learning_rate_reduction_factor,
            )
            # lr_scheduler_config default is frequency=1, interval=epoch
            return {
                "optimizer": opt,
                "lr_scheduler": lr_scheduler
            }
        else:
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



class FinetuneableZoobotClassifier(FinetuneableZoobotAbstract):
    """
    Pretrained Zoobot model intended for finetuning on a classification problem.

    You must also pass either ``checkpoint_loc`` (to a saved encoder checkpoint)
    or `encoder` (to a pytorch model already loaded in memory).
    See :class:FinetuneableZoobotAbstract for more options.

    Any args not in the list below are passed to :class:``FinetuneableZoobotAbstract`` (usually to specify how to carry out the finetuning)

    Args:
        num_classes (int): num. of target classes (e.g. 2 for binary classification).
        label_smoothing (float, optional): See torch cross_entropy_loss docs. Defaults to 0.
        
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

    See FinetuneableZoobotClassifier, above

    Args:
        None besides those from FinetuneableZoobotAbstract, above (1 class, MSE error, for now)
        
    """

    def __init__(
            self,
            unit_interval=False, 
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
        self.loss = mse_loss
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

    # TODO
    # def upload_images_to_wandb(self, outputs, batch, batch_idx):
    #   # self.logger is set by pl.Trainer(logger=) argument
    #     if (self.logger is not None) and (batch_idx == 0):
    #         x, y = batch
    #         y_pred_softmax = F.softmax(outputs['predictions'], dim=1)
    #         n_images = 5
    #         images = [img for img in x[:n_images]]
    #         captions = [f'Ground Truth: {y_i} \nPrediction: {y_p_i}' for y_i, y_p_i in zip(
    #             y[:n_images], y_pred_softmax[:n_images])]
    #         self.logger.log_image( # type: ignore
    #             key='val_images',
    #             images=images,
    #             caption=captions)


class FinetuneableZoobotTree(FinetuneableZoobotAbstract):
    """
    Pretrained Zoobot model intended for finetuning on a decision tree (i.e. GZ-like) problem.

    You must also pass either ``checkpoint_loc`` (to a saved encoder checkpoint)
    or ``encoder`` (to a pytorch model already loaded in memory).
    See :class:FinetuneableZoobotAbstract for more options.

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
      pass  # not yet implemented

    # other functions are simply inherited from FinetunedZoobotAbstract

# https://github.com/inigoval/byol/blob/1da1bba7dc5cabe2b47956f9d7c6277decd16cc7/byol_main/networks/models.py#L29
class LinearHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5, activation=None):
        # input dim is representation dim, output_dim is num classes
        super(LinearHead, self).__init__()
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



def cross_entropy_loss(y_pred, y, label_smoothing=0., weight=None):
    # y should be shape (batch) and ints
    # y_pred should be shape (batch, classes)
    # returns loss of shape (batch)
    # will reduce myself
    return F.cross_entropy(y_pred, y.long(), label_smoothing=label_smoothing, weight=weight, reduction='none')

def mse_loss(y_pred, y):
    # y should be shape (batch) and ints
    # y_pred should be shape (batch, classes)
    # returns loss of shape (batch)
    # will reduce myself
    return F.mse_loss(y_pred, y, reduction='none')


def dirichlet_loss(y_pred, y, question_index_groups):
    # aggregation equiv. to sum(axis=1).mean(), but fewer operations
    # returns loss of shape (batch)
    # my func uses sklearn convention y, y_pred
    return losses.calculate_multiquestion_loss(y, y_pred, question_index_groups).mean()*len(question_index_groups)


class FinetunedZoobotClassifierBaseline(FinetuneableZoobotClassifier):
    # exactly as the Finetuned model above, but with a simple single learning rate
    # useful for training from-scratch model exactly as if it were finetuned, as a baseline

    def configure_optimizers(self):
        head_params = list(self.head.parameters())
        encoder_params = list(self.encoder.parameters())
        return torch.optim.AdamW(head_params + encoder_params, lr=self.learning_rate)


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
    PyTorch Lightning Trainer that carries out the finetuning process.
    Use like so: trainer.fit(model, datamodule)

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

# TODO check exactly which layers get FTd
# def is_tuneable(block_of_layers):
#     if len(list(block_of_layers.parameters())) == 0:
#         logging.info('Skipping block with no params')
#         logging.info(block_of_layers)
#         return False
#     else:
#         # currently, allowed to include batchnorm
#         return True
    
# def get_batch_norm_params_lighting(parent_module, checked_params=set(), batch_norm_params=[]):

#     modules = parent_module.modules()
#     for module in modules:
#         if id(module) not in checked_params:
#             checked_params.add(id(module))
#             if isinstance(module, torch.nn.BatchNorm2d):
#                 batch_norm_params += module.parameters()
#             else:
#                 checked_params, batch_norm_params = get_batch_norm_params_lighting(module, checked_params, batch_norm_params)

#     return checked_params, batch_norm_params



    # when ready (don't peek often, you'll overfit)
    # trainer.test(model, dataloaders=datamodule)

    # return model, checkpoint_callback.best_model_path
    # trainer.callbacks[checkpoint_callback].best_model_path?

# def investigate_structure():

#     from zoobot.pytorch.estimators import define_model


#     model = define_model.get_plain_pytorch_zoobot_model(output_dim=1280, include_top=False)

#     # print(model)
#     # with include_top=False, first and only child is EffNet
#     effnet_with_pool = list(model.children())[0]

#     # 0th is actually EffNet, 1st and 2nd are AvgPool and Identity
#     effnet = list(effnet_with_pool.children())[0]

#     for layer_n, layer in enumerate(effnet.children()):
#         # first bunch are Sequential module wrapping e.g. 3 MBConv blocks
#         print('\n', layer_n)
#         if isinstance(layer, torch.nn.Sequential):
#             print(layer)
#     # so the blocks to finetune are each Sequential (repeated MBConv) block
#     # and other blocks can be left alone
#     # (also be careful to leave batch-norm alone)
