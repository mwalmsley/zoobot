import logging
from functools import partial
from typing import List

import pytorch_lightning as pl
import timm
import torch
import torchmetrics
from torch import nn

from zoobot.pytorch.estimators import custom_layers, efficientnet_custom
from zoobot.pytorch.training import losses, schedulers
from zoobot.shared import schemas

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

    # args will be saved as hparams
    def __init__(self, *args):
        super().__init__()
        # saves all args by default
        self.save_hyperparameters()

    # may sometimes want to ignore nan even in main metrics
    def setup_metrics(self, nan_strategy="error"):
        self.val_accuracy = torchmetrics.Accuracy(task="binary")

        self.loss_metrics = torch.nn.ModuleDict(
            {
                "train/supervised_loss": torchmetrics.MeanMetric(
                    nan_strategy=nan_strategy
                ),
                "validation/supervised_loss": torchmetrics.MeanMetric(
                    nan_strategy=nan_strategy
                ),
                "test/supervised_loss": torchmetrics.MeanMetric(
                    nan_strategy=nan_strategy
                ),
            }
        )

        # TODO handle when schema doesn't exist
        question_metric_dict = {}
        for step_name in ["train", "validation", "test"]:
            question_metric_dict.update(
                {
                    step_name
                    + "/question_loss/"
                    + question.text: torchmetrics.MeanMetric(nan_strategy="ignore")
                    for question in self.schema.questions
                }
            )
        self.question_loss_metrics = torch.nn.ModuleDict(question_metric_dict)

        campaigns = schema_to_campaigns(self.schema)
        campaign_metric_dict = {}
        for step_name in ["train", "validation", "test"]:
            campaign_metric_dict.update(
                {
                    step_name
                    + "/campaign_loss/"
                    + campaign: torchmetrics.MeanMetric(nan_strategy="ignore")
                    for campaign in campaigns
                }
            )
        self.campaign_loss_metrics = torch.nn.ModuleDict(campaign_metric_dict)

    def forward(self, x):
        assert x.shape[1] < 4  # torchlike BCHW
        x = self.encoder(x)
        return self.head(x)

    def make_step(self, batch, step_name):
        x, labels = batch
        predictions = self(x)  # by default, these are Dirichlet concentrations
        loss = self.calculate_loss_and_update_loss_metrics(
            predictions, labels, step_name
        )
        outputs = {"loss": loss, "predictions": predictions, "labels": labels}
        # self.update_other_metrics(outputs, step_name=step_name)
        return outputs

    def configure_optimizers(self):
        raise NotImplementedError("Must be subclassed")

    def training_step(self, batch, batch_idx):
        return self.make_step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.make_step(batch, step_name="validation")

    def test_step(self, batch, batch_idx):
        return self.make_step(batch, step_name="test")

    # def on_train_batch_end(self, outputs, *args):
    #     pass

    # def on_validation_batch_end(self, outputs, *args):
    #     pass

    def on_train_epoch_end(self) -> None:
        # called *after* on_validation_epoch_end, confusingly
        # do NOT log_all_metrics here.
        # logging a metric resets it, and on_validation_epoch_end just logged and reset everything, so you will only log nans
        self.log_all_metrics(subset="train")

    def on_validation_epoch_end(self) -> None:
        self.log_all_metrics(subset="validation")

    def on_test_epoch_end(self) -> None:
        # logging.info('start test epoch end')
        self.log_all_metrics(subset="test")
        # logging.info('end test epoch end')

    def calculate_loss_and_update_loss_metrics(self, predictions, labels, step_name):
        raise NotImplementedError("Must be subclassed")

    def update_other_metrics(self, outputs, step_name):
        raise NotImplementedError("Must be subclassed")

    def log_all_metrics(self, subset=None):
        if subset is not None:
            for metric_collection in (
                self.loss_metrics,
                self.question_loss_metrics,
                self.campaign_loss_metrics,
            ):
                prog_bar = metric_collection == self.loss_metrics
                for name, metric in metric_collection.items():
                    if subset in name:
                        # logging.info(name)
                        self.log(
                            name,
                            metric,
                            on_epoch=True,
                            on_step=False,
                            prog_bar=prog_bar,
                            logger=True,
                        )
        else:  # just log everything
            self.log_dict(
                self.loss_metrics,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )
            self.log_dict(
                self.question_loss_metrics, on_step=False, on_epoch=True, logger=True
            )
            self.log_dict(
                self.campaign_loss_metrics, on_step=False, on_epoch=True, logger=True
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # I can't work out how to get webdataset to return a single item im, not a tuple (im,).
        # this is fine for training but annoying for predict
        # help welcome. meanwhile, this works around it
        if isinstance(batch, list) and len(batch) == 1:
            return self(batch[0])
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
        architecture_name (str, optional): Architecture to use. Passed to timm. Must be in timm.list_models(). Defaults to "efficientnet_b0".
        channels (int, optional): Num. input channels. Probably 3 or 1. Defaults to 1.
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
        # in the simplest case, this is all zoobot needs: grouping of label col indices as questions
        # question_index_groups: List=None,
        # BUT
        # if you pass these, it enables better per-question and per-survey logging (because we have names)
        # must be passed as simple dicts, not objects, so can't just pass schema in
        question_answer_pairs: dict = None,
        dependencies: dict = None,
        # encoder args
        architecture_name="efficientnet_b0",
        channels=1,
        # use_imagenet_weights=False,
        test_time_dropout=True,
        compile_encoder=False,
        timm_kwargs={},  # passed to timm.create_model e.g. drop_path_rate=0.2 for effnet
        # head args
        dropout_rate=0.2,
        learning_rate=1e-3,  # PyTorch default
        # optim args
        betas=(0.9, 0.999),  # PyTorch default
        weight_decay=0.01,  # AdamW PyTorch default
        scheduler_params={},  # no scheduler by default
    ):

        # now, finally, can pass only standard variables as hparams to save
        # will still need to actually use these variables later, this super init only saves them
        super().__init__(
            # these all do nothing, they are simply saved by lightning as hparams
            output_dim,
            question_answer_pairs,
            dependencies,
            architecture_name,
            channels,
            timm_kwargs,
            compile_encoder,
            test_time_dropout,
            dropout_rate,
            learning_rate,
            betas,
            weight_decay,
            scheduler_params,
        )

        logging.info("Generic __init__ complete - moving to Zoobot __init__")

        # logging.info('question_index_groups/dependencies passed to Zoobot, constructing schema in __init__')
        # assert question_index_groups is None,  "Don't pass both question_index_groups and question_answer_pairs/dependencies"
        assert dependencies is not None
        self.schema = schemas.Schema(question_answer_pairs, dependencies)
        # replace with schema-derived version
        question_index_groups = self.schema.question_index_groups

        self.setup_metrics()

        # set attributes for learning rate, betas, used by self.configure_optimizers()
        # TODO refactor to optimizer params
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params

        self.encoder = get_pytorch_encoder(
            architecture_name,
            channels,
            # use_imagenet_weights=use_imagenet_weights,
            **timm_kwargs,
        )
        if compile_encoder:
            logging.warning("Using torch.compile on encoder")
            self.encoder = torch.compile(self.encoder)

        # bit lazy assuming 224 input size
        # logging.warning(channels)
        self.encoder_dim = get_encoder_dim(self.encoder, channels)
        # typically encoder_dim=1280 for effnetb0
        logging.info("encoder dim: {}".format(self.encoder_dim))

        self.head = get_pytorch_dirichlet_head(
            self.encoder_dim,
            output_dim=output_dim,
            test_time_dropout=test_time_dropout,
            dropout_rate=dropout_rate,
        )

        self.loss_func = get_dirichlet_loss_func(question_index_groups)

        logging.info("Zoobot __init__ complete")

    def calculate_loss_and_update_loss_metrics(self, predictions, labels, step_name):
        # self.loss_func returns shape of (galaxy, question), mean to ()
        multiq_loss = self.loss_func(predictions, labels, sum_over_questions=False)
        self.update_per_question_loss_metric(multiq_loss, step_name=step_name)
        # sum over questions and take a per-device mean
        # for DDP strategy, batch size is constant (batches are not divided, data pool is divided)
        # so this will be the global per-example mean
        loss = torch.mean(torch.sum(multiq_loss, axis=1))
        self.loss_metrics[step_name + "/supervised_loss"](loss)
        return loss

    def configure_optimizers(self):
        # designed for training from scratch
        # parameters = list(self.head.parameters()) + list(self.encoder.parameters()) TODO should happen automatically?
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_params.get("name", None) == "plateau":
            logging.info(f"Using Plateau scheduler with {self.scheduler_params}")
            # TODO could generalise this if needed
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                min_lr=1e-6,
                patience=self.scheduler_params.get("patience", 5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "validation/loss",
            }
        elif self.scheduler_params.get("cosine_schedule", False):
            logging.info("Using cosine schedule")
            scheduler = schedulers.CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=self.scheduler_params["warmup_epochs"],
                max_epochs=self.scheduler_params["max_cosine_epochs"],
                start_value=self.learning_rate,
                end_value=self.learning_rate
                * self.scheduler_params["max_learning_rate_reduction_factor"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "validation/loss",
            }
        else:
            logging.info("No scheduler used")
            return optimizer  # no scheduler

    def update_per_question_loss_metric(self, multiq_loss, step_name):
        # log questions individually
        # TODO need schema attribute or similar to have access to question names, this will do for now
        # unlike Finetuneable..., does not use TorchMetrics, simply logs directly
        # TODO could use TorchMetrics and for q in schema, self.q_metric loop

        # if hasattr(self, 'schema'):
        # use schema metadata to log intelligently
        # will have schema if question_answer_pairs and dependencies are passed to __init__
        # assume that questions are named like smooth-or-featured-CAMPAIGN
        for question_n, question in enumerate(self.schema.questions):
            # for logging comparison, want to ignore loss on unlablled examples, i.e. take mean ignoring zeros
            # could sum, but then this would vary with batch size
            nontrivial_loss_mask = (
                multiq_loss[:, question_n] > 0
            )  # 'zero' seems to be ~5e-5 floor in practice

            this_question_metric = self.question_loss_metrics[
                step_name + "/question_loss/" + question.text
            ]
            # raise ValueError
            this_question_metric(
                torch.mean(multiq_loss[nontrivial_loss_mask, question_n])
            )

        campaigns = schema_to_campaigns(self.schema)
        for campaign in campaigns:
            campaign_questions = [
                q for q in self.schema.questions if campaign in q.text
            ]
            campaign_q_indices = [
                self.schema.questions.index(q) for q in campaign_questions
            ]  # shape (num q in this campaign e.g. 10)

            # similarly to per-question, only include in mean if (any) q in this campaign has a non-trivial loss
            nontrivial_loss_mask = (
                multiq_loss[:, campaign_q_indices].sum(axis=1) > 0
            )  # shape batch size

            this_campaign_metric = self.campaign_loss_metrics[
                step_name + "/campaign_loss/" + campaign
            ]
            this_campaign_metric(
                torch.mean(multiq_loss[nontrivial_loss_mask][:, campaign_q_indices])
            )

    # else:
    #     # fallback to logging with question_n
    #     for question_n in range(multiq_loss.shape[1]):
    #         self.log(f'{step_name}/questions/question_{question_n}_loss:0', torch.mean(multiq_loss[:, question_n]), on_epoch=True, on_step=False, sync_dist=True)


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
    multiq_loss = losses.calculate_multiquestion_loss(
        labels, preds, question_index_groups, careful=True
    )
    if sum_over_questions:
        return torch.sum(multiq_loss, axis=1)
    else:
        return multiq_loss


# input_size doesn't matter as long as it's large enough to not be pooled to zero
# channels doesn't matter at all but has to match encoder channels or shape error
def get_encoder_dim(encoder, channels=3):
    device = next(encoder.parameters()).device
    try:
        x = torch.randn(2, channels, 224, 224, device=device)  # BCHW
        return encoder(x).shape[-1]
    except RuntimeError as e:
        if "channels instead" in str(e):
            logging.info(
                "encoder dim search failed on channels, trying with channels=1"
            )
            channels = 1
            x = torch.randn(2, channels, 224, 224, device=device)  # BCHW
            return encoder(x).shape[-1]
        else:
            raise e


def get_pytorch_encoder(
    architecture_name="efficientnet_b0",
    channels=1,
    # use_imagenet_weights=False,
    **timm_kwargs,
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

    # if architecture_name == 'toy':
    #     logging.warning('Using toy encoder')
    #     return ToyEncoder()

    # support older code that didn't specify effnet version
    if architecture_name == "efficientnet":
        logging.warning(
            "efficientnet variant not specified - please set architecture_name=efficientnet_b0 (or similar)"
        )
        architecture_name = "efficientnet_b0"
    return timm.create_model(
        architecture_name, in_chans=channels, num_classes=0, **timm_kwargs
    )


def get_pytorch_dirichlet_head(
    encoder_dim: int, output_dim: int, test_time_dropout: bool, dropout_rate: float
) -> torch.nn.Sequential:
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
        logging.info("Using test-time dropout")
        dropout_layer = custom_layers.PermaDropout
    else:
        logging.info("Not using test-time dropout")
        dropout_layer = torch.nn.Dropout
    modules_to_use.append(dropout_layer(dropout_rate))
    # TODO could optionally add a bottleneck layer here
    modules_to_use.append(
        efficientnet_custom.custom_top_dirichlet(encoder_dim, output_dim)
    )

    return nn.Sequential(*modules_to_use)


def schema_to_campaigns(schema):
    # e.g. [gz2, dr12, ...]
    return [question.text.split("-")[-1] for question in schema.questions]


if __name__ == "__main__":
    encoder = get_pytorch_encoder(channels=1)
    dim = get_encoder_dim(encoder, channels=1)
    print(dim)

    ZoobotTree.load_from_checkpoint
