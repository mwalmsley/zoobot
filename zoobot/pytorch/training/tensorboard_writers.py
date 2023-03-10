from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import torch
from pytorch_lightning.callbacks import Callback

from typing import Any, Callable, Dict, List, Optional


TRAINING_MODE: str = "training"
VALIDATION_MODE: str = "validation"
TEST_MODE: str = "test"
MODES: List[str] = [TRAINING_MODE, VALIDATION_MODE, TEST_MODE]


def prepare_log_metrics(prediction: Tensor,
                        ground_truth: Tensor,
                        criterions: List[Callable],
                        mode: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for criterion in criterions:
        metrics[f'{mode}/{criterion.__name__}'] = criterion(prediction, ground_truth)
    return metrics


class MetricsWriter(Callback):
    def __init__(self,
                 writer: SummaryWriter,
                 criterions: Optional[List[Callable]] = None,
                 mode: str = TRAINING_MODE):
        """
        Args:
            writer (SummaryWriter): Tensorboard SummaryWriter object
            criterions (Optional[List[Callable]], optional): List of metric functions to log. Defaults to None.
            mode (str, optional): Should be "training" or "validation" or "test". Defaults to "training".
        """
        if mode not in MODES:
            raise ValueError("Mode must be one of 'training', 'validation', or 'test'")
        
        self.__writer = writer
        self.__criterions = criterions if criterions else []
        self.__batch_value_sum = {'loss': 0.0}
        for criterion in self.__criterions:
            self.__batch_value_sum[criterion.__name__] = 0.0
        
        self.__batches_counted = 0
        self.__training_epochs_logged = 0
        self.__mode = mode
        
    def __zero_batch_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for k, _ in self.__batch_value_sum.items():
            self.__batch_value_sum[k] = 0.0
        self.__batches_counted = 0
        self.__training_epochs_logged += 1
        
    def __log_epoch_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for k, v in self.__batch_value_sum.items():
            self.__writer.add_scalar(tag=f'{self.__mode}/mean_batch_{k}',
                                    scalar_value=v/self.__batches_counted,
                                    global_step=self.__training_epochs_logged)
            
    def __log_batch_metrics(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int
        ) -> None:
        if type(outputs) == dict:
            loss = outputs['loss']
        elif type(outputs) == torch.Tensor and outputs.shape==():
            loss = outputs.item()
        else:
            # TODO: add warning via logging
            return
        
        self.__batch_value_sum['loss'] += loss
        self.__batches_counted += 1
        
        prediction = pl_module.predict_step(batch, batch_idx)
        for criterion in self.__criterions:
            self.__batch_value_sum[criterion.__name__] += criterion(prediction, batch[1])
        
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if self.__mode == TRAINING_MODE:
            self.__zero_batch_metrics(trainer = trainer,
                                      pl_module = pl_module)
            
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if self.__mode == VALIDATION_MODE:
            self.__zero_batch_metrics(trainer = trainer,
                                      pl_module = pl_module)
    
    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if self.__mode == TEST_MODE:
            self.__zero_batch_metrics(trainer = trainer,
                                      pl_module = pl_module)
        
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if self.__mode == TRAINING_MODE:
            self.__log_epoch_metrics(trainer = trainer,
                                     pl_module = pl_module)
            
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if self.__mode == VALIDATION_MODE:
            self.__log_epoch_metrics(trainer = trainer,
                                     pl_module = pl_module)
            
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if self.__mode == TEST_MODE:
            self.__log_epoch_metrics(trainer = trainer,
                                     pl_module = pl_module)
        
    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int
        ) -> None:
        if self.__mode == TRAINING_MODE:
            self.__log_batch_metrics(trainer = trainer,
                                     pl_module = pl_module,
                                     outputs = outputs,
                                     batch = batch,
                                     batch_idx = batch_idx)
            
    def on_validation_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any,
            batch: Any, batch_idx: int, dataloader_idx: int
        ) -> None:
        if self.__mode == VALIDATION_MODE:
            self.__log_batch_metrics(trainer = trainer,
                                     pl_module = pl_module,
                                     outputs = outputs,
                                     batch = batch,
                                     batch_idx = batch_idx)
            
    def on_test_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any,
            batch: Any, batch_idx: int, dataloader_idx: int
        ) -> None:
        if self.__mode == TEST_MODE:
            self.__log_batch_metrics(trainer = trainer,
                                     pl_module = pl_module,
                                     outputs = outputs,
                                     batch = batch,
                                     batch_idx = batch_idx)
