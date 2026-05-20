from torch import nn, optim
import lightning as L
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    EER,
    F1Score,
    Precision,
    Recall,
)

from ml.utils.binfocloss import BinaryFocalLoss


class SpoofDetectorSystem(L.LightningModule):
    """
    LightningModule wrapper for training and evaluating audio spoofing detectors.
    Handles the training loops, metric tracking, and optimization setup.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        alpha: float = 0.5,
        gamma: float = 2.0,
        feature_extractor: nn.Module = None,
    ):
        """
        Initializes the spoofing detector system.

        Args:
            model (nn.Module): Core classifier network backbone.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay coefficient for AdamW.
            alpha (float): Balance parameter for BinaryFocalLoss.
            gamma (float): Focusing parameter for BinaryFocalLoss.
            feature_extractor (nn.Module, optional): Front-end audio processing layer.
        """
        super().__init__()
        # Ignore heavy module instances to prevent redundant hyperparameter logging
        self.save_hyperparameters(ignore=["model", "feature_extractor"])

        self.model = model
        self.feature_extractor = feature_extractor

        self.criterion = BinaryFocalLoss(
            alpha=self.hparams.alpha, gamma=self.hparams.gamma, reduction="mean"
        )

        metrics = MetricCollection(
            {
                "acc": Accuracy(task="binary"),
                "f1": F1Score(task="binary"),
                "auroc": AUROC(task="binary"),
                "precision": Precision(task="binary"),
                "recall": Recall(task="binary"),
                "eer": EER(task="binary"),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass, optionally extracting features before classification.

        Args:
            x (torch.Tensor): Raw audio waveform or pre-extracted features.

        Returns:
            torch.Tensor: Unnormalized network logits.
        """
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)

        return self.model(x)

    def _shared_step(
        self,
        batch: dict,
        metrics: MetricCollection,
        loss_name: str,
        log_on_step: bool = False,
    ) -> torch.Tensor:
        """
        Computes the loss, updates phase-specific metrics, and handles logging.

        Args:
            batch (dict): Dictionary containing 'features' and 'label' tensors.
            metrics (MetricCollection): Metric tracker group for the current phase.
            loss_name (str): Metric logging key name.
            log_on_step (bool): If True, logs step-level loss values.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x = batch["features"]
        y = batch["label"].float()

        logits = self(x)
        loss = self.criterion(logits, y)

        metrics.update(torch.sigmoid(logits), y.long())

        self.log(
            loss_name,
            loss,
            prog_bar=True,
            on_step=log_on_step,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(
            batch, self.train_metrics, loss_name="train/loss", log_on_step=True
        )

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, self.val_metrics, loss_name="val/loss")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, self.test_metrics, loss_name="test/loss")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict:
        """
        Sets up the AdamW optimizer paired with a CosineAnnealingLR scheduler.

        Returns:
            dict: Lightning-compatible optimizer configuration dictionary.
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Handle edge cases where total stepping steps cannot be automatically inferred
        try:
            t_max = self.trainer.estimated_stepping_steps
        except Exception:
            t_max = 5000

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
