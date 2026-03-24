from functools import partial

import torch as T
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Accuracy

from src.modules import TransformerClassifier, TransformerConfig


class ChessModel(LightningModule):
    """Transformer-based model for predicting winning probability in chess."""

    def __init__(
        self,
        *,
        config: TransformerConfig,
        optimizer: partial,  # noqa: ARG002
        scheduler: partial,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = TransformerClassifier(config)
        self.accuracy = Accuracy(task="multiclass", num_classes=config.output_dim)

    def _shared_step(self, data: dict, prefix: str) -> T.Tensor:
        board = data["board"]
        result = data["result"]
        logits = self.model(board)
        loss = F.cross_entropy(logits, result)
        self.accuracy(logits, result)
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_accuracy", self.accuracy)
        return loss

    def forward(self, board: T.Tensor) -> T.Tensor:
        logits = self.model(board)
        return F.softmax(logits, dim=-1)

    def training_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "train")

    def validation_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "valid")

    def configure_optimizers(self) -> dict:
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.hparams.optimizer(params)
        sched = self.hparams.scheduler(optimizer=opt, model=self)
        return [opt], [{"scheduler": sched, "interval": "step"}]
