from functools import partial

import torch as T
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW, Muon
from torchmetrics import Accuracy

from src.modules import TransformerClassifier, TransformerConfig


class ChessModel(LightningModule):
    """Transformer-based model for predicting winning probability in chess."""

    def __init__(
        self,
        *,
        pytorch_compile: str | None,
        config: TransformerConfig,
        optim: dict,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = TransformerClassifier(config)
        if pytorch_compile is not None:
            self.model = T.compile(self.model, mode=pytorch_compile)
        self.train_acc = Accuracy(task="multiclass", num_classes=config.output_dim)
        self.valid_acc = Accuracy(task="multiclass", num_classes=config.output_dim)
        self.automatic_optimization = False

    def _shared_step(self, data: dict, prefix: str) -> T.Tensor:
        board = data["board"]
        result = data["result"]
        logits = self.model(board)

        loss = F.cross_entropy(logits, result)
        self.log(f"{prefix}_loss", loss)

        acc = getattr(self, f"{prefix}_acc")
        acc(logits, result)
        self.log(f"{prefix}_accuracy", acc)

        return loss

    def forward(self, board: T.Tensor) -> T.Tensor:
        logits = self.model(board)
        return F.softmax(logits, dim=-1)

    def training_step(self, data: dict) -> T.Tensor:
        opt_adamw, opt_muon = self.optimizers()
        sched_adamw, sched_muon = self.lr_schedulers()
        opt_adamw.zero_grad()
        opt_muon.zero_grad()
        loss = self._shared_step(data, "train")
        self.manual_backward(loss)
        self.clip_gradients(opt_adamw, **self.hparams.optim.grad_clip)
        opt_adamw.step()
        opt_muon.step()
        sched_adamw.step()
        sched_muon.step()
        return loss

    def validation_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "valid")

    def configure_optimizers(self) -> dict:
        muon_list = []
        adamw_list = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                use_adamw = any(k in name for k in self.hparams.optim.adamw_params)
                use_adamw |= p.ndim < 2
                if use_adamw:
                    adamw_list.append(p)
                else:
                    muon_list.append(p)
        adamw = AdamW(adamw_list, **self.hparams.optim.adamw_config)
        muon = Muon(muon_list, **self.hparams.optim.muon_config)
        sched_adamw = self.hparams.optim.scheduler(optimizer=adamw, model=self)
        sched_muon = self.hparams.optim.scheduler(optimizer=muon, model=self)
        return [adamw, muon], [sched_adamw, sched_muon]
