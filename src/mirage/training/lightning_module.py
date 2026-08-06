from __future__ import annotations

from typing import Any

import lightning as L
import torch

from mirage.models import MIRAGECore
from mirage.training.alternating import select_alternating_phase
from mirage.training.losses import regime_supervision_loss, total_loss


class MIRAGELightningModule(L.LightningModule):
    def __init__(
        self,
        n_variables: int,
        n_regimes: int = 3,
        max_lag: int = 3,
        hidden_dim: int = 64,
        regime_temperature: float = 1.0,
        student_t_df: float = 5.0,
        prior_weight: float = 0.05,
        sparsity_weight: float = 0.001,
        delta_weight: float = 0.001,
        acyclicity_weight: float = 0.0,
        balance_weight: float = 0.01,
        regime_supervision_weight: float = 0.0,
        alternating: bool = False,
        score_topq: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        allowed_mask: torch.Tensor | None = None,
        prior_expected: torch.Tensor | None = None,
        prior_sign: torch.Tensor | None = None,
        prior_confidence: torch.Tensor | None = None,
        context_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["allowed_mask", "prior_expected", "prior_sign", "prior_confidence"]
        )
        self.model = MIRAGECore(
            n_variables=n_variables,
            n_regimes=n_regimes,
            max_lag=max_lag,
            hidden_dim=hidden_dim,
            regime_temperature=regime_temperature,
            student_t_df=student_t_df,
            allowed_mask=allowed_mask,
        )
        size = n_variables
        self.register_buffer(
            "prior_expected", torch.zeros(size, size) if prior_expected is None else prior_expected.float()
        )
        self.register_buffer(
            "prior_sign", torch.zeros(size, size) if prior_sign is None else prior_sign.float()
        )
        self.register_buffer(
            "prior_confidence",
            torch.zeros(size, size) if prior_confidence is None else prior_confidence.float(),
        )
        # Device-level score should aggregate only non-context variables (paper:
        # top-q mean over non-context nodes), so context variables with strong
        # exogenous driving terms cannot dominate the anomaly score.
        non_context = torch.ones(size, dtype=torch.bool)
        if context_indices:
            non_context[torch.as_tensor(context_indices, dtype=torch.long)] = False
        self.register_buffer("non_context_mask", non_context)

    def forward(self, history: torch.Tensor, target: torch.Tensor):
        return self.model(history, target)

    def _device_score(self, local_nll: torch.Tensor) -> torch.Tensor:
        scored = local_nll[:, self.non_context_mask]
        if scored.numel() == 0:
            scored = local_nll
        k = max(1, min(scored.shape[1], int(self.hparams.score_topq * scored.shape[1])))
        return torch.topk(scored, k, dim=-1).values.mean(dim=-1)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        output = self(batch["values"], batch["target"])
        regularization = self.model.graph.regularization(
            self.prior_expected, self.prior_sign, self.prior_confidence
        )
        regime_loss = None
        if self.hparams.regime_supervision_weight > 0 and "regime" in batch:
            regime_loss = regime_supervision_loss(output.regime_logits, batch["regime"])
        loss, components = total_loss(
            output.local_nll,
            regularization,
            output.regime_probabilities,
            self.hparams.prior_weight,
            self.hparams.sparsity_weight,
            self.hparams.delta_weight,
            self.hparams.acyclicity_weight,
            self.hparams.balance_weight,
            regime_loss=regime_loss,
            regime_supervision_weight=float(self.hparams.regime_supervision_weight),
        )
        self.log(f"{stage}/loss", loss, prog_bar=stage != "train", on_epoch=True, on_step=False)
        for name, value in components.items():
            self.log(f"{stage}/{name}", value, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.hparams.alternating:
            select_alternating_phase(
                self.model,
                self.current_epoch,
                graph_epochs=2,
                mechanism_epochs=4,
            )
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "validation")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        output = self(batch["values"], batch["target"])
        score = self._device_score(output.local_nll)
        self.log("test/score_mean", score.mean(), on_epoch=True, on_step=False)
        result: dict[str, torch.Tensor] = {"score": score, "local_score": output.local_nll}
        if "label" in batch:
            result["label"] = batch["label"]
        return result

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        output = self(batch["values"], batch["target"])
        # Deliberately light payload: exporting per-sample effective graphs /
        # distribution parameters was an O(B*L*D*D) memory bomb under large configs.
        return {
            "index": batch["index"],
            "score": self._device_score(output.local_nll),
            "local_score": output.local_nll,
            "regime_probability": output.regime_probabilities,
            **({"label": batch["label"]} if "label" in batch else {}),
        }

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "validation/loss"},
        }
