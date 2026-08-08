from __future__ import annotations

import torch
from torch import nn


class DualGraphParameterization(nn.Module):
    """Shared graph plus sparse regime-specific deltas, over lags 0..max_lag.

    Lag 0 is the instantaneous slice of the graph (simultaneous effects, e.g.
    controller feedback at the same sampling instant). It is subject to a
    NOTEARS-style acyclicity penalty; lag-0 self-loops are forbidden by the mask.
    """

    def __init__(
        self,
        n_variables: int,
        n_regimes: int,
        max_lag: int,
        allowed_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1")
        self.n_variables = n_variables
        self.n_regimes = n_regimes
        self.max_lag = max_lag
        raw_mask = (
            torch.ones(n_variables, n_variables)
            if allowed_mask is None
            else allowed_mask.float()
        )
        # Expand the 2-D role mask across lags; lag-0 self-loops are forbidden
        # (simultaneous self-causation is not identifiable in a next-step model).
        expanded = raw_mask.unsqueeze(0).repeat(max_lag + 1, 1, 1).clone()
        expanded[0].fill_diagonal_(0.0)
        self.register_buffer("allowed_mask", expanded)
        self.shared_logits = nn.Parameter(torch.empty(max_lag + 1, n_variables, n_variables))
        self.delta_logits = nn.Parameter(
            torch.zeros(n_regimes, max_lag + 1, n_variables, n_variables)
        )
        nn.init.normal_(self.shared_logits[1:], mean=0.0, std=0.04)
        # No prior mass on instantaneous edges: start lag 0 at zero.
        nn.init.zeros_(self.shared_logits[0])

    def shared_graph(self) -> torch.Tensor:
        return torch.tanh(self.shared_logits) * self.allowed_mask

    def regime_graphs(self) -> torch.Tensor:
        shared = self.shared_graph().unsqueeze(0)
        deltas = torch.tanh(self.delta_logits) * self.allowed_mask
        return torch.clamp(shared + deltas, -1.0, 1.0)

    def effective_graph(self, regime_probabilities: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bk,klst->blst", regime_probabilities, self.regime_graphs())

    def regularization(
        self,
        prior_expected: torch.Tensor | None = None,
        prior_sign: torch.Tensor | None = None,
        prior_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        shared = self.shared_graph()
        regime = self.regime_graphs()
        # 稀疏/δ 正则必须用 sum()：mean() 会把梯度除以全图元素数
        # （D*D*(L+1)，12 变量 4 lag 即 576 倍），导致 L1 梯度 ~1e-5 完全
        # 压不动权重，图为学成稠密（SHD 爆表）。sum() 下梯度 = λ/参数。
        losses = {
            "sparsity": shared.abs().sum(),
            "delta": (regime - shared.unsqueeze(0)).abs().sum(),
        }
        # Priors describe lagged edges (lag >= 1); instantaneous slice is not
        # subject to the expected-edge prior.
        aggregated = regime.abs()[:, 1:].amax(dim=(0, 1))
        if prior_expected is None:
            losses["prior"] = shared.new_zeros(())
        else:
            confidence = torch.ones_like(prior_expected) if prior_confidence is None else prior_confidence
            expected_loss = ((1 - aggregated) * prior_expected * confidence).sum() / (
                (prior_expected * confidence).sum() + 1e-8
            )
            if prior_sign is None:
                sign_loss = shared.new_zeros(())
            else:
                # Average only over regimes, keeping the lag axis: [K, L+1, D, D]
                # -> [L+1, D, D]; drop the lag-0 slice for the sign prior.
                signed = regime.mean(dim=0)[1:]
                sign_loss = (
                    torch.relu(-signed * prior_sign) * (prior_sign != 0) * confidence
                ).sum() / (((prior_sign != 0) * confidence).sum() + 1e-8)
            losses["prior"] = expected_loss + sign_loss
        losses["acyclicity"] = self.acyclicity_penalty()
        return losses

    def acyclicity_penalty(self) -> torch.Tensor:
        """NOTEARS-style penalty on the instantaneous (lag-0) slice.

        h(A) = trace(expm(A∘A)) - d  is 0 iff the lag-0 graph is a DAG and
        strictly positive otherwise (Zheng et al., 2018).
        """
        instantaneous = torch.tanh(self.shared_logits[0]) * self.allowed_mask[0]
        product = instantaneous * instantaneous
        exponent = torch.linalg.matrix_exp(product)
        return exponent.trace() - product.shape[0]
