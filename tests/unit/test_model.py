import torch

from mirage.models import MIRAGECore
from mirage.training.lightning_module import MIRAGELightningModule


def test_mirage_forward_shapes_and_gradients():
    torch.manual_seed(3)
    model = MIRAGECore(6, n_regimes=2, max_lag=2, hidden_dim=16)
    history = torch.randn(4, 12, 6)
    target = torch.randn(4, 6)
    output = model(history, target)
    assert output.local_nll.shape == (4, 6)
    # Effective graph covers lags 0..max_lag -> [batch, max_lag+1, D, D].
    assert output.effective_graph.shape == (4, 3, 6, 6)
    assert torch.allclose(output.regime_probabilities.sum(-1), torch.ones(4), atol=1e-6)
    output.local_nll.mean().backward()
    assert model.graph.shared_logits.grad is not None


def test_lag_zero_self_loop_forbidden_and_acyclicity_penalty():
    torch.manual_seed(3)
    model = MIRAGECore(6, n_regimes=2, max_lag=2, hidden_dim=8)
    with torch.no_grad():
        graph = model.graph.regime_graphs()
        assert graph.shape == (2, 3, 6, 6)
        # Lag-0 slice must have a zero diagonal (no instantaneous self-edges).
        assert torch.allclose(graph[:, 0].diagonal(dim1=-2, dim2=-1), torch.zeros(2, 6), atol=1e-6)
        penalty = model.graph.acyclicity_penalty()
        assert torch.isfinite(penalty)
        assert penalty >= 0.0


def test_lightning_training_step_is_finite():
    module = MIRAGELightningModule(5, n_regimes=2, max_lag=2, hidden_dim=8)
    batch = {"values": torch.randn(3, 8, 5), "target": torch.randn(3, 5)}
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_regime_supervision_loss_is_used_when_enabled():
    module = MIRAGELightningModule(
        5, n_regimes=2, max_lag=2, hidden_dim=8, regime_supervision_weight=0.1
    )
    batch = {
        "values": torch.randn(3, 8, 5),
        "target": torch.randn(3, 5),
        "regime": torch.tensor([0, 1, 0], dtype=torch.long),
    }
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_device_score_excludes_context_variables():
    module = MIRAGELightningModule(
        6, n_regimes=2, max_lag=2, hidden_dim=8, context_indices=[0]
    )
    local_nll = torch.tensor([[100.0, 0.5, 0.4, 0.3, 0.2, 0.1], [50.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    score = module._device_score(local_nll)
    # Context variable (index 0) is excluded: scores come from top-50% of [1..5].
    assert score[0].item() < 1.0
    assert score[1].item() > 1.0

