import pytest
import torch

from spiral.nn.ghost_batch import GhostBatchNorm1D  # adjust to your actual path


def make_gbn(n_features=4, batch_size=8, mini_batch_size=2, momentum=0.1):
    return GhostBatchNorm1D(n_features, batch_size, mini_batch_size, momentum=momentum)


class TestShapesAndBasics:
    def test_training_forward_shape(self):
        gbn = make_gbn()
        gbn.train()
        x = torch.randn(8, 4)
        assert gbn(x).shape == x.shape

    def test_eval_forward_shape_any_batch(self):
        gbn = make_gbn()
        gbn.eval()
        for b in [1, 3, 8, 100]:
            x = torch.randn(b, 4)
            assert gbn(x).shape == x.shape

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            make_gbn(batch_size=10, mini_batch_size=3)  # not divisible

    def test_training_forward_wrong_batch_size_fails(self):
        """Documents current constraint: training forward requires exactly b_l samples."""
        gbn = make_gbn(batch_size=8, mini_batch_size=2)
        gbn.train()
        x = torch.randn(6, 4)
        with pytest.raises(RuntimeError):
            gbn(x)


class TestGradientFlow:
    def test_backward_no_nan(self):
        gbn = make_gbn()
        gbn.train()
        x = torch.randn(8, 4, requires_grad=True)
        gbn(x).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()
        assert not torch.isnan(gbn.scale.grad).any()
        assert not torch.isnan(gbn.shift.grad).any()


class TestNormalizationCorrectness:
    def test_ghost_batches_normalized_independently(self):
        """Each ghost batch should use its OWN mean/std, not the full large batch's."""
        gbn = make_gbn(n_features=1, batch_size=4, mini_batch_size=2, momentum=0.5)
        gbn.train()
        chunk1 = torch.tensor([[0.0], [2.0]])       # mean=1
        chunk2 = torch.tensor([[100.0], [102.0]])   # mean=101, same spread
        x = torch.cat([chunk1, chunk2], dim=0)
        out = gbn(x)
        # identical relative spread per chunk -> normalized output should match
        assert torch.allclose(out[0:2], out[2:4], atol=1e-4)

    def test_running_mean_matches_formula_single_chunk(self):
        """Isolates the multiply-vs-add bug: with n_chunks=1 there's no exponent
        ordering ambiguity, so this pins down the update formula exactly."""
        eta = 0.4
        gbn = GhostBatchNorm1D(n_features=1, batch_size=4, mini_batch_size=4, momentum=eta)
        gbn.train()
        x = torch.full((4, 1), 10.0)
        old_mean = gbn.running_mean.clone()
        gbn(x)
        # n_chunks=1 -> single exponent = 0 -> weight = eta * (1-eta)^0 = eta
        expected = (1 - eta) * old_mean + eta * 10.0
        assert torch.allclose(gbn.running_mean, expected, atol=1e-5)
    
    def test_running_stats_converge_over_many_steps(self):
        """Running stats should settle near the true distribution mean/std,
        not drift or diverge, after many forward passes."""
        torch.manual_seed(0)
        gbn = make_gbn(n_features=1, batch_size=16, mini_batch_size=4, momentum=0.3)
        gbn.train()
        true_mean, true_std = 5.0, 2.0
        for _ in range(200):
            x = torch.randn(16, 1) * true_std + true_mean
            gbn(x)
        assert abs(gbn.running_mean.item() - true_mean) < 0.5, gbn.running_mean.item()
        assert abs(gbn.running_std.item() - true_std) < 0.5, gbn.running_std.item()


class TestWeightingSanity:
    def test_update_weights_sum_to_one(self):
        """Convex-combination weights in update_running should sum to ~1 —
        otherwise running stats drift toward 0 or blow up over training."""
        gbn = make_gbn(n_features=1, batch_size=8, mini_batch_size=2, momentum=0.3)
        n, eta = gbn.n_chunks, gbn.momentum
        decay_weight = (1 - eta) ** n
        chunk_weights = sum(eta * (1 - eta) ** i for i in range(1, n + 1))
        total = decay_weight + chunk_weights
        assert abs(total - 1.0) < 1e-6, (
            f"weights sum to {total:.4f}, not 1 — check the exponent ordering "
            "against a manual unroll of the EMA recurrence"
        )