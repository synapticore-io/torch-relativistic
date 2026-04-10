"""
Forward/backward/shape tests for all public neural modules.

These are mechanical regression checks: given inputs of the shapes the
README advertises, every module must

  1. return outputs of the expected shape and dtype,
  2. propagate gradients back to its learnable parameters without NaN/Inf,
  3. be deterministic under a fixed seed,
  4. survive a state_dict save / load round-trip.

The tests are intentionally module-agnostic about the *value* of the
output — proving that a relativistic feature transform is "physically
correct" is a different, much larger exercise (see benchmarks/). Here we
lock down the API contract so future refactors don't silently break it.
"""

from __future__ import annotations

import pytest
import torch

from torch_relativistic.attention import RelativisticSelfAttention
from torch_relativistic.gnn import MultiObserverGNN, RelativisticGraphConv
from torch_relativistic.snn import RelativisticLIFNeuron, TerrellPenroseSNN
from torch_relativistic.transforms import LorentzBoost, TerrellPenroseTransform


def _assert_finite_grads(module: torch.nn.Module) -> None:
    """Every learnable parameter must have a finite gradient after .backward()."""
    total = 0
    for name, p in module.named_parameters():
        if p.requires_grad and p.grad is not None:
            total += 1
            assert torch.all(
                torch.isfinite(p.grad)
            ), f"non-finite gradient on parameter {name}: {p.grad}"
    assert total > 0, f"no parameters received gradients on {type(module).__name__}"


# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------


class TestRelativisticGraphConv:
    @pytest.fixture
    def conv(self):
        torch.manual_seed(0)
        return RelativisticGraphConv(
            in_channels=8, out_channels=16, max_relative_velocity=0.8
        )

    @pytest.fixture
    def sample(self):
        # 5-node cycle graph
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 4]], dtype=torch.long
        )
        x = torch.randn(5, 8)
        return x, edge_index

    def test_forward_shape(self, conv, sample):
        x, edge_index = sample
        out = conv(x, edge_index)
        assert out.shape == (5, 16)
        assert out.dtype == x.dtype

    def test_backward_produces_finite_grads(self, conv, sample):
        x, edge_index = sample
        out = conv(x, edge_index)
        out.sum().backward()
        _assert_finite_grads(conv)

    def test_forward_deterministic_with_seed(self, sample):
        x, edge_index = sample
        torch.manual_seed(42)
        conv1 = RelativisticGraphConv(8, 16, max_relative_velocity=0.7)
        out1 = conv1(x, edge_index)

        torch.manual_seed(42)
        conv2 = RelativisticGraphConv(8, 16, max_relative_velocity=0.7)
        out2 = conv2(x, edge_index)

        assert torch.allclose(out1, out2)

    def test_state_dict_roundtrip(self, conv, sample):
        x, edge_index = sample
        out_before = conv(x, edge_index)

        state = conv.state_dict()
        new_conv = RelativisticGraphConv(8, 16, max_relative_velocity=0.8)
        new_conv.load_state_dict(state)
        out_after = new_conv(x, edge_index)

        assert torch.allclose(out_before, out_after)

    def test_normalize_changes_output(self, sample):
        """With normalize=True, GCN-style D^(-1/2) A D^(-1/2) is applied."""
        x, edge_index = sample
        torch.manual_seed(0)
        conv_no_norm = RelativisticGraphConv(
            8, 16, max_relative_velocity=0.5, normalize=False
        )
        torch.manual_seed(0)
        conv_norm = RelativisticGraphConv(
            8, 16, max_relative_velocity=0.5, normalize=True
        )
        out_no = conv_no_norm(x, edge_index)
        out_yes = conv_norm(x, edge_index)
        # Normalization adds self-loops and rescales — output shapes match
        # but values must differ (because normalize changes the edge set)
        assert out_no.shape[1] == out_yes.shape[1] == 16
        # The two outputs should NOT be identical
        assert not torch.allclose(out_no, out_yes, atol=1e-4)

    def test_normalize_backward_finite(self, sample):
        x, edge_index = sample
        conv = RelativisticGraphConv(8, 16, max_relative_velocity=0.5, normalize=True)
        out = conv(x, edge_index)
        out.sum().backward()
        _assert_finite_grads(conv)


class TestMultiObserverGNN:
    def test_forward_shape(self):
        torch.manual_seed(0)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        x = torch.randn(4, 16)
        model = MultiObserverGNN(
            feature_dim=16, hidden_dim=32, output_dim=8, num_observers=4
        )
        out = model(x, edge_index)
        assert out.shape == (4, 8)

    def test_backward_produces_finite_grads(self):
        torch.manual_seed(0)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        x = torch.randn(4, 16)
        model = MultiObserverGNN(
            feature_dim=16, hidden_dim=32, output_dim=8, num_observers=4
        )
        out = model(x, edge_index)
        out.sum().backward()
        _assert_finite_grads(model)


# ---------------------------------------------------------------------------
# SNN
# ---------------------------------------------------------------------------


class TestRelativisticLIFNeuron:
    def test_forward_shape_and_state(self):
        torch.manual_seed(0)
        batch_size, input_size = 8, 10
        neuron = RelativisticLIFNeuron(input_size=input_size, threshold=1.0, beta=0.9)
        state = neuron.init_state(batch_size=batch_size)
        spikes_in = torch.bernoulli(torch.ones(batch_size, input_size) * 0.3)

        spikes_out, new_state = neuron(spikes_in, state)

        # spikes_out is a single spike per sample in the batch
        assert spikes_out.shape[0] == batch_size
        # state should be the same structure
        assert type(new_state) is type(state)

    def test_backward_through_surrogate(self):
        torch.manual_seed(0)
        batch_size, input_size = 4, 10
        neuron = RelativisticLIFNeuron(input_size=input_size, threshold=1.0, beta=0.9)
        state = neuron.init_state(batch_size=batch_size)
        spikes_in = torch.rand(batch_size, input_size, requires_grad=True)

        spikes_out, _ = neuron(spikes_in, state)
        spikes_out.sum().backward()

        _assert_finite_grads(neuron)

    def test_forward_produces_binary_spikes(self):
        """Sanity check: LIF output spikes are in {0, 1}."""
        torch.manual_seed(0)
        batch_size, input_size = 4, 10
        neuron = RelativisticLIFNeuron(input_size=input_size, threshold=1.0, beta=0.9)
        state = neuron.init_state(batch_size=batch_size)
        spikes_in = torch.ones(batch_size, input_size) * 2.0  # strong input
        spikes_out, _ = neuron(spikes_in, state)
        unique_values = set(spikes_out.unique().tolist())
        assert unique_values <= {
            0.0,
            1.0,
        }, f"expected binary spikes, got {unique_values}"


class TestTerrellPenroseSNN:
    def test_forward_shape(self):
        torch.manual_seed(0)
        batch_size, input_size, output_size = 4, 10, 5
        snn = TerrellPenroseSNN(
            input_size=input_size,
            hidden_size=20,
            output_size=output_size,
            simulation_steps=25,
        )
        spikes_in = torch.bernoulli(torch.ones(batch_size, input_size) * 0.3)
        out = snn(spikes_in)
        assert out.shape == (batch_size, output_size)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class TestRelativisticSelfAttention:
    def test_forward_shape(self):
        torch.manual_seed(0)
        batch, seq, hidden = 4, 12, 32
        attn = RelativisticSelfAttention(
            hidden_dim=hidden, num_heads=4, dropout=0.0, max_velocity=0.9
        )
        seq_in = torch.randn(batch, seq, hidden)
        out = attn(seq_in)
        assert out.shape == (batch, seq, hidden)

    def test_backward_produces_finite_grads(self):
        torch.manual_seed(0)
        batch, seq, hidden = 2, 8, 16
        attn = RelativisticSelfAttention(
            hidden_dim=hidden, num_heads=4, dropout=0.0, max_velocity=0.9
        )
        seq_in = torch.randn(batch, seq, hidden, requires_grad=True)
        out = attn(seq_in)
        out.sum().backward()
        _assert_finite_grads(attn)

    def test_forward_with_1d_positions(self):
        """Positions as [batch, seq] — scalar time indices."""
        torch.manual_seed(0)
        batch, seq, hidden = 2, 6, 16
        attn = RelativisticSelfAttention(
            hidden_dim=hidden, num_heads=4, dropout=0.0, max_velocity=0.9
        )
        seq_in = torch.randn(batch, seq, hidden)
        pos = torch.arange(seq).unsqueeze(0).expand(batch, seq).float()
        out = attn(seq_in, positions=pos)
        assert out.shape == (batch, seq, hidden)

    def test_forward_with_3d_positions(self):
        """Positions as [batch, seq, 3] — 3D spatial positions, reduced via L2 norm."""
        torch.manual_seed(0)
        batch, seq, hidden = 2, 6, 16
        attn = RelativisticSelfAttention(
            hidden_dim=hidden, num_heads=4, dropout=0.0, max_velocity=0.9
        )
        seq_in = torch.randn(batch, seq, hidden)
        pos_3d = torch.randn(batch, seq, 3)
        out = attn(seq_in, positions=pos_3d)
        assert out.shape == (batch, seq, hidden)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTerrellPenroseTransform:
    def test_forward_shape(self):
        torch.manual_seed(0)
        transform = TerrellPenroseTransform(
            feature_dim=64, max_velocity=0.9, mode="rotation"
        )
        x = torch.randn(8, 64)
        out = transform(x)
        assert out.shape == (8, 64)
        assert out.dtype == x.dtype

    def test_backward_produces_finite_grads(self):
        torch.manual_seed(0)
        transform = TerrellPenroseTransform(
            feature_dim=64, max_velocity=0.9, mode="rotation"
        )
        x = torch.randn(8, 64, requires_grad=True)
        out = transform(x)
        out.sum().backward()
        _assert_finite_grads(transform)


class TestLorentzBoost:
    def test_forward_shape(self):
        torch.manual_seed(0)
        boost = LorentzBoost(feature_dim=8, time_dim=0, max_velocity=0.8)
        x = torch.randn(8, 8)
        out = boost(x)
        assert out.shape == (8, 8)

    def test_backward_produces_finite_grads(self):
        torch.manual_seed(0)
        boost = LorentzBoost(feature_dim=8, time_dim=0, max_velocity=0.8)
        x = torch.randn(8, 8, requires_grad=True)
        out = boost(x)
        out.sum().backward()
        _assert_finite_grads(boost)

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        boost = LorentzBoost(feature_dim=8, time_dim=0, max_velocity=0.8)
        x = torch.randn(4, 8)
        out_before = boost(x)

        state = boost.state_dict()
        new_boost = LorentzBoost(feature_dim=8, time_dim=0, max_velocity=0.8)
        new_boost.load_state_dict(state)
        out_after = new_boost(x)

        assert torch.allclose(out_before, out_after)
