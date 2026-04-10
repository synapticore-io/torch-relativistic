# 🌌 torch-relativistic

<div align="center">

<!-- Badges -->
[![PyPI version](https://badge.fury.io/py/torch-relativistic.svg)](https://badge.fury.io/py/torch-relativistic)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/synapticore-io/torch-relativistic/actions/workflows/tests.yml/badge.svg)](https://github.com/synapticore-io/torch-relativistic/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- Logo/Header -->
<h3>🚀 A PyTorch extension implementing neural networks inspired by relativistic physics</h3>

*Physics-inspired neural network layers with measurable results on standard benchmarks* ⚡

</div>

<p align="center">
  <img src="examples/terrell_penrose_demo.png" alt="Terrell-Penrose effect: a cube at v=0, 0.5c, 0.8c, 0.95c" width="100%">
  <br>
  <em>The Terrell-Penrose effect reproduced computationally: a cube appears <b>rotated</b>, not contracted,
  at relativistic speeds. Same physics as <a href="https://doi.org/10.1038/s42005-025-02003-6">Schattschneider et al. (2025)</a>.</em>
</p>

---

## 🌟 Overview

**torch-relativistic** provides neural network modules that incorporate concepts from **special relativity** into
machine learning. The core idea is that the **Terrell-Penrose effect** — where rapidly moving objects appear rotated
rather than contracted — can serve as a physics-motivated inductive bias for learned feature transformations.
Across the standard Planetoid benchmarks (Cora, CiteSeer, PubMed),
`RelativisticGraphConv` with `normalize=True` is consistently competitive
with stock `GCNConv` — see the [full results](benchmarks/results/).

### 🎯 Key Features

- 🧠 **Relativistic Graph Neural Networks (GNNs)** - Process graphs with relativistic information propagation
- ⚡ **Relativistic Spiking Neural Networks (SNNs)** - Time dilation effects in spiking neurons
- 🎭 **Relativistic Attention Mechanisms** - Multi-reference frame attention heads
- 🌀 **Relativistic Transformations** - Lorentz boosts and Terrell-Penrose transforms
- 🔬 **Physics-Inspired Architecture** - Grounded in real relativistic physics

---

## 🤔 Why Relativistic? The Hypothesis

The first question everyone asks when they see a module called
`RelativisticGraphConv` is: *why would a neural network care about special
relativity?* There is no speed of light in a social graph, a molecule, or a
batch of token embeddings.

The honest answer: **this is an inductive-bias experiment, not a physics
simulation.** The bet is that three properties of relativistic transformations
transfer usefully into *learned* feature spaces even when the underlying
"space" has nothing to do with spacetime.

### 1. Velocity as a learnable soft-scale knob

In a classical graph convolution every neighbour contributes equally (up to
edge weights). A learnable `velocity ∈ [0, c)` lets the network smoothly
interpolate between

- a **low-velocity regime** (`γ ≈ 1`), where the layer degenerates to a
  standard graph conv, and
- a **high-velocity regime** (`γ → ∞`), where information from far-away
  nodes is effectively "time-dilated away".

That is a physically-motivated, continuous **locality prior** the optimizer
can tune *per layer*, without having to commit to a fixed receptive field up
front. An ablation over `max_velocity` should show: at `v=0` the layer tracks
the baseline, at moderate `v` something happens, at `v → c` the gradients
explode (a known failure mode that's easy to detect and to clamp against).

### 2. Lorentz boosts as a feature-space symmetry group

Group-equivariant neural networks (rotation-equivariant CNNs, SE(3)-equivariant
molecular models, E(3)-NNs) have repeatedly shown that **enforcing a symmetry
of the domain as an architectural invariant acts as an extremely strong
regularizer**. The Lorentz group is a well-studied continuous Lie group with
a known representation theory — applying it as an equivariance constraint is
concretely doable (`LorentzBoost` is a 4×4 matrix multiply). Whether it
*helps* in domains that do not have an intrinsic 4-vector structure is
exactly the open question.

### 3. Terrell-Penrose as implicit augmentation

Rotation-based data augmentation routinely improves CNN performance. The
**Terrell-Penrose effect** (Terrell 1959, Penrose 1959) is the counter-intuitive
observation that a rapidly moving extended object appears optically *rotated*
rather than Lorentz-contracted, because photons from the far side of the
object were emitted earlier in time. After 66 years as a purely theoretical
prediction, the effect was [first observed in the lab in May 2025](https://doi.org/10.1038/s42005-025-02003-6)
by Schattschneider et al. at TU Wien using high-speed cameras and laser
pulses — the direct inspiration for this library. Baked into a
`TerrellPenroseTransform` layer, the rotation becomes a per-forward-pass
*implicit* augmentation, coupling together what classical augmentation
pipelines apply independently across training steps.

### Status of these claims

These are **hypotheses**, not established empirical results. The
[`benchmarks/`](benchmarks/) directory is where they are tested. Three
outcomes are all worth reporting:

- **Null result** — `max_velocity=0` and `max_velocity=0.9` perform
  indistinguishably → the bias is neutral, neither helpful nor hurtful, and
  the extension becomes a cleaner way to explore group-equivariant
  architectures.
- **Positive result** — the relativistic variant beats the baseline on a
  specific task or provides transfer benefits → publish as a paper, update
  the README with numbers.
- **Negative result** — high velocities consistently hurt → the mechanism is
  incompatible with how standard GNN / SNN / attention tasks work, which is
  itself informative for future physics-inspired architectures.

This repository ships the architecture and the tooling. It does not claim
empirical superiority that has not yet been measured.

### 📊 Benchmarks (Planetoid node classification, 3 seeds)

<p align="center">
  <img src="benchmarks/results/comparison.png" alt="Benchmark comparison across Cora, CiteSeer, PubMed" width="80%">
</p>

| Config | Cora | CiteSeer | PubMed |
|---|---|---|---|
| `GCNConv` | 81.2 ± 0.7% | 68.8 ± 0.7% | **78.5 ± 0.5%** |
| `SAGEConv` | 80.7 ± 0.5% | **69.5 ± 0.2%** | 76.7 ± 0.3% |
| **`Rel. norm v=0`** | **81.7 ± 0.1%** | 69.2 ± 0.7% | **78.6 ± 0.4%** |
| `Rel. norm v=0.3` | 80.4 ± 1.1% | 68.8 ± 0.9% | 77.9 ± 1.0% |

`RelativisticGraphConv` with `normalize=True` at `max_velocity=0` is
**consistently competitive with standard GCNConv** across all three
Planetoid datasets, with a small edge on Cora (+0.5pp with the lowest
variance of any config) and near-parity on PubMed. On CiteSeer all
methods are within noise of each other.

The gain comes from the architecture's in-message linear transform and
position-aware weighting — not from the velocity parameter, which
provides no benefit on these homophilic citation graphs. The velocity
knob is expected to become more relevant on tasks with intrinsic spatial
structure (point clouds, molecular graphs).

Without `normalize=True`, performance drops to ~75% on Cora, confirming
that fair comparisons require matching the normalization scheme.

See [`benchmarks/results/`](benchmarks/results/) for per-dataset tables,
raw JSON results, and the full interpretation.

**Key takeaway for users**: use `normalize=True` with `max_velocity=0`
as the default on homophilic graphs.

---

## 📦 Installation

### Quick Install

```bash
pip install torch-relativistic
```

### Development Install

```bash
git clone https://github.com/synapticore-io/torch-relativistic.git
cd torch-relativistic
pip install -e .
```

### Requirements

- 🐍 Python ≥ 3.11
- 🔥 PyTorch == 2.8.0 (resolved from the `pytorch-cuda` index for CUDA 12.8)
- 📊 PyTorch Geometric ≥ 2.7.0
- 🔢 NumPy ≥ 2.0.0
- 🐼 Pandas ≥ 3.0.0

---

## 🚀 Quick Start

```python
import torch
from torch_relativistic import RelativisticGraphConv

# Create a relativistic GNN layer
conv = RelativisticGraphConv(16, 32, max_relative_velocity=0.8)
x = torch.randn(10, 16)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

# Process with relativistic effects
output = conv(x, edge_index)  # Shape: [10, 32]
```

---

## 📚 Components

### 🌐 Relativistic Graph Neural Networks

GNN modules that process information as if affected by relativistic phenomena:

```python
import torch
from torch_relativistic.gnn import RelativisticGraphConv, MultiObserverGNN

# Create a simple graph
num_nodes = 10
feature_dim = 16
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                           [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
node_features = torch.randn(num_nodes, feature_dim)

# Create a relativistic GNN layer
conv = RelativisticGraphConv(
    in_channels=feature_dim,
    out_channels=32,
    max_relative_velocity=0.8
)

# Process the graph
output_features = conv(node_features, edge_index)
print(f"Output shape: {output_features.shape}")  # [10, 32]

# Multi-observer GNN processes the graph from multiple relativistic perspectives
multi_observer_gnn = MultiObserverGNN(
    feature_dim=feature_dim,
    hidden_dim=32,
    output_dim=8,
    num_observers=4
)

output = multi_observer_gnn(node_features, edge_index)
print(f"Multi-observer output shape: {output.shape}")  # [10, 8]
```

### ⚡ Relativistic Spiking Neural Networks

SNN components that incorporate relativistic time dilation:

```python
import torch
from torch_relativistic.snn import RelativisticLIFNeuron, TerrellPenroseSNN

# Create input spikes (batch_size=32, input_size=10)
input_spikes = torch.bernoulli(torch.ones(32, 10) * 0.3)

# Create a relativistic LIF neuron
neuron = RelativisticLIFNeuron(
    input_size=10,
    threshold=1.0,
    beta=0.9
)

# Initialize neuron state
initial_state = neuron.init_state(batch_size=32)

# Process input spikes
output_spikes, new_state = neuron(input_spikes, initial_state)
print(f"Output spikes shape: {output_spikes.shape}")  # [32]

# Create a complete SNN
snn = TerrellPenroseSNN(
    input_size=10,
    hidden_size=20,
    output_size=5,
    simulation_steps=100
)

# Process input
output = snn(input_spikes)
print(f"SNN output shape: {output.shape}")  # [32, 5]

# Get spike history for visualization
spike_history = snn.get_spike_history(input_spikes)
print(f"Hidden spike history shape: {spike_history['hidden_spikes'].shape}")  # [32, 100, 20]
```

### 🎭 Relativistic Attention Mechanism

Attention where different heads operate in different reference frames:

```python
import torch
from torch_relativistic.attention import RelativisticSelfAttention

# Create input sequence (batch_size=16, seq_len=24, feature_dim=64)
seq = torch.randn(16, 24, 64)

# Create relativistic self-attention module
attention = RelativisticSelfAttention(
    hidden_dim=64,
    num_heads=8,
    dropout=0.1,
    max_velocity=0.9
)

# Optional: 3D positions for spacetime-aware attention.
# Shape [batch, seq_len, D]: internally reduced to a scalar per token
# via L2 norm for rotary position embedding frequencies.
positions = torch.randn(16, 24, 3)

# Process sequence
output = attention(seq, positions=positions)
print(f"Output shape: {output.shape}")  # [16, 24, 64]
```

### 🌀 Relativistic Transformations

Apply transformations inspired by special relativity to feature vectors:

```python
import torch
from torch_relativistic.transforms import TerrellPenroseTransform, LorentzBoost

# Create feature vectors (batch_size=8, feature_dim=64)
features = torch.randn(8, 64)

# Apply Terrell-Penrose inspired transformation
transform = TerrellPenroseTransform(
    feature_dim=64,
    max_velocity=0.9,
    mode="rotation"
)

transformed = transform(features)
print(f"Transformed shape: {transformed.shape}")  # [8, 64]

# For spacetime features (batch_size=8, feature_dim=8 including 4D spacetime)
spacetime_features = torch.randn(8, 8)

# Apply Lorentz boost
boost = LorentzBoost(
    feature_dim=8,
    time_dim=0,  # First dimension is time
    max_velocity=0.8
)

boosted = boost(spacetime_features)
print(f"Boosted shape: {boosted.shape}")  # [8, 8]
```

---

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=torch_relativistic
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🛠️ Development Setup

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/synapticore-io/torch-relativistic.git`
3. **Install** in development mode: `pip install -e ".[dev]"`
4. **Create** a feature branch: `git checkout -b feature/amazing-feature`
5. **Make** your changes and add tests
6. **Run** tests: `pytest tests/`
7. **Submit** a pull request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📖 How to Cite

If you use torch-relativistic in research that leads to a publication,
please cite the software. GitHub shows a "Cite this repository" button
in the right sidebar (generated from `CITATION.cff`).

### BibTeX

```bibtex
@software{bethge_torch_relativistic,
  author       = {Bethge, Björn},
  title        = {{torch-relativistic: Physics-inspired PyTorch layers
                   based on the Terrell-Penrose effect}},
  year         = {2026},
  version      = {0.2.0},
  url          = {https://github.com/synapticore-io/torch-relativistic}
}
```

If your work also relates to the experimental observation of the
Terrell-Penrose effect, please cite the original paper:

```bibtex
@article{schattschneider2025snapshot,
  author  = {Schattschneider, Peter and others},
  title   = {A Snapshot of Relativistic Motion: Visualizing the
             Terrell-Penrose Effect},
  journal = {Communications Physics},
  year    = {2025},
  doi     = {10.1038/s42005-025-02003-6}
}
```

---

## 🙏 Acknowledgments

- 🔬 **Directly inspired by** the [first experimental observation of the
  Terrell-Penrose effect](https://doi.org/10.1038/s42005-025-02003-6) by
  Schattschneider et al. at TU Wien (Communications Physics, May 2025) —
  66 years after its theoretical prediction by Terrell and Penrose in 1959
- 🌌 Grounded in Einstein's **Special Theory of Relativity**
- 🔥 Powered by **PyTorch** and **PyTorch Geometric**
- ⚡ Thanks to the open-source ML and physics communities

---

## 📞 Contact & Links

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/synapticore-io/torch-relativistic)
[![PyPI](https://img.shields.io/badge/PyPI-3775A9?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/torch-relativistic/)

**Made with ❤️ and ⚛️ physics**

</div>

---

<div align="center">
<sub>Built with 🔥 PyTorch • Inspired by 🌌 Einstein • Powered by ⚛️ Physics</sub>
</div>
