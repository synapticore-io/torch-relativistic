"""
Cora node-classification benchmark: baseline GCN vs RelativisticGraphConv.

What this measures
------------------
On the Cora citation graph (2708 nodes, 7 classes, 1433 features) we train
a small two-layer graph neural network two ways:

  - baseline  : stock `torch_geometric.nn.GCNConv`
  - relativistic : `torch_relativistic.gnn.RelativisticGraphConv`

For the relativistic variant we sweep `max_relative_velocity` over
{0.0, 0.3, 0.6, 0.9}. At v=0 the layer should be a strong functional analogue
of the baseline (gamma ≈ 1, no relativistic message weighting). At higher v
the per-message transformation gets progressively more aggressive, and we
want to see whether that helps, hurts, or is a no-op.

Each config is run across multiple random seeds and we report mean ± std of
test accuracy at the epoch of best validation accuracy.

Usage
-----
    python benchmarks/cora_gcn_vs_relativistic.py

Results are written to:
    benchmarks/results/cora.json        (raw numbers, one row per seed)
    benchmarks/results/cora.md          (summary table suitable for README)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from torch_relativistic.gnn import RelativisticGraphConv


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BaselineGCN(nn.Module):
    """Stock 2-layer GCNConv network."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class RelativisticGCN(nn.Module):
    """Same architecture with RelativisticGraphConv layers."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        max_velocity: float,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = RelativisticGraphConv(
            in_channels, hidden, max_relative_velocity=max_velocity
        )
        self.conv2 = RelativisticGraphConv(
            hidden, out_channels, max_relative_velocity=max_velocity
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    config: str
    seed: int
    best_val_acc: float
    test_acc_at_best_val: float
    train_time_s: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(
    model: nn.Module,
    data,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> tuple[float, float]:
    """Train `model` on `data` and return (best_val_acc, test_acc_at_best_val)."""
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 0.0
    test_at_best_val = 0.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
            val_acc = val_correct / int(data.val_mask.sum().item())
            if val_acc > best_val:
                best_val = val_acc
                test_correct = (
                    (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
                )
                test_at_best_val = test_correct / int(data.test_mask.sum().item())

    return best_val, test_at_best_val


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    hidden: int = 64
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    relativistic_velocities: list[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.6, 0.9]
    )


def run(cfg: BenchmarkConfig, dataset_root: Path, device: torch.device) -> list[RunResult]:
    dataset = Planetoid(root=str(dataset_root), name="Cora")
    data = dataset[0]
    in_ch = dataset.num_node_features
    out_ch = dataset.num_classes

    results: list[RunResult] = []

    # Baseline
    for seed in cfg.seeds:
        set_seed(seed)
        model = BaselineGCN(in_ch, cfg.hidden, out_ch, cfg.dropout)
        t0 = time.perf_counter()
        best_val, test_at_best = train_and_evaluate(
            model, data, device, cfg.epochs, cfg.lr, cfg.weight_decay
        )
        dt = time.perf_counter() - t0
        results.append(
            RunResult("baseline_gcn", seed, best_val, test_at_best, dt)
        )
        print(
            f"[baseline_gcn] seed={seed}  val={best_val:.4f}  test={test_at_best:.4f}  ({dt:.1f}s)"
        )

    # Relativistic variants
    for v in cfg.relativistic_velocities:
        name = f"relativistic_v{v:.2f}"
        for seed in cfg.seeds:
            set_seed(seed)
            model = RelativisticGCN(in_ch, cfg.hidden, out_ch, v, cfg.dropout)
            t0 = time.perf_counter()
            best_val, test_at_best = train_and_evaluate(
                model, data, device, cfg.epochs, cfg.lr, cfg.weight_decay
            )
            dt = time.perf_counter() - t0
            results.append(RunResult(name, seed, best_val, test_at_best, dt))
            print(
                f"[{name}] seed={seed}  val={best_val:.4f}  test={test_at_best:.4f}  ({dt:.1f}s)"
            )

    return results


def summarize(results: list[RunResult]) -> dict[str, dict[str, float]]:
    by_config: dict[str, list[RunResult]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    for name, runs in by_config.items():
        test_accs = [r.test_acc_at_best_val for r in runs]
        val_accs = [r.best_val_acc for r in runs]
        times = [r.train_time_s for r in runs]
        summary[name] = {
            "n_seeds": len(runs),
            "test_acc_mean": mean(test_accs),
            "test_acc_std": stdev(test_accs) if len(test_accs) > 1 else 0.0,
            "val_acc_mean": mean(val_accs),
            "val_acc_std": stdev(val_accs) if len(val_accs) > 1 else 0.0,
            "train_time_s_mean": mean(times),
        }
    return summary


def write_markdown(summary: dict[str, dict[str, float]], path: Path) -> None:
    lines = [
        "# Cora benchmark: GCN vs RelativisticGraphConv",
        "",
        "2-layer GNN trained on the Planetoid Cora split with the standard",
        "public train/val/test masks. Each config uses the same hidden size,",
        "optimiser, and number of epochs — only the convolution layer differs.",
        "",
        "| Config | Test accuracy (mean ± std) | Val accuracy (mean ± std) | Seeds | Train time (s) |",
        "|---|---|---|---|---|",
    ]
    ordered = sorted(
        summary.keys(),
        key=lambda k: (k != "baseline_gcn", k),  # baseline first
    )
    for name in ordered:
        s = summary[name]
        lines.append(
            f"| `{name}` | {s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f} "
            f"| {s['val_acc_mean']:.4f} ± {s['val_acc_std']:.4f} "
            f"| {int(s['n_seeds'])} "
            f"| {s['train_time_s_mean']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--velocities", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    parser.add_argument("--data-root", type=str, default="benchmarks/data")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    cfg = BenchmarkConfig(
        hidden=args.hidden,
        epochs=args.epochs,
        seeds=list(args.seeds),
        relativistic_velocities=list(args.velocities),
    )

    dataset_root = Path(args.data_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run(cfg, dataset_root, device)
    summary = summarize(results)

    json_path = output_dir / "cora.json"
    md_path = output_dir / "cora.md"

    json_payload = {
        "config": {
            "hidden": cfg.hidden,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "seeds": cfg.seeds,
            "velocities": cfg.relativistic_velocities,
            "device": str(device),
        },
        "summary": summary,
        "runs": [
            {
                "config": r.config,
                "seed": r.seed,
                "best_val_acc": r.best_val_acc,
                "test_acc_at_best_val": r.test_acc_at_best_val,
                "train_time_s": r.train_time_s,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)

    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print()
    print("Summary:")
    for name, s in summary.items():
        print(
            f"  {name:25s}  test={s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f}"
        )


if __name__ == "__main__":
    main()
