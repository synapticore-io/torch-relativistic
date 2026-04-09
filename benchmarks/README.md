# Benchmarks

This directory is where the claims in the project's motivation section
([README › Why Relativistic? The Hypothesis](../README.md#-why-relativistic-the-hypothesis))
get tested against real data.

The guiding principle is honesty: **every run writes its raw per-seed
numbers to `results/` alongside a summary, and the README table is
regenerated from those files.** No cherry-picking, no silent hyperparameter
sweeps — if the numbers are disappointing, they get reported anyway,
because a measured null result is still informative.

## What's here

| Script | Dataset | Compares |
|---|---|---|
| [`cora_gcn_vs_relativistic.py`](cora_gcn_vs_relativistic.py) | Planetoid / Cora | `GCNConv` vs `RelativisticGraphConv` sweeping `max_relative_velocity` ∈ {0, 0.3, 0.6, 0.9} |

More scripts will be added as new modules (`RelativisticSelfAttention`,
`TerrellPenroseSNN`, …) get benchmarked on appropriate tasks.

## Running

From the repo root, after `uv sync`:

```bash
# default config: 3 seeds, 200 epochs, velocities 0.0/0.3/0.6/0.9
uv run python benchmarks/cora_gcn_vs_relativistic.py

# custom seeds and velocity sweep
uv run python benchmarks/cora_gcn_vs_relativistic.py \
    --seeds 0 1 2 3 4 \
    --velocities 0.0 0.2 0.4 0.6 0.8 \
    --epochs 300
```

Planetoid/Cora is downloaded on first run into `benchmarks/data/`
(about 170 KB compressed). Training is CPU-friendly: a single seed on
CPU takes ~30–60 s for 200 epochs, so a full default run of
1 baseline + 4 relativistic configs × 3 seeds ≈ 15 runs is roughly
10–15 minutes of wall-clock time on a laptop without a GPU.

## Outputs

Each run writes:

- `results/cora.json` — raw per-seed `(best_val_acc, test_acc_at_best_val,
  train_time_s)` and the exact config that produced them.
- `results/cora.md` — a human-readable summary table with mean ± std
  test accuracy over the seeds, suitable for pasting into the project
  README.

`benchmarks/data/` (the dataset cache) is git-ignored. The `results/`
directory is tracked so that published numbers are reproducible from
git history.

## Expected outcomes

Three classes of outcome are all valid contributions:

1. **Null result** — the curves for `v=0` and `v=0.9` lie on top of
   each other. The relativistic machinery is a no-op on this task;
   the extension is then a cleaner way to explore Lorentz-equivariant
   architectures in general.
2. **Positive result** — the relativistic variant beats the baseline by
   more than the between-seed noise. This would be a genuine finding
   worth writing up as a short paper / blog post.
3. **Negative result** — high velocities hurt. This tells us the
   inductive bias interacts badly with standard GCN tasks, which is
   itself useful information about the limits of the approach.

Whatever the numbers turn out to be, they get reported here, not
quietly buried.
