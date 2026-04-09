# Cora benchmark: GCN vs RelativisticGraphConv

2-layer GNN trained on the Planetoid Cora split with the standard
public train/val/test masks. Each config uses the same hidden size,
optimiser, and number of epochs — only the convolution layer differs.

| Config | Test accuracy (mean ± std) | Val accuracy (mean ± std) | Seeds | Train time (s) |
|---|---|---|---|---|
| `baseline_gcn` | 0.8083 ± 0.0021 | 0.7940 ± 0.0072 | 3 | 1.6 |
| `relativistic_v0.00` | 0.7390 ± 0.0139 | 0.7707 ± 0.0196 | 3 | 1.7 |
| `relativistic_v0.30` | 0.7610 ± 0.0075 | 0.7707 ± 0.0076 | 3 | 1.6 |
| `relativistic_v0.60` | 0.7473 ± 0.0235 | 0.7600 ± 0.0125 | 3 | 1.8 |
| `relativistic_v0.90` | 0.7573 ± 0.0169 | 0.7627 ± 0.0232 | 3 | 1.8 |

## Interpretation

This is a **null-to-negative** result for the relativistic hypothesis on
Cora, and the reason is instructive: the comparison is methodologically
not fully controlled.

**The dominant effect is architectural, not relativistic.** At
`max_velocity = 0.0`, the Lorentz factor is γ ≈ 1 and the per-message
relativistic transformation is effectively an identity. The model is
still **~7 percentage points behind** the GCNConv baseline. That gap
therefore cannot be attributed to relativistic physics — it comes from
the fact that `GCNConv` performs the classical symmetric normalization
`D⁻¹ᐟ² A D⁻¹ᐟ² X W` while `RelativisticGraphConv` does not. On Cora,
where GCN-style normalization is historically what drives most of the
performance, that architectural choice alone is the bulk of the
difference.

**The velocity sweep itself is flat within noise.** Test accuracies at
`v ∈ {0.3, 0.6, 0.9}` all fall within ~1 percentage point of each other,
well inside the across-seed standard deviation of individual configs.
The relativistic effect, in isolation, is not statistically distinguishable
on this dataset with these hyperparameters.

**What this means going forward.** Cora is the wrong benchmark to isolate
the relativistic signal. A cleaner follow-up would compare
`RelativisticGraphConv` against a non-normalised message-passing baseline
(e.g. `SAGEConv` without the mean aggregation), or extend
`RelativisticGraphConv` with an optional `normalize=True` argument so the
two sides of the comparison share their normalisation. Larger / different
tasks where edge normalisation is less decisive — molecular property
prediction, OGB-Arxiv, heterophilic benchmarks — may also paint a very
different picture.

These numbers are reported as-is, not tuned, not cherry-picked. That is
the point of the `benchmarks/` directory.
