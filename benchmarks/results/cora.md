# Cora benchmark: GCN vs RelativisticGraphConv

2-layer GNN trained on the Planetoid Cora split with the standard
public train/val/test masks. Each config uses the same hidden size,
optimiser, and number of epochs — only the convolution layer differs.

| Config | Test accuracy (mean ± std) | Val accuracy (mean ± std) | Seeds | Train time (s) |
|---|---|---|---|---|
| `baseline_gcn` | 0.8117 ± 0.0067 | 0.7900 ± 0.0053 | 3 | 1.7 |
| `baseline_sage` | 0.8073 ± 0.0049 | 0.7920 ± 0.0020 | 3 | 1.2 |
| `rel_norm_v0.00` | 0.8167 ± 0.0012 | 0.7893 ± 0.0061 | 3 | 2.2 |
| `rel_norm_v0.30` | 0.8040 ± 0.0114 | 0.7940 ± 0.0072 | 3 | 2.2 |
| `rel_norm_v0.60` | 0.8157 ± 0.0038 | 0.7913 ± 0.0095 | 3 | 4.0 |
| `rel_norm_v0.90` | 0.8077 ± 0.0078 | 0.7960 ± 0.0060 | 3 | 4.6 |
| `relativistic_v0.00` | 0.7523 ± 0.0189 | 0.7700 ± 0.0100 | 3 | 1.6 |
| `relativistic_v0.30` | 0.7657 ± 0.0117 | 0.7740 ± 0.0053 | 3 | 1.5 |
| `relativistic_v0.60` | 0.7467 ± 0.0115 | 0.7667 ± 0.0042 | 3 | 1.6 |
| `relativistic_v0.90` | 0.7580 ± 0.0161 | 0.7800 ± 0.0060 | 3 | 1.6 |
