# Cora benchmark: GCN vs RelativisticGraphConv

2-layer GNN trained on the Planetoid Cora split with the standard
public train/val/test masks. Each config uses the same hidden size,
optimiser, and number of epochs — only the convolution layer differs.

| Config | Test accuracy (mean ± std) | Val accuracy (mean ± std) | Seeds | Train time (s) |
|---|---|---|---|---|
| `baseline_gcn` | 0.6877 ± 0.0070 | 0.7013 ± 0.0050 | 3 | 1.5 |
| `baseline_sage` | 0.6947 ± 0.0015 | 0.7027 ± 0.0061 | 3 | 3.1 |
| `rel_norm_v0.00` | 0.6917 ± 0.0067 | 0.6967 ± 0.0042 | 3 | 7.9 |
| `rel_norm_v0.30` | 0.6880 ± 0.0087 | 0.7007 ± 0.0081 | 3 | 7.6 |
| `rel_norm_v0.60` | 0.6987 ± 0.0084 | 0.7040 ± 0.0104 | 3 | 7.8 |
| `rel_norm_v0.90` | 0.6953 ± 0.0140 | 0.7040 ± 0.0122 | 3 | 7.7 |
| `relativistic_v0.00` | 0.6183 ± 0.0114 | 0.6467 ± 0.0076 | 3 | 5.4 |
| `relativistic_v0.30` | 0.6313 ± 0.0146 | 0.6493 ± 0.0046 | 3 | 5.3 |
| `relativistic_v0.60` | 0.6277 ± 0.0023 | 0.6487 ± 0.0023 | 3 | 5.5 |
| `relativistic_v0.90` | 0.6380 ± 0.0142 | 0.6507 ± 0.0058 | 3 | 5.8 |
