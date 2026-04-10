# Cora benchmark: GCN vs RelativisticGraphConv

2-layer GNN trained on the Planetoid Cora split with the standard
public train/val/test masks. Each config uses the same hidden size,
optimiser, and number of epochs — only the convolution layer differs.

| Config | Test accuracy (mean ± std) | Val accuracy (mean ± std) | Seeds | Train time (s) |
|---|---|---|---|---|
| `baseline_gcn` | 0.7853 ± 0.0049 | 0.7933 ± 0.0031 | 3 | 2.1 |
| `baseline_sage` | 0.7673 ± 0.0031 | 0.8020 ± 0.0020 | 3 | 2.7 |
| `rel_norm_v0.00` | 0.7860 ± 0.0044 | 0.7867 ± 0.0012 | 3 | 10.2 |
| `rel_norm_v0.30` | 0.7790 ± 0.0095 | 0.7887 ± 0.0042 | 3 | 9.3 |
| `rel_norm_v0.60` | 0.7823 ± 0.0021 | 0.7880 ± 0.0053 | 3 | 9.1 |
| `rel_norm_v0.90` | 0.7793 ± 0.0125 | 0.7940 ± 0.0035 | 3 | 9.8 |
| `relativistic_v0.00` | 0.7770 ± 0.0078 | 0.7880 ± 0.0035 | 3 | 8.0 |
| `relativistic_v0.30` | 0.7620 ± 0.0114 | 0.7927 ± 0.0092 | 3 | 7.7 |
| `relativistic_v0.60` | 0.7627 ± 0.0085 | 0.7840 ± 0.0020 | 3 | 8.0 |
| `relativistic_v0.90` | 0.7657 ± 0.0095 | 0.7907 ± 0.0042 | 3 | 7.6 |
