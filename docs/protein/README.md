# Protein: do loops help on real biology?

**Intent.** Ask the looped-math question on proteins: does a looped model beat a same-weights
shallow model, and match a deeper one, on protein tasks where structure needs many steps?

**Status.** Idea stage. P0 done on 2026-11-24 (below). Next candidate: P1, probe ESM-2 8M for
secondary structure (`notes.md`).

## Results

**P0 (2026-11-24): ESM-2's amino-acid table encodes side-chain chemistry.** In ESM-2 8M, 14 of
20 amino acids (70%) have their nearest neighbour in the same side-chain group. Chance is 14%
(label shuffles, 95th percentile 30%). The main direction (PC1) correlates with Kyte–Doolittle
hydropathy at r = 0.80. ESM-2 35M: 65%, r = 0.74. Pairs: K–R, D–E, S–T, F–Y, I–V, N→D, Q→E.
Clustering with no labels (6 clusters, scored by adjusted Rand index against the 6 side-chain
groups; 0 = random, 1 = exact): k-means ARI 0.62 for 8M, 0.52 for 35M; hierarchical 0.39 / 0.45.
Chance ARI is about 0 (95th percentile 0.12–0.14). k-means 8M clusters: A G P C | V L I M | F W Y |
S T | N D E | Q K R H. It splits our polar group and puts each amide (N, Q) with charged neighbours.
Not shown: anything about structure (these are the 20 input embeddings, before any protein is read).
Script `scripts/protein_p0.py`, output `results/protein/p0/p0.json`, page `visual/p0-amino-acids.html`
(artifact https://claude.ai/artifact/9oU5iL23HiH9cMV6efM1ri). CPU only, 6 threads, under 10 s, 1 GB RAM. Page template:
`scripts/protein_p0_page.template.html` (data goes in at `__DATA__`).

## Files

| file | holds |
|---|---|
| `notes.md` | basics, candidate tasks P1–P2, data sources, ready-made models, ESM-2 training data (unverified) |
| `visual/p0-amino-acids.html` | P0 page: the amino-acid map |

When the first experiment is planned, add the project files listed in `../README.md`.
