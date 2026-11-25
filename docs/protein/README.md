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
Not shown: anything about structure (these are the 20 input embeddings, before any protein is read).
Script `scripts/protein_p0.py`, output `results/protein/p0/p0.json`, page `visual/p0-amino-acids.html`
(artifact https://claude.ai/artifact/9oU5iL23HiH9cMV6efM1ri). CPU only, 6 threads, 8 s, 1 GB RAM.

## Files

| file | holds |
|---|---|
| `notes.md` | basics, candidate tasks P1–P2, data sources, ready-made models, ESM-2 training data (unverified) |
| `visual/p0-amino-acids.html` | P0 page: the amino-acid map |

When the first experiment is planned, add the project files listed in `../README.md`.
