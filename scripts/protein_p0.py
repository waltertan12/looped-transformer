#!/usr/bin/env python
"""Protein P0: does ESM-2 place chemically similar amino acids near each other?

Reads the 20 amino-acid input embeddings of ESM-2 (8M and 35M), projects them to 2D with PCA,
and scores them against side-chain chemistry (no training, CPU only, about 1 minute).

    OMP_NUM_THREADS=6 nice -n 10 .venv/bin/python scripts/protein_p0.py   # -> results/protein/p0/p0.json

Scores:
  group match  share of amino acids whose nearest neighbour (cosine, full space) is in the same
               side-chain group. Chance level is computed by shuffling the group labels.
  hydropathy   Pearson r between PC1 / PC2 and the Kyte-Doolittle hydropathy scale.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/data/walter/huggingface")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

torch.set_num_threads(6)
REPO = Path(__file__).resolve().parent.parent
MODELS = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D"}
GROUP = {**dict.fromkeys("AVLIM", "greasy (aliphatic)"), **dict.fromkeys("FWY", "aromatic"),
         **dict.fromkeys("STNQ", "polar, no charge"), **dict.fromkeys("DE", "acidic (−)"),
         **dict.fromkeys("KRH", "basic (+)"), **dict.fromkeys("GPC", "special (G, P, C)")}
NAME = dict(A="alanine", V="valine", L="leucine", I="isoleucine", M="methionine", F="phenylalanine",
            W="tryptophan", Y="tyrosine", S="serine", T="threonine", N="asparagine", Q="glutamine",
            D="aspartate", E="glutamate", K="lysine", R="arginine", H="histidine", G="glycine",
            P="proline", C="cysteine")
KD = dict(I=4.5, V=4.2, L=3.8, F=2.8, C=2.5, M=1.9, A=1.8, G=-0.4, T=-0.7, S=-0.8, W=-0.9, Y=-1.3,
          P=-1.6, H=-3.2, E=-3.5, Q=-3.5, D=-3.5, N=-3.5, K=-3.9, R=-4.5)
AAS = list(GROUP)


def nn_match(X: np.ndarray, labels: list[str]) -> tuple[float, list[str]]:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(1)
    return float(np.mean([labels[i] == labels[j] for i, j in enumerate(nn)])), [AAS[j] for j in nn]


def main() -> None:
    out = {"groups": GROUP, "names": NAME, "hydropathy": KD, "models": {}}
    rng = np.random.default_rng(0)
    labels = [GROUP[a] for a in AAS]
    for tag, hf in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf)
        model = AutoModelForMaskedLM.from_pretrained(hf)
        W = model.get_input_embeddings().weight.detach().numpy()
        X = W[[tok.convert_tokens_to_ids(a) for a in AAS]].astype(np.float64)
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Xc @ Vt[:2].T
        var = (S**2 / (S**2).sum())[:2]
        match, nn = nn_match(X, labels)
        chance = [nn_match(X, list(rng.permutation(labels)))[0] for _ in range(2000)]
        kd = np.array([KD[a] for a in AAS])
        r = [float(np.corrcoef(P[:, k], kd)[0, 1]) for k in range(2)]
        out["models"][tag] = {
            "hf": hf, "dim": int(X.shape[1]),
            "points": {a: [float(P[i, 0]), float(P[i, 1])] for i, a in enumerate(AAS)},
            "var_explained": [float(v) for v in var], "nearest": dict(zip(AAS, nn)),
            "group_match": match, "chance_mean": float(np.mean(chance)),
            "chance_p95": float(np.quantile(chance, 0.95)),
            "p_value": float(np.mean(np.array(chance) >= match)), "r_hydropathy_pc": r,
        }
        print(f"{tag}: dim {X.shape[1]}, group match {match:.2f} (chance {np.mean(chance):.2f}, "
              f"95th pct {np.quantile(chance, 0.95):.2f}), r(PC1,KD) {r[0]:+.2f}, r(PC2,KD) {r[1]:+.2f}, "
              f"PC var {var[0]:.2f}/{var[1]:.2f}")
    p = REPO / "results/protein/p0/p0.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("->", p)


if __name__ == "__main__":
    main()
