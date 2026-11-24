# Protein track (idea stage)

A second, small experiment track in this repo: do loops help on real biology, the way they help
on nested arithmetic? Nothing is built or running yet. Written 2026-11-24 from a conversation with
Walter. Facts are from memory: check sizes and numbers before planning around them.

## The basics

- **Residue** = one amino acid in a protein chain. A protein is a string over a 20-letter
  alphabet, like `MKTAYIAKQR…`, and each letter is a residue. (Chemistry name: when amino acids
  link up, each loses a few atoms, and what's left in the chain is the "residue".)
- **DNA → RNA → protein is a fixed table**, the genetic code. DNA becomes RNA by swapping T for U.
  RNA is read 3 letters at a time (a *codon*): 64 codons map to 20 amino acids plus "stop", for
  example `ATG` → M (start), `TTT` → F, `TAA` → stop. Biopython: `Seq("ATGTTT").translate()`.
- **The hard part is protein letters → what the protein does.** The chain folds into a 3D shape
  and the shape decides its job. Which letters matter, which ones touch after folding, and which
  changes break it are not in any table.
- **What protein embeddings do.** A protein language model reads millions of real proteins with
  some letters hidden and learns to guess them. To guess well it picks up which positions never
  change (essential) and which change together (they usually touch in 3D). The embedding is its
  summary of that, one vector per residue. Proteins that do similar jobs get similar vectors.

## How accuracy is checked

| Check | What it asks | What "bad" looks like |
|---|---|---|
| Masked-letter prediction (perplexity) | Hide 15% of letters, how well does it guess them? | Random over 20 letters = 20. Lower is better |
| Secondary structure | From the embedding, label each residue helix, strand or coil (a small probe, like H19) | Always guessing the most common label, roughly 40–50% |
| Contact prediction | Which residue pairs touch in the folded shape? Scored on the top guesses | Near 0 for random. Long-range contacts (far apart in the chain) are the hard ones |
| Mutation effects (ProteinGym) | Does the model's surprise at a mutation match how much it breaks the protein in the lab? | Correlation near 0 |

## Why loops might help

- **AlphaFold 2 already loops.** Its "recycling" runs the network, feeds the predicted structure
  back in and runs it again (3 times by default), improving each round. That's a looped model
  with input injection, like ours.
- **Contacts are a puzzle of linked guesses.** If residue 10 touches 80 and 80 touches 150, then
  10 is probably near 150. Working that out is step-by-step refinement, the kind of work loops
  helped with on arithmetic. Long proteins have more of these chains, so they may need more passes.

## The experiment it points to

The arithmetic design, moved to proteins:
- Train shallow, looped and deep protein models from scratch on Swiss-Prot (same weights and
  training budget logic as D-012). Masked-letter training.
- Score contacts split by **protein length** and **contact range**. These play the roles of
  bracket depth and number size.
- **Prediction, written before any run:** looped beats shallow most on long-range contacts in long
  proteins, and matches deep there (like H1 and H2). Extra passes at test time help longer
  proteins (like H9).
- A negative result counts: loops helping only short-range contacts, or not at all.
- Caveat: small from-scratch models are weak at contacts. Compare our models with each other,
  not with AlphaFold.

## Steps, smallest first

| # | Step | Why | Cost |
|---|---|---|---|
| P1 | Run ESM-2 8M (`esm2_t6_8M_UR50D`, HuggingFace) and probe secondary structure and contacts | Learn the data, the tools and what "good" looks like, with no training | An afternoon |
| P2 | Train tiny shallow / looped / deep protein models on Swiss-Prot, probe the same way, ESM-2 8M as a reference | The real loop test (above) | Like D-012, hours per run |
| P3 | Genetic-code tasks as a wind tunnel: translate, reverse strand, find the gene (first `ATG` to stop), remove introns, pick the reading frame | Unlimited data, exact answers, a natural step count | Small. Risk: too easy or too artificial |

Recommendation: P1, then P2. P3 only if we want a controlled toy.

## Data and models to know about

- **Sequences:** Swiss-Prot (about 570k hand-curated proteins, a good first set), UniRef50 (tens of
  millions), PDB (about 200k 3D structures).
- **Labeled benchmarks:** TAPE (secondary structure, contacts, stability, fluorescence), FLIP,
  ProteinGym (mutation effects). Secondary structure has roughly 10k training proteins.
- **Fully synthetic with exact answers:** genetic-code tasks; lattice proteins (the HP model:
  chains of H and P beads folded on a 2D grid, best fold computable); Potts-model sequences
  (invent which positions interact, sample sequences, ask the model to recover the interactions).
- **Ready-made embeddings:** proteins ESM-2 (smallest about 8M weights), ProtT5, ProtBERT. DNA
  DNABERT-2, Nucleotide Transformer, HyenaDNA, Evo 2 (huge). Codons CaLM. RNA RNA-FM.

## Running alongside the arithmetic track

One GPU, at most 2 trainings at a time across both tracks. Before launching anything, run
`scripts/lock.py show`. Register a new queue with `scripts/lock.py add`. The record lives in
`.locks/running.json`, which is not committed.
