# Virtual cell: ideas to explore (jotted 2026-11-24)

Not planned. Walter's interest: what is the smallest "virtual cell" problem that works like our
tiny looped transformer on arithmetic? Facts are from memory: check before relying on them.

## What "virtual cell" means

A model that predicts how a cell responds to a change. The usual form: switch off gene X (a
*perturbation*), predict how the activity (*expression*) of every other gene changes. Real data
comes from single-cell RNA sequencing, which measures thousands of genes in each of many cells.
Perturbation screens (Perturb-seq) do this after switching off one gene, or a pair, per cell.
Arc Institute ran a Virtual Cell Challenge in 2025. Known models: Geneformer, scGPT, scFoundation.

**The warning to remember:** several papers found that simple baselines (predict the average
change, or add up the effects of single knockouts) match or beat the deep models on perturbation
prediction. Any experiment here must beat those baselines first.

## Why loops fit

Genes switch each other on and off in a network. A knockout's effect spreads hop by hop:
X turns off Y, Y stops repressing Z, and so on. The cell's final state is where this settles, a
fixed point found by repeating the same update. A looped model that repeats one block is shaped
like that process. **Hops in the network play the role of bracket depth.**

## Toy ideas, smallest first

| # | Idea | Depth dial | Exact answer? |
|---|---|---|---|
| V1 | **Boolean network.** 10–50 genes, each on or off, each updated by a fixed rule of a few others. Given a start state, predict the state after t steps | t, and the longest chain in the network | Yes, simulate it |
| V2 | **Knockout spread on a network without cycles.** Switch off one gene, predict which genes change | Hops from the knocked-out gene | Yes |
| V3 | **Steady state.** Same as V1, but predict where the network settles. Test extra loops at test time on networks that take longer to settle (like H9) | Steps to settle | Yes |
| V4 | **Synthetic expression data** from a simulator such as SERGIO or BoolODE (noisy, realistic counts generated from a known network) | Network depth | Ground-truth network is known |
| V5 | **Small real dataset:** a combinatorial Perturb-seq set (pairs of knockouts, about 100 genes). Predict pair effects from single effects | None built in | No. Must beat "add the two single effects" |

Each one copies our arithmetic recipe: shallow vs looped vs deep with the same weights, a dial
for the number of steps, rules written before the runs. V1–V3 can generate unlimited data in
seconds. V5 is the first contact with real biology and its noise.

## ESM-2's data

ESM-2 (Meta, 2022) was trained on public data: UniRef50 clusters from UniProt, sampling member
sequences from UniRef90. Code and weights are public on GitHub and HuggingFace.

## Links in this repo

[Protein track](protein.md) · [Ideas](ideas.md#idea-protein)
