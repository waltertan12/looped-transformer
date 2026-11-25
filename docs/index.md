# looped-transformer

Research on looped transformers (the same layers run several times in a row). One active
project and two at the idea stage. Code: [GitHub](https://github.com/waltertan12/looped-transformer).
How these docs are organized: [docs README](README.md).

## Looped math (active)

Does looping add reasoning depth? Tiny models trained from scratch on nested arithmetic.
[Project README](looped-math/README.md)

- **[Results page](visual/loop-depth-results.html)**: charts, a primer on every term, what we learned, and the roadmap
- **[Attention viewer](visual/attention.html)**: click any character and see where each model looks, pass by pass
- [Hypotheses](looped-math/hypotheses.md): what's been tested and what's open
- [Plan](looped-math/plan.md): next steps, with decision rules written in advance
- Results: [H13 / H14](looped-math/results/ops.md) (pure addition / subtraction, operator mix) ·
  [H18](looped-math/results/tags.md) (tag fixes for longer numbers) ·
  [H19](looped-math/results/h19.md) (does each loop pass handle one level of brackets?)
- [Related work](looped-math/prior-art/related-work.md): the closest papers, compared with our results
- [Research log](looped-math/research-log.md) · [Decisions](looped-math/decisions.md) · [History](looped-math/history/README.md)

## Protein (idea stage)

The same loop question on proteins. [README](protein/README.md) · [Notes](protein/notes.md)

## Virtual cell (idea stage)

Toy gene-network problems shaped like our arithmetic. [README](virtual-cell/README.md) · [Notes](virtual-cell/notes.md)

---

[Glossary](glossary.md)
