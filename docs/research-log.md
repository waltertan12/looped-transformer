# Research log

How I'm thinking about this project, in my own words where possible. Each entry records the
reasoning at the time, so it can be explained and revisited later.

---

## 2026-11-21 — Starting point

**The core idea.** Take a pretrained Qwen transformer, slice it in half, and turn it into a
loop transformer: run a block of layers repeatedly with shared weights. Then vary the number
of loops and run it against a benchmark (SWE-bench, or really any benchmark) to see if looping
can improve performance with smaller models.

**The question underneath it.** Can a smaller model (fewer unique parameters) get back the
quality of a bigger one by spending more compute (more passes through the same layers)
instead of having more weights?

**How I'm sequencing the work.**
1. First, just be able to run Qwen in general against a benchmark locally. Get a generic
   harness that works before any experimentation.
2. Second, ask whether the loop transformer can even work at all. A sliced-and-looped model
   probably won't work out of the box, so it has to be trained.
3. Only after that does the actual experimentation begin (different loop counts, sizes,
   benchmarks).

**Constraints I'm working within.**
- One RTX 5090 on this machine, power-limited to 450 W. Things should generally fit inside the
  GPU, so experiments are sized for a single card.

**Process preferences.**
- Used a stronger model (Opus) for the setup because getting the foundation right matters.
- Log my thinking as I go, tracked in git, so changes in how I think are visible over time.
  First capture *how I'm thinking*; get it explained back to me later.

### What the first run showed
Qwen3-0.6B-Base, 0-shot, 300 examples per task:

| model | unique params | arc_easy acc | hellaswag acc_norm |
|---|---|---|---|
| base (28 layers) | 596M | 0.617 | 0.523 |
| looped `4:16:2:28`, untrained | 407M | 0.357 | 0.297 |

Expected: slicing and looping without training breaks the model (close to chance, about 0.25).
This confirms step 2, which is to train it.

---

## 2026-11-21 — First healing experiment (exp1)

**My thinking.** Do something simple and straightforward first: have the original model give
the answers, and train ("distill") the looped model to match them. I don't know the right way
to do this yet, so start with the basic version and see whether it moves the numbers at all.

**What's actually running.** The original model and the looped model both read the same web
text (fineweb-edu). At every word, the looped model is trained to match the original model's
full probability spread over next words, not just its single top pick. 500 steps, about 16M
tokens, with checkpoints at steps 250 and 500.

**What I'm looking for.** Does a short bit of training pull the looped model back up toward
the original (0.617 / 0.523), away from where it started untrained (0.357 / 0.297)?

**Result:** partial recovery. Training helped a lot at first, then levelled off well short of
the original.

| model | arc_easy acc | hellaswag acc_norm |
|---|---|---|
| original (596M) | 0.617 | 0.523 |
| looped, untrained (407M) | 0.357 | 0.297 |
| looped, 250 steps (~8M tokens) | 0.440 | 0.393 |
| looped, 500 steps (~16M tokens) | 0.447 | 0.403 |

- The gap to the original's predictions (KL) fell from 5.7 to about 1.2 by step ~250, then
  stopped improving; the learning rate was already decaying.
- Most of the gain came in the first 250 steps. Steps 250→500 changed little (within noise
  at 300 examples, ±0.03).
- It recovered roughly a third to half of what slicing lost.
- 12.4 GB of GPU memory, about 11.8k tokens/s, about 23 minutes.

**My read so far.** The basic "match the original" recipe works, but a short run isn't
enough. Either it needs much more training, or (per the playbook) the model shouldn't be
broken at the start: initialize so the looped model matches the original at step 0.

---

## 2026-11-21 — The playbook: where this is heading

Wrote down a fuller plan in [`playbook.md`](playbook.md). The main shifts in thinking:

- **Base model for Phase 1 stays Qwen** (currently Qwen3-0.6B-Base, 28 layers). The
  playbook was written for TinyLlama-1.1B, but its ideas carry over to Qwen; we're keeping
  Qwen rather than switching.
- **Two separate questions, kept separate.** Track A: does looping *add* reasoning depth
  with the same weights? Track B: can looping *replace* layers with fewer weights at the same
  depth? The current Qwen exp1 (`4:16:2:28`) is a Track B experiment.
- **Don't break the model at the start.** Initialize so the looped model behaves exactly
  like the original at step 0 (near-zero gates on extra passes), instead of slicing it and
  then trying to recover.
- **Never loop the first or last layers.** Keep a prelude and a coda.
- **Measure curves, not points.** Evaluate every checkpoint at several loop counts and plot
  metric vs. K. A fast synthetic task (multi-hop tracking, parity) should come first because
  it shows the effect quickly. Factual recall (HellaSwag, TriviaQA) is a control that should
  stay flat.
- **Phase 2 is protein language models** (ESM-2 / ESMC). Folding is iterative constraint
  satisfaction, so looping fits it naturally. These models have no KV cache and almost all
  their weights sit in the transformer blocks.

---

## Open questions / ideas to come back to
- Does more loops at eval time help on a trained checkpoint, or does it only work at the loop
  count it was trained with?
- Training with a random loop count each step so the model works at any depth.
- Train only the looped block vs the whole model.
- Bigger models (Qwen3-1.7B, 4B) and harder benchmarks (gsm8k, code). SWE-bench needs an
  agent setup and Docker, so it comes later.
