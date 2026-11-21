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

**Result:** _pending_

---

## Open questions / ideas to come back to
- Does more loops at eval time help on a trained checkpoint, or does it only work at the loop
  count it was trained with?
- Training with a random loop count each step so the model works at any depth.
- Train only the looped block vs the whole model.
- Bigger models (Qwen3-1.7B, 4B) and harder benchmarks (gsm8k, code). SWE-bench needs an
  agent setup and Docker, so it comes later.
