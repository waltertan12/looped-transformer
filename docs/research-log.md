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

## 2026-11-21 — Memory is the scorecard; compare against the next size up (exp2)

**My thinking.**
- Track memory use in general. *If we end up needing as much memory as the regular model,
  then we've lost.*
- The framing: use the model as-is as the baseline, then compare the looped version against
  the next step up in the same model family. If a looped Qwen3-0.6B gets close to Qwen3-1.7B
  while using 0.6B-sized memory, that's a win.
- Don't train on the benchmarks. Train on general text and keep the benchmarks for testing.
- Keep experiments and datasets on the spare SSD (`/mnt/data`). That migration waits until I
  have access to set up permissions. Until then, everything runs on the main drive.

**How this changes the plan.** Looping with "dials" that start at zero (gated extra passes)
keeps the weights at exactly 0.6B size. The model starts identical to the original 0.6B and
learns to use the extra passes. The teacher is now the 1.7B, which shares the 0.6B's
vocabulary. Distilling the 0.6B from itself would teach it nothing new.

**Why there's a control run.** Learning from the 1.7B might improve the 0.6B by itself, with
no looping at all. So exp2 trains a no-loop 0.6B on the same data and teacher. Looping only
counts as helping if the looped model beats this control.

**What exp2 measures.**

| model | what it tells us |
|---|---|
| Qwen3-0.6B, unchanged | starting point |
| Qwen3-1.7B, unchanged | the target |
| 0.6B distilled from 1.7B, no loops | how much the training alone helps |
| 0.6B distilled from 1.7B, looped (layers 8–19, 2–4 passes), scored at K = 1, 2, 3, 4, 6 | whether more passes = better, and whether it extrapolates past training (K = 6) |

Benchmarks: ARC-Easy, ARC-Challenge, HellaSwag, GSM8K (300 questions each). HellaSwag is the
control: it's mostly knowledge, so looping shouldn't help it much. ARC-Challenge and GSM8K
are multi-step, so looping should help there if it helps anywhere.

Memory is measured at a fixed workload (batch 1, 2048-token prompt, 256 new tokens):
weights, generation memory (KV cache), peak GPU memory, and speed. Extra passes don't add
weights, but they do add KV-cache memory and compute, so that cost is reported too.

**Result: the loop never switched on.**

| model | weights | ARC-Easy | ARC-Chall.* | HellaSwag* | GSM8K | gen tok/s | KV cache |
|---|---|---|---|---|---|---|---|
| 0.6B, unchanged | 1.11 GB | 0.617 | 0.380 | 0.523 | 0.540 | 122 | 0.25 GB |
| 1.7B, unchanged (target) | 3.21 GB | 0.700 | 0.443 | 0.577 | 0.723 | 118 | 0.25 GB |
| 0.6B trained, no loops (control) | 1.11 GB | 0.643 | 0.373 | 0.520 | 0.440 | 121 | 0.25 GB |
| looped, K=1 | 1.11 GB | 0.643 | 0.373 | 0.523 | 0.470 | 119 | 0.25 GB |
| looped, K=2 | 1.11 GB | 0.643 | 0.380 | 0.520 | 0.473 | 84 | 0.35 GB |
| looped, K=3 | 1.11 GB | 0.640 | 0.370 | 0.523 | 0.457 | 67 | 0.46 GB |
| looped, K=4 | 1.11 GB | 0.643 | 0.377 | 0.517 | 0.467 | — | — |
| looped, K=6 | _pending_ | | | | | | |

\*acc_norm (accuracy adjusted for answer length). Margin of error is about ±0.03. Memory test: batch 1, 2048-token prompt, 256 new tokens.

- Scores are flat across K. The looped model is indistinguishable from the no-loop control.
- The dials ended at 0.028 (pass 2) and 0.004 (passes 3 and 4), where 1.0 is a full pass.
  The extra passes were almost no-ops.
- Training loss matched the control almost exactly (KL 0.2824 vs 0.2823).
- Extra passes still cost compute and KV-cache memory (122 → 67 tok/s at K=3).
- Training itself hurt math: GSM8K 0.54 → 0.44 for the control. FineWeb-Edu is general web
  text, and the original Qwen was trained on a lot of math.

### Thoughts and conclusions (2026-11-21)

**My questions.** Is the training method bad? Does looping not inherently provide an
advantage, so that we must be more "intelligent" about it? Do we need exp3 to understand
better?

**Is the training method bad? Partly, yes, in a specific way: it let the model opt out.**
- The dials started at 0, so the model's easiest option was to leave them there. The
  pretrained layers were only ever trained to be applied once. Applying them a second
  time, even a little, probably looks like noise to the rest of the network and makes the
  loss slightly worse. So the safe move was to keep the dials shut, and it took that.
- Starting from "identical to the original" removed the broken start, but it also removed
  any pressure to use the loop.

**Did the training ask for extra thinking at all? Probably not.**
- On web text, the 0.6B is already close to the 1.7B (KL about 0.3). There was little gap
  for extra passes to close.
- The gap that matters, GSM8K 0.54 vs 0.72, is multi-step reasoning. The training data has
  very little of that, so nothing in training rewarded extra steps of thinking.

**Does looping inherently help? Not answered yet.** This doesn't show looping is useless.
It shows that this setup never made the model use the loop. As I understand published work
on looped / recurrent-depth models, the gains come when the model is trained with the loop
forced on (often from scratch, or with a long retrofit) on data where extra steps pay off.
To check before relying on it.

**So we have to be more deliberate**, in three ways:
1. **Force the loop on**, or start the dials open, so the model has to learn to use the
   extra passes instead of ignoring them.
2. **Train on data where thinking pays off**: math, multi-step problems, synthetic
   state-tracking (the playbook's "fast signal" task). Mixing in math also counters the
   GSM8K loss.
3. **Measure the fast signal first.** A synthetic multi-hop task should show quickly whether
   K=3 beats K=1, before spending hours on slow benchmarks.

**Do we need exp3? Yes, but not the planned one.** The planned exp3 compared where to put
the loop, using the same dials-start-at-zero recipe. It would almost certainly have shown
dials stuck at zero in every position. Placement only matters once the loop is on. So
**exp3 now tests how to switch the loop on**, and placement moves to exp4.

---

## 2026-11-21 — What we learned from exp1 and exp2

**"Why did we even give it the option?"** The dials-start-at-zero design was a reaction to
exp1. There, cutting layers out and looping broke the model on day one (accuracy near
random), and training only partly repaired it. The playbook recommends identity-preserving
gates for exactly this. They fixed the broken start, but they also gave the model a free way
to opt out of the loop, and it took it. Nobody flagged that downside before the run.

**How people usually train looped models (from memory, to verify against the papers):**
- Most working looped models are trained with the loop **from the start**: Universal
  Transformers, ALBERT, Huginn-3.5B (recurrent depth, 2025), Ouro/LoopLM (2025). The loop is
  always fully on, and the number of loops is sampled at random each step.
- Converting an existing model into a looped one (Relaxed Recursive Transformers, 2024;
  retrofitted recurrence, 2025) uses **continued pretraining on billions of tokens**, with
  the loop forced on and often a curriculum that adds loops gradually.
- We used ~33M tokens of post-training, roughly 100× less than those or more, with an
  opt-out. A short post-training bolt-on loop isn't a known working recipe.

**Lessons.**
1. **A loop the model can switch off will be switched off.** Pretrained layers were trained
   to run once. A second application looks like noise at first, so the easy move is to
   ignore it. To test looping, the model can't be allowed to opt out.
2. **Safe start and forced use pull against each other.** Cut-and-loop (exp1) breaks the
   model; zero dials (exp2) never use the loop. The middle ground is to force the loop on but
   ease in, for example by adding loops gradually or starting the dials open.
3. **The training data has to reward thinking.** On web text, the 0.6B is already close to
   the 1.7B (KL about 0.3), so extra passes have little to gain. The real gap is multi-step
   reasoning (GSM8K 0.54 vs 0.72), and the training text has little of it.
4. **Training on general web text erodes math.** GSM8K dropped 0.54 → 0.44 even with no
   looping. Any future recipe needs math in the mix, and must be judged against a control
   trained the same way.
5. **Always run a control.** Without the no-loop control, exp2's small ARC-Easy gain would
   have looked like a looping win. It came from the training alone.
6. **Extra passes cost something even when unused.** At K=3: 122 → 67 tok/s, KV cache
   0.25 → 0.46 GB. At K=4: 54 tok/s, 0.56 GB. Weights stay at 1.11 GB. Any looping gain has
   to be weighed against this.
7. **The harness works.** Benchmarks, a fixed memory test, checkpoints that can be scored at
   any K, and cache-correctness tests are all in place, so the next experiments are cheap to
   set up.
8. **Scale matters.** Published looped models use billions to trillions of tokens. On this
   GPU (~7k tok/s for looped training), a few hundred million tokens is about a day. Any
   positive result here is early evidence, not a match for the papers.

**What this means for next steps.** Exp3 tests forcing the loop on ("forced"), starting the
dials open ("half"), and faster-learning dials ("fastdial"). If none of them beats the
control, the likely next steps are a loop-count curriculum, math in the training mix, and
longer training, before concluding anything about looping itself.

---

## 2026-11-21 — Exp3: switching the loop on

Same setup as exp2's looped run (layers 8–19, teacher 1.7B, K ∈ {2,3,4}), 500 steps each,
scored at K = 1, 2, 4:

| run | how the extra passes are used |
|---|---|
| forced | no dials; every extra pass applied in full |
| half | dials start at 0.5 and are learned |
| fastdial | dials start at 0 but learn 10× faster |

**What we're looking for:** a run whose score rises with K and beats the no-loop control
(ARC-Easy 0.643, ARC-Challenge 0.373, GSM8K 0.440). "forced" will probably start out
damaged, like exp1, so it may need more steps to recover.

**Result:** _pending_

---

## Open questions / ideas to come back to

- Exp4: loop placement (early 4–15 / late 12–23 / small 10–15), once a recipe switches the
  loop on.
- Mix math / reasoning data into training (e.g. OpenMathInstruct-2), both to reward extra
  passes and to stop GSM8K forgetting. Streamed, not downloaded.
- Does more loops at eval time help on a trained checkpoint, or does it only work at the loop
  count it was trained with?
- Training with a random loop count each step so the model works at any depth.
- Train only the looped block vs the whole model.
- Bigger models (Qwen3-1.7B, 4B) and harder benchmarks (gsm8k, code). SWE-bench needs an
  agent setup and Docker, so it comes later.
