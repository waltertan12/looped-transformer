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
| looped, K=4 | 1.11 GB | 0.643 | 0.377 | 0.517 | 0.467 | 54 | 0.56 GB |
| looped, K=6 | 1.11 GB | 0.647 | 0.377 | 0.520 | 0.467 | 40 | 0.77 GB |

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

**Result: no recipe beat the control.**

| run | K | ARC-Easy | ARC-Chall.* | HellaSwag* | GSM8K |
|---|---|---|---|---|---|
| forced | 1 / 2 / 4 | 0.630 / 0.620 / 0.610 | 0.387 / 0.383 / 0.370 | 0.523 / 0.520 / 0.513 | 0.390 / 0.430 / 0.270 |
| half | 1 / 2 / 4 | 0.637 / 0.640 / 0.640 | 0.387 / 0.387 / 0.387 | 0.523 / 0.523 / 0.523 | 0.470 / 0.477 / 0.480 |
| fastdial | 1 / 2 / 4 | 0.637 / 0.637 / 0.637 | 0.377 / 0.380 / 0.380 | 0.517 / 0.513 / 0.513 | 0.453 / 0.470 / 0.470 |

\*acc_norm. ±0.03 at 300 questions.

- **Training closes the dials itself.** "half" started at 0.5 and ended at 0.061 / 0.013 /
  0.006. "fastdial" ended at 0.028 / 0.007 / 0.005. Not a speed problem: the loss prefers
  the loop off.
- **Forced on, more passes hurt.** GSM8K 0.39 → 0.27 from K=1 to K=4.
- Memory and speed as in exp2: weights 1.11 GB; KV cache 0.25 / 0.35 / 0.56 GB and
  ~120 / 84 / 55 tok/s at K = 1 / 2 / 4.

**Review: problems with the experiment itself.**
1. **bf16 weights drop small updates.** Tested on a real Qwen layer: at lr 1e-4, 82% of
   weights change; at 2e-5, only 29%. The late part of every run (exp2 included) was mostly
   frozen. Dials are fp32, so "dials close" still holds. Fix: fp32 master weights, bf16 compute.
2. **No input injection.** On pass 2, layers 8–19 read layer 19's output instead of layer 7's,
   which they were never trained on. Huginn-style designs feed the prelude output into every
   pass. This is the likely reason forced passes hurt.
3. **No matched control.** Exp3 runs are 500 steps; the benchmarked control is 1000 steps.
4. **fastdial was redundant.** Exp2 dials could already move ~0.01/step.
5. **Benchmarks are too noisy** (±0.03) for these effect sizes. Held-out loss at each K is a
   cheaper, less noisy primary metric.
6. **Data doesn't reward thinking.** Web text; the 0.6B is already within ~0.3 KL of the 1.7B.

---

## 2026-11-22 — Stepping back: when does looping actually help?

**My thinking (loose, captured as-is).**
- Looping needs two things: the model needs enough reasoning ability to begin with, and
  then the looping itself needs to work.
- Maybe looping matters more for long-context work, because there's more to reason over.
  Long problems, not necessarily short ones.
- Is looping implicitly doing something like **compaction + continuation**, the way agentic
  systems do? Each pass compresses and carries the state forward, and the output is the end
  result of all those reasoning chains.
- Can we mirror that to work through a longer-form problem, and have the model **know when
  to exit early**, i.e. cut the loop?
- Does looping help embeddings? Could you embed something better because the model
  understands the surrounding context better?
- Improving on Qwen is hard. Maybe Qwen is fine as a base, and I should start from the 1.7B.
- Was yesterday more of a waste than anything?

**Synthesis.**
- **Depth, not length.** A loop gives each token more serial computation over the *same*
  context. It doesn't add context. What it should help with is problems that need *many
  sequential steps* (multi-hop, state tracking, arithmetic carries). Long-context tasks often
  contain many hops, which is why the two feel linked, but the variable to test is number of
  steps, not number of tokens. A short prompt with 10 hops should need more loops than a long
  prompt with 1.
- **Compaction + continuation is a good analogy.** The per-token hidden state is a fixed-size
  scratchpad: each pass reads it, refines it and writes it back, without growing. That's
  "latent chain of thought". The difference from agent compaction: nothing is emitted as
  tokens, and the state never gets larger, so its capacity is fixed.
- **Knowing when to exit is an existing research thread**: adaptive computation (ACT,
  PonderNet), learned early exit in Ouro/LoopLM. It's a natural step 2, once a loop that
  helps exists at all. There's nothing to exit early from yet.
- **Embeddings: not now.** Retrieval quality comes mostly from knowledge and contrastive
  data, so a loop has little to do there. The place where "better embeddings from more
  passes" makes sense is Phase 2 (protein models), where structure is iterative constraint
  satisfaction.
- **1.7B: not yet.** The failures were about method (opt-out dials, no input injection, bf16,
  data), not model size. A bigger model fixes none of them and makes each run ~3× slower.
  Memory: full fine-tuning 1.7B with fp32 weights + Adam is ~27 GB before activations,
  too tight on 32 GB. It fits with LoRA, an 8-bit optimizer, or training only the loop block.
- **Yesterday wasn't a waste, but each lesson cost too much.** It produced a working harness,
  showed that a loop the model can switch off gets switched off, found the bf16 bug and the
  missing input injection, and showed that retrofits need ~1B+ tokens. But it spent hours of
  slow benchmarks on questions a small synthetic task could answer in minutes.

**Minimum problem that tests all of the above.** Train small looped models (~10M params) from
scratch on a **multi-hop task with a controllable number of steps**, e.g. pointer chasing or
variable tracking (`a=3; b=a+2; c=b*2; ... ; what is c?`), with 1–N hops.
- Does a looped model match a non-looped one with the same total depth, at fewer weights?
- Does the number of loops needed grow with the number of hops?
- Train on ≤ 8 hops, test on 16: do more loops at test time solve longer chains?
- Then: can it learn to exit early on easy (few-hop) problems?
Minutes per run on the 5090, no retrofit damage, and the data rewards depth by design. Then
carry the working recipe (input injection, random K, exit) back to Qwen.

---

## 2026-11-22 — New direction: tiny models from scratch, staged goals

**My thinking.** Reset to the smallest problem that can answer the question, then climb:
1. **Can I build an arithmetic model at all?** A tiny transformer (~4–20M params) trained
   from scratch on mod-10 PEMDAS expressions, answering directly (no written-out steps).
2. **Does looping help on long step chains?** Train on depth ≤ 8, test on depth 9–16. Does
   a looped model match a deeper one with fewer weights, and do extra loops at test time solve
   deeper problems?
3. **Do "steps" give it the power to do algebra?** Middle-school algebra (e.g. solve
   `3x+5=20`). This is a change of goal, not just more of the same: algebra needs undoing
   operations and choosing what to do next, not only evaluating left to right. It may need a
   new model or tokenizer (variables, `=` on both sides), but the data can still be
   generated and checked automatically.

Each stage only starts if the one before says yes. This is the step-wise path from
"can it compute" to "does looping add reasoning power".

**Why small.** It's a wind tunnel, not a plane: minutes per run, a clean ~10% → ~100%
signal instead of ±0.03 benchmark noise, and the design rules learned (input injection,
random loop counts, test-time extra loops) carry back to Qwen later.

**Setup (stage 1–2).** Code in `src/looptf/toy/`: `data.py` (generator: exact op count and
exact depth, minimal PEMDAS parens, answer mod 10), `model.py` (prelude / looped core with
concat input injection / coda, RoPE, fp32 weights + bf16 compute), `train.py`. Depth, not
length, is the controlled variable, so train and test cover the same range of lengths.
Comparison: shallow (4 layers), looped (same 4.3M params, core run 1–8× in training), deep
(19M params, 18 layers).

---

## 2026-11-22 — Arithmetic stage, first runs (mod 7, generated on the fly)

**Setup.** Mod-7 PEMDAS (digits 1–6, answer 0–6, chance ≈ 0.14), single answer token read at
'='. Shallow = 4 layers (4.3M); looped = same weights, core run 1–8×; deep18 = 18 layers (19M).

**Without a curriculum, nothing learns past depth 2.** 20k steps on depths 1–8 mixed:
shallow 1.00 / 0.69 / chance from depth 3; looped 1.00 / 0.22 / chance. Most of each batch was
unlearnable, so no foothold.

**With a depth curriculum** (start at depth ≤ 2, add a level at 90% on the top depth):

| run (lr) | depth 2 mastered | depth 3 mastered | stopped at |
|---|---|---|---|
| shallow (1e-3) | 6,000 | never (0.61 at 40k) | 40k |
| looped 1–8 (1e-3) | 5,000 | never (0.29 at 31k; same at every K) | 40k |
| deep18 (1e-3) | not by 14.4k | — | stopped |
| looped fixed K=4 (1e-3) | 6,250 | not by 16k | stopped |
| **looped 1–8 (3e-4)** | **3,500** | **12,250** | stopped at 15k, on depth 4 |

**Read.** The learning rate was likely the main blocker, not depth: at 1e-3 even the 18-layer
deep model could not master depth 2, a classic sign of lr too high for a deep net (a looped
model at K=8 is ~20 layers deep). One seed, so to be confirmed. At lr 1e-3, the looped model
scored identically at every K, i.e. it was not using its extra passes.

**Why stopped.** Switched away from mod 7: answers that aren't normal arithmetic are hard to
read and reason about. Next datasets use real values.

## 2026-11-22 — Switch to real values

**My thinking.** Mod 7 is weird. Just use the real result: we're training the model to
actually do math, so large answers are fair. (Considered a ±1000 bound; decided against it.)

**Design.** Exact integer answers, no bound and no modulus. Leaves 1–9 (no 0). Measured answer
sizes: nested median 4 chars, max 10; chain-16 median 6, max 12. Answers are written digit by
digit with a stop token (`(8-3)*(4-6)+7=-3#`), so the output layer stays the vocabulary size
(46) rather than one class per number. Scored by exact match on the whole answer
(teacher-forced argmax at every answer position, which equals greedy decoding). Chance is ~0,
so the signal is cleaner than mod 7's 1/7 floor. Dropped the "deepest step must matter" check:
without mod 7 and zeros, shortcuts are rare. Train all models at lr 3e-4 on the frozen data.

---

## 2026-11-22 — Review fixes and learning-rate sweep, before the grid

**Review fixes.** (1) Depth 1–2 test problems all appeared in training (only 243 distinct
depth-1 problems). Now 20% of the distinct depth-1/2 problems are held out of training and the
test set uses only those: depth 1 tests 49 unseen problems, depth 2 tests 1,000; overlap is 0
at every depth. (2) Scoring uses all 1,000 test problems per depth. (3) Every test problem's
result is saved (`per_example.npz`) so accuracy can be split by answer length ("too many
steps" vs "numbers too big"). (4) Checked: hidden state stays stable over 16 passes at init;
all depth-16 cells (2×8, 4×4, 8×2, 16×1) cost ~31 ms/step, so the diagonal is compute-matched.

**LR sweep** (10k steps, curriculum, test accuracy on unseen depth-3 problems):

| model | 1e-4 | 3e-4 | 1e-3 |
|---|---|---|---|
| 2×1 shallow (4.4M) | 0.32 | **0.35** | 0.30 |
| 2×8 looped (4.4M) | 0.37 | **0.57** | 0.62 |
| 16×1 deep (19M) | 0.48 | **0.61** | 0.51 |

3e-4 is best or near-best everywhere, so the grid uses 3e-4 for all. Early hint (one seed,
short): the looped 4.4M model matches the 19M deep one at depth 3, and both beat shallow.

**Grid launched** (seed 0): core ∈ {2,4,8,16} × fixed loops (effective depth ≤ 16), plus
random-K runs 2×(1–8) and 4×(1–4). 40k steps, lr 3e-4, `data/nested`. Script:
`scripts/arith_grid.sh`; report: `scripts/arith_report.py` → `docs/results-arith-grid.md`.

---

<a id="log-grid"></a>

## 2026-11-22 — Grid results, seed 0 (curriculum, real-valued nested data)

Full tables: `docs/results-arith-grid.md`. Depth-3 test accuracy (unseen problems, 1,000 per
depth, ±0.02 at 95%), at each run's training loop count:

| core \ loops | ×1 | ×2 | ×4 | ×8 |
|---|---|---|---|---|
| 2 (4.4M) | 0.80 | 0.84 | 0.86 | 0.86 |
| 4 (6.4M) | 0.83 | 0.86 | 0.88 | |
| 8 (10.6M) | 0.81 | 0.88 | | |
| 16 (19.0M) | 0.88 | | | |

- **Same weights, more loops helps:** 2-core 0.80 → 0.86, saturating after ~4 loops.
- **Same compute, fewer weights is about as good:** effective depth 16: 2×8 0.86 vs 4×4 / 8×2 /
  16×1 0.88. Looped gets within ~0.02 of deep with 4× fewer weights. One seed.
- **Number size dominates errors:** depth 3, deep model: 1-digit answers 0.99, 2-digit 0.95,
  3-digit 0.75, 4-digit 0.40. The curriculum "stall at depth 3" was mostly big answers, which
  were only ~1% of training at depth ≤ 3.
- **Random loop counts hurt at this scale:** 2×(1–8) 0.83, 4×(1–4) 0.79; slower to master
  depth 2 (~7.5k vs ~5k steps). Fixed-K models only work at their training K.
- **Depth 4+ uninterpretable:** most runs reached depth 4 only in the last ~5k steps.
- Bug: checkpoints for these 12 runs weren't saved (subdirectory not created); fixed.

**Read.** First positive looping result: looping buys depth without weights, but the gain is
small because this task's bottleneck is mental arithmetic on big numbers, not step count.
Next: separate the two (D-007 digit order, D-008 balanced answer sizes + input-size dial),
mixed-depth training (D-006).

---

## 2026-11-22 — Separating "big numbers" from "many steps"

**Problem.** The grid showed errors are driven by answer size more than depth. In this data
big answers only appear with depth (depth 1 maxes out at 81), and were rare in training
(4-digit ≈ 1% of the depth ≤ 3 mix). So the loop's effect on depth is hidden behind a
number-size problem.

**Decisions.**
- D-006 (Walter): no curriculum gate. Train on depths 1–8 mixed; judge per-depth test accuracy
  with 95% confidence intervals.
- D-008 (Walter): new dataset `data/nested_bal`: balanced answer sizes, multi-digit inputs at
  depth 1–3 (`347*28`) as a separate difficulty dial, test cells of up to 1,000 per
  (input size × answer size).
- D-007 (open): answers written normal (`258`) or reversed (`852`). Both are next-digit
  prediction, but in normal order the first digit needs every carry worked out in advance
  (mental arithmetic, no paper); reversed lets each digit be a small local step, with written
  digits acting as scratch paper. Plan: run both.

**Related work (checked 2026-11-22).**
- Lee et al., *Teaching Arithmetic to Small Transformers* (ICLR 2024,
  https://arxiv.org/abs/2307.03381): reversed output significantly improves small-transformer
  arithmetic, because normal order makes the model emit the most significant digit before the
  sum is known.
- McLeish et al., *Transformers Can Do Arithmetic with the Right Embeddings* (NeurIPS 2024,
  https://arxiv.org/abs/2405.17399): arithmetic failures stem largely from losing track of each
  digit's position within a long number. "Abacus" embeddings (a digit's place within its
  number) let 20-digit training generalize to 120 digits. They used **looped transformers with
  input injection** (our design); adding those raised out-of-distribution accuracy from 92.9%
  to 99.1%.
- Large math LLMs (Qwen-Math, DeepSeekMath) solve arithmetic by writing their working as text
  and calling tools, the visible-token alternative to looping; not the question here.

**Read.** These fixes target number representation; the loop targets step count. Borrowing
them removes the confound rather than replacing the loop. Model is d_model 256, 4 heads × 64.

---

## 2026-11-22 — D-007 wave 1: looped 2×8, normal vs reversed (balanced data, mixed depths)

**Rule check (pre-registered in plan.md).** Primary metric (1-digit inputs, 3–5 digit answers,
depths 2–6): normal 0.20, reversed 0.16 (−0.04; adopt needed ≥ +0.05). Small answers (1–2
digits, depths 2–6): normal 0.82, reversed 0.71 (−0.11; allowed ≥ −0.03). **Keep normal order**
for the looped model. One seed; shallow/deep pending. Contrary to Lee et al.; guess: their gain
relied on aligned digits in simple addition, which nested expressions lack. Reversal may only
help together with place-value embeddings (untested hypothesis).

**Clean depth effect, separated from number size** (looped, normal, 1-digit inputs):

| depth | 1-digit ans | 2-digit | 3-digit | 4-digit |
|---|---|---|---|---|
| 2 | 1.00 | 1.00 | 0.98 | |
| 3 | 0.99 | 0.88 | 0.47 | 0.23 |
| 4 | 0.90 | 0.66 | 0.22 | 0.05 |
| 6 | 0.84 | 0.54 | 0.15 | 0.04 |
| 8 | 0.74 | 0.39 | 0.12 | 0.03 |
| 10 (untrained) | 0.62 | 0.25 | 0.09 | 0.01 |
| 13 (untrained) | 0.44 | 0.09 | 0.03 | 0.01 |

- At fixed answer size, accuracy falls steadily with depth (step-count effect now visible).
- At fixed depth, accuracy collapses with answer size (number size still dominates).
- Graceful extrapolation past training depth on small answers (0.62 at depth 10).
- Like-for-like vs the grid: depth 3, 2-digit answers 0.88 vs 0.94; balanced data has fewer
  easy examples.
- 3-digit × 3-digit multiplication (depth 1): 0.17–0.32.

---

## 2026-11-22 — D-007 wave 2: shallow baseline. Looping buys depth, not arithmetic

Same 4.4M weights, normal order, 1-digit inputs (seed 0):

| answer size | depth | shallow 2×1 | looped 2×8 | loop gain |
|---|---|---|---|---|
| 1 digit | 4 | 0.74 | 0.90 | +0.16 |
| 1 digit | 6 | 0.58 | 0.84 | +0.26 |
| 1 digit | 8 | 0.47 | 0.74 | +0.27 |
| 1 digit | 10 (untrained) | 0.42 | 0.62 | +0.20 |
| 2 digits | 6 | 0.25 | 0.54 | +0.29 |
| 2 digits | 8 | 0.14 | 0.39 | +0.25 |
| 3–5 digits (depths 2–6) | | 0.17 | 0.20 | +0.03 |

**Read.** The clearest looping result so far: with the same weights, 8 passes add 16–29 points
on problems that need many steps but have small answers; the gain grows with depth to ~8 and
carries past the training depth. On big answers looping adds ~3 points. Loops buy sequential
depth, not arithmetic capacity: the project's hypothesis, visible now that the balanced data
separates the two. One seed; deep 16×1 (same compute, 4× weights) pending.

**Digit order settled:** reversed also hurts shallow (primary −0.04, small −0.08), so it is a
general effect, not loop-specific. Keep normal order (D-007).

---

## 2026-11-22 — D-007 wave 3: deep 16×1. Loops match depth; weights buy arithmetic

Normal order, 1-digit inputs, seed 0 (full tables: `docs/results-arith-bal.md`):

| model | weights | small ans, depth 2–6 | big (3–5d), depth 2–6 | multi-digit inputs | 1-digit ans @ depth 4 / 6 / 8 / 10 / 13 |
|---|---|---|---|---|---|
| shallow 2×1 | 4.4M | 0.66 | 0.17 | 0.21 | 0.74 / 0.58 / 0.47 / 0.42 / 0.35 |
| looped 2×8 | 4.4M | 0.82 | 0.20 | 0.28 | 0.90 / 0.84 / 0.74 / 0.62 / 0.44 |
| deep 16×1 | 19.0M | 0.84 | 0.29 | 0.34 | 0.91 / 0.83 / 0.72 / 0.61 / 0.45 |

**Read.** At equal compute, the looped model matches the deep one on many-step problems
(depth curves within ±0.02 at every depth, including untrained depths 10 and 13) with 4× fewer
weights. On big answers and multi-digit inputs, deep's extra weights help (+0.09 / +0.06 over
looped) while loops barely do (+0.03 over shallow). Interpretation: **loops supply sequential
depth; unique weights supply arithmetic capacity** (facts and procedures for big numbers). One
seed; needs confirmation. Reversed order: worse on the primary metric for all three (deep
0.22 vs 0.29), though deep-reversed is better on multi-digit inputs (0.40 vs 0.34) and depth 13
(0.52 vs 0.45): a hint that reversal + capacity interact, worth seeing under D-011.

---

## 2026-11-22 — D-011 screen, shallow pair: place-value tags

**Pre-registered rule (1-digit-input problems):** tags vs no tags (normal order): big answers
−0.01, small +0.02 → no clear effect, don't adopt. **Metric mistake (Claude):** with 1-digit
inputs every input digit is a "ones" digit, so the tags carry almost no information there; the
rule could not detect the effect it was testing. Should have used multi-digit inputs.

**Exploratory (found after the rule said no; needs a second seed):**

| shallow 2×1 | no tags, normal | no tags, rev | tags, normal | tags, rev |
|---|---|---|---|---|
| depth 1, 3-digit inputs (all) | 0.14 | 0.26 | 0.18 | **0.43** |
| … 3-digit answers | 0.38 | 0.28 | 0.42 | **0.89** |
| … 4-digit answers | 0.16 | 0.15 | 0.17 | **0.69** |
| depth 2, 2-digit inputs | 0.35 | 0.30 | **0.43** | 0.33 |
| depth 3, 2-digit inputs | 0.16 | 0.13 | 0.17 | 0.14 |

- Tags + reversed together nearly solve a single big multiplication (neither alone does): the
  pairing reported in the literature.
- Only for one step: at depth ≥ 2 reversed loses again; tags + normal is best at depth 2.
- Read: shallow can use tags on big inputs only for short problems; carrying big-number work
  through several steps is where the looped model should help (looped pair running).

**Rule amended before the looped results:** for D-011, multi-digit-input problems (depths 1–3,
by input size) are a co-primary metric with the same ≥ 0.05 bar.

---

<a id="log-tags"></a>

## 2026-11-22 — D-011 screen, looped pair: tags on, normal order (→ D-012 launched)

Looped 2×8 at K=8, seed 0, 40k steps. Differences with 95% bootstrap intervals (test-set noise
only; seed noise still unmeasured).

| looped 2×8 | no tags | tags, normal | tags, rev |
|---|---|---|---|
| big answers (3–5 d), 1-digit inputs, depths 2–6 | 0.20 | **0.26** | 0.17 |
| small answers (1–2 d), 1-digit inputs, depths 2–6 | 0.82 | **0.83** | 0.77 |
| all 1-digit-input problems, depths 1–3 | 0.79 | **0.85** | 0.74 |
| multi-digit-input problems, depths 1–3 | 0.28 | 0.31 | **0.35** |
| depth 1, 3-digit inputs | 0.16 | 0.19 | **0.48** |
| depth 2, 2-digit inputs | 0.46 | **0.53** | 0.45 |
| 1-digit answers, depth 13 | 0.44 | **0.51** | 0.50 |

- **Tags, normal vs no tags:** big +0.055 [+0.044, +0.065], small +0.006, multi-digit inputs
  +0.035 [+0.028, +0.043]. Passes the pre-registered rule (≥ 0.05 big, ≤ 0.03 cost) → **adopt tags.**
  Unlike shallow (−0.01), the looped model can use the tags on big answers.
- **Tags, reversed vs tags, normal:** multi-digit inputs +0.033 (< 0.05 bar), 1-digit inputs
  −0.110 (> 0.03 cost). Fails both halves of D-012's rule → **normal order.** Same pattern as
  shallow: reversed + tags nearly solves a single big multiplication (depth 1, 3-digit inputs
  0.16 → 0.48) and loses everywhere with more steps.
- Looped − shallow with tags (normal): big +0.09, small +0.15, multi-digit inputs +0.06
  (without tags: +0.03 / +0.16 / +0.06). With tags the loop starts to help on big answers;
  elsewhere the gap is unchanged.
- Still weak: big answers are at 0.26 at best. Tags help a little; they don't fix number size.

**D-012 launched 21:40** with `--place_emb` (normal order): `results/toy/long/`, queue log
`results/toy/long.queue.log`. Analysis script: `scripts/d011_compare.py`.

<a id="log-long-shallow"></a>

## 2026-11-22 — D-012 shallow pair (120k steps, tags, normal): training longer helps a lot

| shallow 2×1 + tags | 40k s0 | 120k s0 | 120k s1 |
|---|---|---|---|
| big answers, 1-digit inputs, depths 2–6 | 0.16 | 0.25 | 0.26 |
| small answers, 1-digit inputs, depths 2–6 | 0.68 | 0.78 | 0.79 |
| multi-digit inputs, depths 1–3 | 0.25 | 0.40 | 0.38 |
| depth 1, 3-digit inputs | 0.18 | 0.44 | 0.41 |
| 1-digit answers, depth 8 / 10 / 13 | 0.49 / 0.44 / 0.35 | 0.62 / 0.55 / 0.42 | 0.63 / 0.56 / 0.43 |

- **First seed-to-seed spread:** 0.01–0.03 on these slices. The 0.05 bar was about right.
- 3× training lifts the shallow model by +0.08 to +0.26. The 40k runs were well short of done.
- **Caution for the loop claim:** shallow at 120k (big 0.25, small 0.78) ≈ looped at 40k
  (0.26, 0.83). The 40k loop gap may be partly "learns faster", not "can do more". The looped
  120k pair decides this. If looped at 120k doesn't stay clearly ahead, the headline changes.
- Train accuracy at the end is still 0.42 (loss 0.68): not converged.

<a id="log-long-looped"></a>

## 2026-11-23 00:30 — D-012 looped pair (120k): the loop gap is real, and grows with depth

| tags, normal order | shallow 120k s0 / s1 | looped 120k s0 / s1 | looped − shallow (s0 / s1) |
|---|---|---|---|
| big answers, 1-digit inputs, depths 2–6 | 0.25 / 0.26 | 0.37 / 0.41 | +0.13 / +0.15 |
| small answers, 1-digit inputs, depths 2–6 | 0.78 / 0.79 | 0.91 / 0.93 | +0.13 / +0.15 |
| multi-digit inputs, depths 1–3 | 0.40 / 0.38 | 0.43 / 0.45 | +0.03 / +0.07 |
| 1-digit answers, depth 4 | 0.84 / 0.85 | 0.96 / 0.96 | +0.12 / +0.11 |
| … depth 8 | 0.62 / 0.63 | 0.86 / 0.87 | +0.24 / +0.25 |
| … depth 13 (past training depth 8) | 0.42 / 0.43 | 0.62 / 0.60 | +0.21 / +0.17 |

- **Answers the worry from the shallow pair:** shallow at 120k ≈ looped at 40k, but looped also
  improves with training and stays far ahead. The gap is "can do more", not just "learns faster".
- **The gap grows with step count:** +0.11 at depth 4 → +0.25 at depth 8, on both seeds.
- **New: loops now help big answers too** (+0.13/+0.15, was +0.03 at 40k without tags).
- Multi-digit inputs: small, noisy gain (+0.03/+0.07); input size is still mostly a weights issue.
- Seed spread for looped is wider on multi-digit-input slices (up to 0.07 at depth 1, 3-digit
  inputs); 0.01–0.04 elsewhere.
- Still undertrained? Train accuracy 0.50 at the end. Deep pair (same compute, 4× weights) next.

<a id="log-long-normal"></a>

## 2026-11-23 02:55 — D-012 normal arm complete: loops match depth on steps; weights win on number size

120k steps, tags, normal order, seeds 0 / 1. Shallow 2×1 (4.4M), looped 2×8 (4.4M), deep 16×1 (19M, same compute as looped).

| slice | shallow | looped | deep | deep − looped |
|---|---|---|---|---|
| 1-digit answers, depth 4 | 0.84 / 0.85 | 0.96 / 0.96 | 0.98 / 0.96 | +0.02 / +0.00 |
| … depth 8 | 0.62 / 0.63 | 0.86 / 0.87 | 0.88 / 0.86 | +0.02 / −0.02 |
| … depth 13 (past training) | 0.42 / 0.43 | 0.62 / 0.60 | 0.65 / 0.60 | +0.03 / −0.00 |
| small answers, depths 2–6 | 0.78 / 0.79 | 0.91 / 0.93 | 0.94 / 0.93 | +0.03 / −0.01 |
| 3-digit answers, depths 3–6 | 0.32 / 0.33 | 0.52 / 0.59 | 0.65 / 0.62 | +0.13 / +0.03 |
| 4-digit answers, depths 3–6 | 0.15 / 0.18 | 0.29 / 0.32 | 0.41 / 0.40 | +0.12 / +0.08 |
| multi-digit inputs, depths 1–3 | 0.40 / 0.38 | 0.43 / 0.45 | 0.51 / 0.54 | +0.09 / +0.09 |
| depth 1, 3-digit inputs | 0.44 / 0.41 | 0.46 / 0.53 | 0.61 / 0.66 | +0.15 / +0.14 |

- **Step count: looped = deep** within ±0.03 at every depth, both seeds, with 4× fewer weights.
  Meets plan step 4's success bar (within 0.03 at depths 4–8) on the probe set.
- **Number size: deep > looped** by +0.05…+0.15 (big answers, multi-digit inputs), both seeds.
  Loops do help here over shallow (+0.13), but unique weights help more. Same split as at 40k,
  now on 2 seeds and 3× training: **loops buy steps, weights buy arithmetic.**
- Seed spread is largest on 3-digit answers for looped and deep (up to 0.10): the 0.05 bar is
  too tight for that slice; use both seeds.
- Reversed arm started 02:55 (shallow pair first).

## 2026-11-23 03:35 — D-012 reversed arm, shallow pair: reversed still loses

Shallow 2×1, tags, 120k, normal vs reversed, seeds 0 / 1 (reversed − normal):
- **Rule:** multi-digit inputs (depths 1–3) +0.01 / +0.03 (needs ≥ +0.05); 1-digit inputs
  (depths 1–3, same slice as the 40k screen) −0.10 / −0.10 (allowed ≤ −0.03). **Fails both.**
- Big answers −0.07 / −0.08; small answers −0.07 / −0.08.
- Reversed still wins only a single big multiplication: depth 1, 3-digit inputs +0.12 / +0.15.
- Longer training didn't change the 40k picture. Looped and deep reversed pairs still to come.

## 2026-11-23 05:50 — D-012 reversed arm, looped pair: closer call, still fails the rule

Looped 2×8, tags, 120k, reversed − normal, seeds 0 / 1:
- **Rule:** multi-digit inputs (depths 1–3) +0.038 / +0.062 (needs ≥ +0.05 on both: seed 0
  misses); 1-digit inputs (depths 1–3) −0.014 / −0.012 (allowed ≤ −0.03: passes). **Fails**,
  narrowly, on seed 0's gain.
- Very different from 40k (cost −0.110) and from shallow at 120k (cost −0.10): with loops and
  long training, reversed no longer breaks multi-step problems. Big answers −0.03 / −0.03;
  depth 1, 3-digit inputs +0.07 / +0.11; 1-digit answers at depth 8 +0.02 / +0.04.
- Read: for the looped model, order is close to a wash, a slight edge to reversed on big
  inputs. Not a reason to switch the default; noted as "order matters less with loops".

<a id="log-reversed"></a>

## 2026-11-23 08:15 — D-012 reversed arm, deep pair: rule says normal, but the trend flips with capacity

Deep 16×1, tags, 120k, reversed − normal, seeds 0 / 1:
- **Rule:** multi-digit inputs (depths 1–3) +0.025 / −0.006 (needs ≥ +0.05) → **fails; keep
  normal order** (all three models now fail the rule).
- But reversed **helps** deep elsewhere: 1-digit inputs depths 1–3 +0.02 / +0.02; big answers
  (3–5 digits, depths 2–6) **+0.13 / +0.10**; small answers +0.05 / +0.05; depth-8 1-digit
  answers +0.06 / +0.04. Odd one out: depth 1 with 3-digit inputs −0.08 / −0.14 (opposite of
  shallow, where reversed won that cell by +0.12…+0.15).
- **Pattern across the probe set (reversed − normal, big answers):** shallow −0.07/−0.08,
  looped −0.03/−0.03, deep +0.13/+0.10. On 1-digit inputs: −0.10, −0.01, +0.02. Reversed order
  pays off only with enough capacity; it costs the small models. Not covered by the rule's
  metric, so noted as a finding, not a recipe change. Possible follow-up: does looped catch up
  with more training?

<a id="log-h13-h14"></a>

## 2026-11-23 08:40 — H13 / H14 on the D-012 checkpoints (no training): weights buy number size, not lookup; operator mix doesn't matter

Full tables: `docs/results-ops.md`. New test-only sets `data/nested_add` (depth 1–15) and
`data/nested_sub` (depth 1–8), exactly 1–4-digit inputs, verified. Normal-order models, seeds 0 / 1.

**H13 (do loops match deep on pure add/sub?).** Pre-registered rule: deep − looped on 2–4-digit
inputs over all depths = add +0.024 / +0.033, sub +0.018 / +0.017 → "unclear". **That verdict is
a floor effect (Claude's rule-design mistake, same kind as D-011):** most cells are at ~0 for every
model, so averaging them in shrinks every gap. Where models are off the floor and in distribution
(depths 1–3, where training had multi-digit inputs):

| depths 1–3 | looped | deep | deep − looped |
|---|---|---|---|
| add, 2-digit inputs | 0.76 / 0.77 | 0.80 / 0.88 | +0.04 / +0.11 |
| add, 3-digit inputs | 0.30 / 0.28 | 0.35 / 0.42 | +0.05 / +0.14 |
| sub, 2-digit inputs | 0.77 / 0.83 | 0.81 / 0.84 | +0.04 / +0.01 |
| sub, 3-digit inputs | 0.21 / 0.29 | 0.36 / 0.41 | +0.15 / +0.12 |
| mixed data, same region | | | +0.07…+0.10 |

**Corrected after code review (depth-1 2-digit prompts are ~97% training items, so the table
above mixes recall with held-out problems).** Unseen prompts only, same region (depths 1–3,
2–3-digit inputs): add looped 0.44 / 0.43 vs deep 0.49 / 0.58 (**+0.05 / +0.15**); sub 0.39 / 0.48
vs 0.51 / 0.55 (**+0.11 / +0.08**). Conclusion unchanged. (Reversed-order arm, unseen: add
deep − looped −0.10 / −0.13, i.e. looped wins; sub +0.15 / +0.02. Noted, not explained.)

→ **H13 rejected in practice:** deep's edge on big numbers is as large without `*` as with it.
"Weights buy arithmetic" is about **number size**, not multiplication lookup. (Contrast McLeish:
their looped models matched depth on addition with far more training and length-generalizing tricks.)

**Out-of-distribution findings (all models):**
- **4-digit inputs: 0% everywhere, even a single `a+b`.** Training never had 4-digit inputs; no
  input-length generalization at all. Motivates McLeish's random tag offset (H18 / I-M2).
- Multi-digit inputs beyond depth 3 (never in training) collapse: add 2–4-digit inputs at depth 5:
  looped 0.04–0.05, deep 0.08–0.12. Depth generalization does not carry over to bigger numbers.
- 1-digit sub (in distribution): looped ≈ deep (0.81/0.82 vs 0.82/0.83), %Gap 0.97 — the
  step-count result again.

**H14 (operator mix).** Within cells matched on depth, answer digits and input digits, low − high
`*`-share third: shallow −0.006 / −0.011, looped +0.001 / +0.015, deep +0.008 / +0.009 →
**not supported.** Unmatched, high-`*` problems look much harder (looped 0.50 → 0.19), but that is
entirely answer size and depth. Operator type adds nothing once those are fixed.

<a id="log-4digit"></a>

## 2026-11-23 09:10 — Why 4-digit inputs score 0%: the place-value tags break out-of-range numbers

Greedy decoding of 300 random `a+b` with 4-digit a, b (`scripts/diag_4digit.py`; no training run).

| model | 3-digit `a+b` exact | 4-digit exact | right length | first 2 digits right | within 5% |
|---|---|---|---|---|---|
| looped, **no tags** (bal, 40k) | 0.20 | 0.00 | 0.98 | 0.54 | **0.92** |
| deep, **no tags** (bal, 40k) | 0.24 | 0.01 | 0.98 | 0.55 | **0.93** |
| looped, tags (long, 120k) | – | 0.00 | 0.16 | 0.05 | 0.01 |
| deep, tags (long, 120k) | 0.74 | 0.00 | 0.63 | 0.07 | 0.10 |

- Without tags, 4-digit sums come out **approximately right** (e.g. 6268+9071=15339 → 15765):
  right length, right leading digits, wrong low digits. Exactness fails, magnitude generalizes.
- With tags, the answer is **garbage** (15339 → 7539): wrong length, wrong digits.
- So tags trade robustness for exactness: they lift in-range 3-digit sums 0.24 → 0.74 but wreck
  anything longer than seen. (Confound: no-tag models had 40k steps, tag models 120k. The gap is
  too large for training length to explain, but a clean no-tag 120k model would settle it.)
- **Likely mechanism (untested):** the tags are *absolute*: a question digit gets id 1 + its place
  from the end, and one embedding table serves question and answer digits. In training, question
  digits only ever had ids 1–3; id 4 appeared only on the 4th *answer* digit. A thousands digit in
  the question therefore arrives labelled "4th answer digit". This is the known weakness of
  absolute position embeddings (McLeish §3.1); their fix is a random offset during training.
- Fix candidates (I-M2 / H18): (a) random offset on tags; (b) separate tag tables for question and
  answer; (c) a few 4–5-digit inputs in training (tests interpolation, not extrapolation).

<a id="log-h19"></a>

## 2026-11-23 14:20 — H19 probes: no one-level-per-pass staircase; only levels 1–3 are readable

Linear probes (D-013 option b) read the last digit of every sub-expression's value from the hidden
state at the token right after it ends, after each pass (looped) or layer (deep, shallow).
~5,600 test problems (1-digit inputs, depths 2–8), 20% held out by problem.
`scripts/probe_levels.py`, results `results/toy/probe/levels.json`.

| level (height) | best accuracy: looped s0 / s1 | deep s0 / s1 | shallow | guessing |
|---|---|---|---|---|
| 1 | 0.70 / 0.67 | 0.67 / 0.57 | 0.70 | 0.14 |
| 2 | 0.33 / 0.38 | 0.47 / 0.33 | 0.28 | 0.16 |
| 3 | 0.27 / 0.31 | 0.33 / 0.25 | 0.23 | 0.18 |
| 4–7 | ≤ +0.05 over guessing everywhere | | | 0.16–0.19 |

- **Rule (pre-registered): not supported.** Looped "ready pass" for levels 1–5: 0, 2, 4, 2, 4
  (s0, Spearman 0.74) and 0, 6, 7, 4, 3 (s1, 0.10). But levels 4–5 are never readable, so their
  ready pass is noise: the rule again included floor cells (same mistake as H13).
- **Where readable (levels 1–3):** level 1 is readable *before* the first pass (the prelude layer
  does it) and fades over passes; levels 2 and 3 rise together and slowly (looped s0 peaks ~pass 4,
  s1 still rising at pass 8). Deep shows the same order over layers (level 1 by ~layer 7, levels 2–3
  by ~9–12). An ordering, not one level per pass.
- **Read:** the models don't hold intermediate values as linearly readable digits where each
  sub-expression ends, yet they answer depth-4 problems at 0.96. The work is elsewhere (later
  positions, e.g. at '='), or in a non-linear form, or the last digit of big intermediates is the
  wrong target. This probe can't answer H19 beyond level 3.
- Possible next probes (not planned): read every sub-expression value at the '=' token; non-linear
  (small MLP) probes; sign / small-value targets; the per-pass attention maps already saved in
  `results/toy/probe/attn_*.json`.
- Practical: with a 6-thread cap the probes took ~45 s per model vs ~8 min uncapped (thread
  oversubscription with sklearn's BLAS).

<a id="log-h18"></a>

## 2026-11-23 15:20 — H18 screen: the random offset fixes the nonsense, separate tags don't

40k steps, seed 0, normal order. Full tables: `docs/results-tags.md` (`scripts/tags_report.py`).

| looped 2×8 | 4-digit `a+b` within 5% | 4-digit exact | in-range 3-digit `a+b` exact | trained sizes: multi-digit inputs / big answers |
|---|---|---|---|---|
| **tags + random offset** | **0.97** | 0.01 | 0.21 | 0.29 / 0.25 |
| tags, separate tables | 0.00 | 0.00 | 0.15 | 0.30 / 0.26 |
| tags, no fix | 0.03 | 0.00 | 0.14 | 0.31 / 0.26 |
| no tags | 0.90 | 0.00 | 0.16 | 0.28 / 0.20 |

Shallow: offset 0.23, separate 0.01, tags 0.07, no tags 0.78 (within 5%).

- **Mechanism confirmed:** training the question-digit tags on ids beyond 3 (offset) turns nonsense
  into near-misses (0.03 → 0.97 within 5%, better than no tags). Separating question and answer
  tables does nothing (0.00). The failure was *untrained labels*, not the question/answer collision.
- **No cost on trained sizes:** offset is within 0.02 of plain tags on multi-digit inputs and big answers.
- **Exact 4-digit stays ~0 for every variant, but the bar was out of reach:** these 40k models get
  only 0.14–0.21 of *in-range* 3-digit sums exactly right. The D-014 rule (≥ 0.5 exact) couldn't be
  met at 40k by any extrapolation fix; it should have been relative to in-range accuracy (rule-design
  mistake, Claude). By the rule as written: offset "approximate only", separate "fail".
- Card says "approximate only → report, don't adopt". Offset nonetheless dominates plain tags
  (same in-range, 0.97 vs 0.03 out of range). Whether to make it the default is Walter's call.

<a id="log-attn-viewer"></a>

## 2026-11-23 16:30 — Attention viewer: look first, no hypothesis built in

Walter wanted to *see* where the models look, not test an expected pattern (H19 tested one).
`scripts/attn_export.py` runs 9 problems through looped, deep and shallow (120k, seed 0), using
each model's own greedy answer (all 27 correct). It records every attention call in order:
prelude, each core block on each pass, coda. Page: `docs/visual/attention.html`, data
`docs/visual/attn/*.json` (weights stored as one base-36 digit each, about 3% resolution).
Artifact: https://claude.ai/artifact/BrSUsELtTfMcQZrj6rLxbp.

First look, one problem `(8-3)*(4-6)+7=` (anecdote, not a result):
- The `=` that writes the first answer digit looks mostly at itself, the `7` and the `+`, not into
  the brackets. Bracket work probably happens earlier, at the `)` tokens.
- Looped heads keep one job across passes (block 2 head 4 fixes on the `7` every pass). Deep's
  heads change from layer to layer. Expected: looped reuses the same weights.

How the loop feeds itself (asked by Walter): concat injection. Each pass starts from
`adapter([h, e])`: `h` is the previous pass's output (after core block 2) and `e` is the
prelude output (the embedded problem), the same on every pass. So the model sees both its
running state and the original problem each pass. This led to H20
([idea](ideas.md#idea-intermediate)).

<a id="log-curves"></a>

## 2026-11-24 — Training curves: nothing has converged

Walter asked whether we ever trained to convergence. No. `scripts/curves_export.py` reads every
`log.jsonl`. Page: `docs/visual/training-curves.html`. On the 120k runs, held-out accuracy
(depths 1–8, at the training loop count) was still climbing almost linearly at the end. From 78k
to 120k: deep +0.12 / +0.11, looped +0.08 / +0.13, shallow +0.08 / +0.09 (seeds 0 / 1). Loss still
falling with the learning rate at a tenth of peak. Train vs test accuracy close (looped 0.52 vs
0.45), so it isn't memorizing. 120k × 256 is about 3 passes over the 10M training problems.
Consequence: every comparison so far is at a fixed budget. "Deep beats looped on numbers" may be
partly "deep learns faster". A run to convergence (one model of each kind) would separate the two.
H16 still compares at the same budget, so its rules stand, but read its result the same way.

<a id="log-h16"></a>

## 2026-11-24 — H16 started: wider blocks (D-015)

Widths 256 / 384 / 512 (feed-forward 4× width, 4 heads) × shallow 2×1, looped 2×8, deep 16×1. No
place-value tags, normal order, `data/nested_bal`, lr 3e-4, 120k schedule, seed 0. Width 256 is
rerun because no existing run matches (old 40k runs had a 40k schedule, old 120k runs had tags).
New `--pause_at` / `--resume` in `train.py` (test: paused + resumed = unbroken run). Each run
pauses at 40k and writes `checkpoints/toy/wide/<run>@40k.pt` and
`results/toy/wide/<run>/per_example@40k.npz`. Queue: `scripts/arith_wide.sh pause`, log
`results/toy/wide-pause.queue.log`. Speed at 512: about 0.17 s/step (2.4× width 256), so the 40k
stage takes about 5 h and the rest to 120k about 10 h. Continue with `scripts/arith_wide.sh resume`.

**Crash at the 40k pause (24 Nov, about 09:40).** Both width-512 runs reached 40k, then failed
to save: the pause code didn't create `checkpoints/toy/wide/` (run names contain `/`, and the test
used a name without one). About 2 h of GPU lost. Fix: create the folder (done by hand at once, so
the running 384 runs save fine) and in `train.py`. The test now uses a nested name. The 512 runs
rerun from scratch after the queue (`scripts/arith_wide_rerun512.sh`). Their first logs are kept in
`results/toy/wide-crash1/` (valid to 40k). Early read at 36k, exact accuracy depths 1–8: looped
512 0.33, deep 512 0.23, looped 256 (old, tags) 0.23. One seed, mid-training, and the no-tag 256
baseline isn't run yet, so this is a hint only.

**Plan change (24 Nov, Walter).** Continue all 9 to a second pause at 80k, then to 120k, without
waiting at each step. Added `--save_every 10000` (atomic state saves, resume drops log lines from a
crashed attempt) and tests for two pauses and for resuming from a periodic save. Scoring script
`scripts/wide_report.py`. Note on rule 2: at 40k, shallow at depth 8 is below 0.1 (floor), so read
rule 2 at 80k and 120k.

The 120k target from the old runs (tags on), exact: multi-digit inputs 0.40 / 0.44 / 0.52, big
answers 0.25 / 0.39 / 0.47 (shallow / looped / deep).

---

## Open questions / ideas to come back to

- **Number base as a difficulty dial.** Same problems written in base 2 / 10 / 16. Base 2 makes
  every digit step trivial but numbers ~3.3× longer with long carry chains (many easy steps in
  sequence); base 16 the opposite. Question: does looping help more for "many easy steps" than
  "few hard steps"? Needs place-value embeddings (base-agnostic) and a renderer change.

- **Loop placement (which blocks recur).** Intuition: early blocks read, middle blocks think,
  late blocks write, so loop the middle (prelude / core / coda, as Huginn does). Test at fixed
  4 unique blocks and fixed compute: 0+4+0 (loop everything, Universal-Transformer style),
  1+2+1 (current), 2+1+1, 1+1+2, 3+1+0. Cheap: `--n_prelude/--n_core/--n_coda` already exist.
  (Playbook's original exp4.) Priority 1 of the future-architecture ideas.
- **Adaptive depth / early exit.** Learn when to stop looping per problem or per token (ACT,
  PonderNet, Ouro's learned exit; Mixture-of-Recursions for per-token loop counts). Priority 2,
  once looping clearly helps.
- **Routing / branching.** Skip blocks per token (Mixture-of-Depths), route to experts (MoE),
  or build a per-problem network shaped like the expression tree (Neural Module Networks).
  Looped transformers ≈ iterated message passing (cf. recurrent GNNs in neural algorithmic
  reasoning / CLRS). Likely problem-specific (arithmetic has a clean tree); bigger build.
  Priority 3.

- **Side quest: look at the embedding / hidden-state space of these models, including per loop
  pass.** Curiosity, not on the critical path; unclear what it will teach. Possible angles: PCA /
  UMAP of the hidden state at the '=' token, coloured by answer, depth or loop pass; whether
  states converge to a fixed point as K grows; a "logit lens" (decode the answer from each
  pass's state) to see at which pass the answer appears, and whether harder problems need more
  passes before it does; how the digit embeddings arrange themselves (a number line?).

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
