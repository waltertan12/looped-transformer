# Status log

Structured snapshots of where the project is. Newest first. Written with the `status-update`
skill (`.claude/skills/status-update/SKILL.md`); the research log (`research-log.md`) holds
the reasoning, this file holds the state.

---

## 2026-11-22 19:57 EST

**Aim.** Unchanged (see 18:30): stage 2, does looping add reasoning depth.

**Done** (since 18:30)
- Sliced the D-007 probe-set results by depth × answer size × input size and published a
  visual report: https://claude.ai/artifact/BBw1f6Y9HGTgb75UBM1Nfj (source:
  `docs/visual/loop-depth-results.html`). New slices beyond 18:30:
  - Depth 1 → 8, 1-digit answers: shallow loses 0.53, looped 0.26, deep 0.28.
  - Deep − looped gain is concentrated on 3-digit answers at depths 3–6 (+0.13 to +0.30);
    near zero on 1–2 digit answers. Looped − shallow gain is on 1–2 digit answers, growing
    with depth.
  - Fixed-8-loop model at other test-time loop counts: ≤ 0.11 on small answers (vs 0.78 at
    8 loops, depth 4). Extra test-time loops need random-loop training.
  - Depth 1, 3-digit inputs (`639*53`): ≤ ~0.38 for every model on any answer size.

**Doing.** Nothing; GPU idle.

**Blocked / problems.** Same as 18:30 (big-number arithmetic; one seed).

**Next decisions.** D-011 (place-value embeddings; recommended option b, 4 runs ≈ 1.6 h), then a
second seed.

---

## 2026-11-22 18:30 EST

**Aim.** Stage 2: does looping add reasoning depth? Separate "many steps" from "big numbers",
then test whether a looped model matches a deeper one with fewer weights. Plan and decision
rules: `docs/plan.md`.

**Done** (since 14:15)
- Grid seed 0 finished (`docs/results-arith-grid.md`): at depth 3, looped 2×8 (4.4M) 0.86 vs
  deep 16×1 (19M) 0.88; errors dominated by answer size, not depth.
- Built `data/nested_bal` (2.6 GB): balanced answer sizes, multi-digit inputs at depth 1–3,
  ~1,000 test problems per (depth × input size × answer size) cell, zero overlap.
- Workflow: decision cards (`docs/decisions.md`, D-001…D-011), plan with pre-registered rules,
  glossary (plain + agent), pre-launch checklist; place-value embeddings implemented (off).
- **D-007 probe set finished** (shallow / looped / deep × normal / reversed, mixed depths 1–8,
  seed 0; `docs/results-arith-bal.md`). Normal order, 1-digit inputs:

  | model | weights | small answers, depth 2–6 | big (3–5 digit), depth 2–6 | multi-digit inputs | 1-digit ans @ depth 8 / 10* / 13* |
  |---|---|---|---|---|---|
  | shallow 2×1 | 4.4M | 0.66 | 0.17 | 0.21 | 0.47 / 0.42 / 0.35 |
  | looped 2×8 | 4.4M | 0.82 | 0.20 | 0.28 | 0.74 / 0.62 / 0.44 |
  | deep 16×1 | 19.0M | 0.84 | 0.29 | 0.34 | 0.72 / 0.61 / 0.45 |

  (*never trained on.) **Looping matches depth, weights buy arithmetic:** on many-step
  problems the looped model equals the deep one with 4× fewer weights (and beats shallow by up
  to +0.27); on big answers the deep model's extra weights help (+0.09 over looped) and loops
  barely do (+0.03 over shallow).
- Digit order decided: reversed worse for all three on the primary metric → keep normal (D-007).

**Doing**
- Nothing on the GPU (idle). Waiting on D-011.

**Blocked / problems**
- Big-number arithmetic is the remaining bottleneck (3–5 digit answers ≤ 0.29 for every model).
- One seed for everything; the looped ≈ deep result needs a second seed before it's a claim.
- Small cost from balanced data on easy cases (depth 3, 2-digit answers 0.88 vs 0.94 on grid).

**Next decisions**
1. **D-011 (needs Walter):** place-value embeddings: (a) normal on probe set, 3 runs ≈ 1.3 h;
   (b) (a) + looped reversed, 4 runs ≈ 1.6 h (recommended); (c) both orders, 6 runs ≈ 2.5 h.
2. Second seed of the normal-order probe set (≈ 1 h) to confirm looped ≈ deep on depth.

---

## 2026-11-22 14:15 EST

**Aim.** Find the smallest setting where looping clearly helps. Stage 1: can a tiny transformer
learn arithmetic from scratch? Stage 2: does looping add reasoning depth: does a looped model
with fewer weights match a deeper one, and do extra loops solve longer problems? Stage 3
(later): does that carry over to middle-school algebra?

**Done**
- Qwen track (exp1–3) closed: retrofitting a loop onto pretrained Qwen3-0.6B didn't help.
  Lessons logged (opt-out dials, bf16 bug, no input injection, data didn't reward depth).
- From-scratch setup in `src/looptf/toy/`: tiny transformer with a loopable core and input
  injection; frozen datasets `data/nested` (2.3 GB, 20M problems) and `data/chain` (2.0 GB,
  7M); exact real-valued answers written as digits + `#`; zero test/train overlap at every
  depth; independent verifier passes; 52 tests pass.
- Learning-rate sweep: 3e-4 best or near-best for shallow, looped and deep.
- Mod-7 and generate-on-the-fly runs: superseded; showed a curriculum and lower lr are needed.

**Doing**
- Grid seed 0 (12 runs, ~75% through the first 6): core blocks {2,4,8,16} × loops, plus
  random-loop runs. Early read at depth 3 (unseen problems, at each run's training loops):

  | run | weights | eff. depth | depth 3 | depth 4 |
  |---|---|---|---|---|
  | 16×1 deep | 19.0M | 16 | 0.83 | 0.07 |
  | 2×8 looped | 4.4M | 16 | 0.81 | 0.12 |
  | 4×4 looped | 6.5M | 16 | 0.81 | 0.10 |
  | 8×2 looped | 10.7M | 16 | 0.78 | 0.09 |
  | 2×(1–8) random | 4.4M | ≤16 | 0.76 (best K) | 0.08 |
  | 4×(1–4) random | 6.5M | ≤16 | 0.74 (best K) | 0.06 |

  → On the depth-16 diagonal, the 4.4M looped model matches the 19M deep one. One seed;
  not yet final.

**Blocked / problems**
- **Curriculum stalls at depth 3 in every run.** No run reaches 90% exact match on depth-3
  training problems (they plateau ~0.85–0.9), so none ever trains on depth 4+. Depth ≥ 4
  scores (~0.1) therefore measure "never-seen depth", not looping. The grid can only compare
  models at depth ≤ 3 this round.
- Fixed-loop models only work at their training loop count (e.g. 2×8 scores ~0 at K=4);
  random-loop models are stable across K, but extra loops don't help yet (flat at depth 3).

**Next decisions**
1. Let the remaining 6 grid cells finish (~1.5 h) so the depth-3 comparison is complete, or
   stop now and fix the curriculum first.
2. Curriculum fix: lower the threshold (e.g. 0.8) or advance after a plateau; then rerun the
   grid so models actually train on depths 4–8.
3. Check `per_example.npz` at the end: are depth-3 failures the big-number problems
   (answer length) rather than step count?
