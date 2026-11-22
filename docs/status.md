# Status log

Structured snapshots of where the project is. Newest first. Written with the `status-update`
skill (`.claude/skills/status-update/SKILL.md`); the research log (`research-log.md`) holds
the reasoning, this file holds the state.

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
