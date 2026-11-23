# H19 results: does each loop pass handle one level of brackets?

**Question.** The looped model runs the same layers 8 times. Does each pass work through one
level of a nested problem, innermost brackets first?

**Method.** For every bracketed piece of a test problem (its *level* is how deeply it nests:
`8-3` is level 1, `(8-3)*(4-6)` is level 2), a probe (a tiny linear classifier) tries to read
the last digit of the piece's value from the model's internal state, at the token right after
the piece ends. We train one probe per pass (looped) or per layer (deep, shallow). If passes
handle levels, level 1 should become readable early, level 2 later, and so on: a staircase.
5,600 test problems with 1-digit inputs, depths 2–8; 20% held out to score the probes.
Decision card D-013; code `scripts/probe_levels.py`.

## Result

- **No staircase of one level per pass.** Levels 2 and 3 become readable together and gradually.
- **Only levels 1–3 can be read at all.** Levels 4 and up stay at guessing level in every model,
  even though looped answers depth-4 problems correctly 96% of the time. The intermediate
  values aren't stored as readable digits where each piece ends.
- **Level 1 is done before the loop starts** (by the single layer in front of it) and fades over
  the passes.
- **Deep shows the same order over its layers**, so this isn't specific to looping.
- By the pre-set rule H19 is *not supported*, but the rule counted unreadable levels 4–5, so the
  honest summary is *not shown*: this probe can't see past level 3.

Possible next probes: read every piece's value at the final `=`, use a small non-linear probe,
or study the per-pass attention maps saved in `results/toy/probe/attn_*.json`.

## Probe accuracy by level and pass

Each cell is the share of held-out pieces whose last digit the probe gets right. *Guess* is
always picking the most common digit. Column 0 is before the first pass (or first core layer).

### Looped, run 1 (8 passes)

| level | pass 0 | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 | pass 7 | pass 8 | best | guess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.66 | **0.70** | 0.66 | 0.62 | 0.59 | 0.56 | 0.54 | 0.53 | 0.52 | 0.70 | 0.14 |
| 2 | 0.16 | 0.21 | 0.31 | 0.32 | **0.33** | 0.31 | 0.29 | 0.28 | 0.27 | 0.33 | 0.16 |
| 3 | 0.18 | 0.19 | 0.22 | 0.23 | **0.27** | 0.25 | 0.25 | 0.23 | 0.24 | 0.27 | 0.18 |
| 4 | 0.19 | 0.20 | 0.21 | 0.21 | **0.23** | 0.22 | **0.23** | 0.21 | 0.21 | 0.23 | 0.19 |
| 5 | 0.16 | 0.18 | 0.17 | 0.17 | **0.21** | 0.21 | 0.19 | 0.18 | 0.19 | 0.21 | 0.18 |

### Looped, run 2 (8 passes)

| level | pass 0 | pass 1 | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 | pass 7 | pass 8 | best | guess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.62 | 0.65 | **0.67** | 0.64 | 0.60 | 0.57 | 0.57 | 0.56 | 0.56 | 0.67 | 0.14 |
| 2 | 0.16 | 0.23 | 0.26 | 0.30 | 0.32 | 0.34 | 0.36 | 0.37 | **0.38** | 0.38 | 0.16 |
| 3 | 0.18 | 0.20 | 0.24 | 0.24 | 0.25 | 0.26 | 0.28 | 0.30 | **0.31** | 0.31 | 0.18 |
| 4 | 0.19 | 0.19 | 0.24 | 0.22 | 0.27 | 0.25 | **0.28** | 0.28 | 0.27 | 0.28 | 0.19 |
| 5 | 0.18 | 0.19 | 0.21 | 0.22 | 0.21 | 0.21 | 0.22 | 0.22 | **0.24** | 0.24 | 0.18 |

### Deep, run 1 (16 core layers)

| level | layer 0 | layer 2 | layer 4 | layer 6 | layer 8 | layer 10 | layer 12 | layer 14 | layer 16 | best | guess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.35 | 0.45 | 0.52 | 0.58 | 0.66 | 0.67 | 0.66 | 0.65 | 0.64 | 0.67 | 0.14 |
| 2 | 0.16 | 0.18 | 0.20 | 0.26 | 0.36 | 0.45 | **0.47** | 0.46 | 0.46 | 0.47 | 0.16 |
| 3 | 0.17 | 0.18 | 0.16 | 0.18 | 0.25 | 0.30 | **0.33** | 0.32 | 0.32 | 0.33 | 0.18 |
| 4 | 0.18 | 0.18 | 0.18 | 0.19 | 0.22 | 0.22 | **0.25** | 0.24 | 0.24 | 0.25 | 0.19 |
| 5 | 0.16 | 0.16 | 0.18 | 0.19 | 0.16 | 0.18 | 0.21 | **0.22** | 0.21 | 0.22 | 0.18 |

### Deep, run 2 (16 core layers)

| level | layer 0 | layer 2 | layer 4 | layer 6 | layer 8 | layer 10 | layer 12 | layer 14 | layer 16 | best | guess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.35 | 0.53 | 0.46 | 0.48 | **0.57** | 0.56 | 0.55 | 0.54 | 0.54 | 0.57 | 0.14 |
| 2 | 0.15 | 0.17 | 0.20 | 0.22 | 0.30 | 0.32 | 0.32 | 0.33 | 0.33 | 0.33 | 0.16 |
| 3 | 0.18 | 0.18 | 0.16 | 0.17 | 0.22 | 0.22 | 0.24 | 0.24 | **0.25** | 0.25 | 0.18 |
| 4 | 0.18 | 0.18 | 0.18 | 0.17 | 0.20 | 0.20 | 0.21 | **0.22** | 0.21 | 0.22 | 0.19 |
| 5 | 0.16 | 0.18 | 0.16 | 0.16 | 0.15 | 0.15 | 0.18 | 0.17 | **0.21** | 0.21 | 0.18 |

### Shallow, run 1 (2 core layers)

| level | layer 0 | layer 1 | layer 2 | best | guess |
|---|---|---|---|---|---|
| 1 | **0.70** | 0.65 | 0.60 | 0.70 | 0.14 |
| 2 | 0.21 | 0.28 | **0.28** | 0.28 | 0.16 |
| 3 | 0.17 | **0.23** | 0.23 | 0.23 | 0.18 |
| 4 | 0.18 | 0.21 | **0.21** | 0.21 | 0.19 |
| 5 | 0.16 | **0.21** | 0.20 | 0.21 | 0.18 |

