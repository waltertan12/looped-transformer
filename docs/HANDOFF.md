# Handoff: read this first in a new session

_Written 2026-11-22 21:25 EST, updated 2026-11-23 15:00. Update it whenever the state below changes._

## What this project is
Walter (software engineer, newer to ML research) is testing whether **looped transformers**
(the same layers run several times) add reasoning depth. Goal: a **mini paper** with an
evidence-backed opinion about loops. One RTX 5090 (32 GB, 450 W cap) at home.

Current track: **tiny transformers trained from scratch on synthetic PEMDAS arithmetic**
(`src/looptf/toy/`). The earlier Qwen-retrofit track (exp1–3) is paused: it didn't work.

## Running now (2026-11-24)
Update 12:40: `scripts/arith_wide_chain.sh` (log `results/toy/wide-chain.queue.log`) waits for the
512 reruns to pause at 40k, then runs all 9 to an 80k pause (`arith_wide.sh resume80`), then to 120k
(`arith_wide.sh resume`). Walter approved both steps. State is also saved every 10k steps: after a
crash, rerun the same mode. Score each checkpoint with `scripts/wide_report.py 40k|80k|120k`.
Expected: 40k ≈ 14:30 (deep 512 restarted 12:43), 80k ≈ 21:00, 120k ≈ 00:30.
**Never run tests or evals on the GPU while trainings run** (two width-512 runs fill it; one OOM'd on 24 Nov).
**First run `scripts/lock.py show`.** It lists every queue running on this machine, from any
track (record in `.locks/running.json`, not committed). Register new queues with `scripts/lock.py add`.
Tracks in this repo: arithmetic (`src/looptf/toy/`, everything below) and protein (idea stage, `docs/protein.md`).

H16 / D-015: `scripts/arith_wide.sh pause` (9 runs, 2 at a time, log `results/toy/wide-pause.queue.log`).
The two width-512 runs crashed at the pause (fixed) and rerun afterwards via `scripts/arith_wide_rerun512.sh` (log `results/toy/wide-rerun512.queue.log`). Each run pauses at 40k. When all 9 say "done … (pause)": score the @40k results against the D-015
rules and show Walter. If he says continue: `setsid nohup scripts/arith_wide.sh resume > results/toy/wide-resume.queue.log 2>&1 &`.

## Where things are

| what | file |
|---|---|
| Code on GitHub (public, branch `main`) | https://github.com/waltertan12/looped-transformer |
| Public docs site (GitHub Pages from `main:/docs`, updates on push) | https://waltertan12.github.io/looped-transformer/ |
| Forward plan, with decision rules written **before** results | `docs/plan.md` |
| Hypotheses (status + Walter's ranking) | `docs/hypotheses.md` |
| Idea backlog (short / medium / long) | `docs/ideas.md` |
| Closest papers, compared with ours | `docs/related-work.md` |
| Every decision (cards D-001…D-012), who made it, outcome | `docs/decisions.md` |
| Reasoning and results, chronological | `docs/research-log.md` |
| State snapshots (newest first) | `docs/status.md` |
| Glossary: plain (for Walter) / terse (for agents) | `docs/glossary.md`, `docs/glossary-agent.md` |
| Visual results page (claude.ai artifact, private to Walter; republish to keep the URL) | https://claude.ai/artifact/BBw1f6Y9HGTgb75UBM1Nfj, source `docs/visual/loop-depth-results.html` |
| Attention viewer (artifact, private; publish from a scratchpad copy with `attn/*.json` as files) | https://claude.ai/artifact/BrSUsELtTfMcQZrj6rLxbp, source `docs/visual/attention.html`, data `scripts/attn_export.py` |
| Training curves page (artifact, private) | https://claude.ai/artifact/3gsN2gCjago7zG2AQs1eFA, source `docs/visual/training-curves.html`, data `scripts/curves_export.py` and `scripts/breakdown_export.py` (add/sub scores: `scripts/wide_ops.sh 40k|80k|120k`) → `docs/visual/curves/` (re-run to refresh H16). Local only: the JSON is gitignored |
| Skills | `.claude/skills/status-update`, `.claude/skills/decision-brief` (has a pre-launch checklist) |
| Working rules | `CLAUDE.md` (bottom section "Working with Walter") |

## Main results (as of 2026-11-23 15:00; full list in `docs/hypotheses.md`)
Shallow 2×1 (4.4M), looped 2×8 (4.4M, 8 passes), deep 16×1 (19M, same compute as looped).
120k steps, 2 seeds, place-value tags, normal answer order (D-012).
- **Loops buy steps:** looped beats shallow by +0.11 (depth 4) to +0.25 (depth 8) and matches
  deep within ±0.03, also past training depth (13).
- **Weights buy number size:** deep beats looped by +0.05…+0.15 on big numbers, with or without `*` (H13).
- **Operator mix doesn't matter** at fixed depth and number size (H14).
- **Tags break longer numbers:** 4-digit inputs 0% exact everywhere; with tags answers are
  nonsense, without tags within 5% (research log 09:10). → H18.
- **H19 not shown:** probes read only levels 1–3; no one-level-per-pass staircase
  (`docs/results-h19.md`).
- Fixed-K models fail at other loop counts. Reversed order: fails the rule, helps only deep.
- Prior work: Saunshi et al. ICLR 2025, McLeish et al. NeurIPS 2024 (`docs/related-work.md`).

## In flight right now
- **Nothing running.** H18 screen finished and scored (`docs/results-tags.md`, research log 15:20):
  random tag offset → 4-digit `a+b` within 5% 0.97 (plain tags 0.03, no tags 0.90), no cost in range;
  separate tables do nothing. Exact 4-digit ~0 for all (in-range 3-digit exact only 0.14–0.21 at 40k).

## Next actions, in order
1. **Walter decides:** adopt `--tag_offset 8` as the tag default? (The D-014 rule says "approximate
   only, don't adopt", but offset matches plain tags in range and fixes out-of-range nonsense.)
   If yes: card for the long probe set with it (shallow / looped / deep × 2 seeds, 120k, ≈ 6 h:
   `scripts/arith_long_arm.sh "--place_emb --tag_offset 8" offset`), which also shows whether exact
   4-digit accuracy rises once in-range exactness is higher.
2. Update the results page (`docs/visual/loop-depth-results.html` + claude.ai artifact
   BBw1f6Y9HGTgb75UBM1Nfj) with H19 and H18. v2 style: what / why / result / meaning, no
   semicolons, no H-codes in the body.
3. Then H9 (variable loop count, plan step 5).

## Practical notes
- Run Python as `/mnt/data/walter/loop-transformer/.venv/bin/python`. Tests: `pytest tests -q -k toy`.
- **Leave CPU headroom** for ssh and other work (Walter, 2026-11-23; not a strict half): cap CPU-heavy
  analysis (`threadpoolctl` + `torch.set_num_threads`, e.g. `--threads 6`–`12` of 24) and `nice -n 10`.
  sklearn/numpy otherwise grab every core.
- Run at most **2 trainings at a time**, each with `--mem_fraction 0.4`. GPU at ~94% use is normal.
- Don't use `pkill -f <pattern>`: it kills the calling shell. Get PIDs from
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader` or `ps -eo pid,args | grep "[p]attern"`.
- `rm -rf` on globbed paths is blocked by a safety check; move things to `results/toy/stopped/` instead.
- Headless Firefox screenshots hang on this machine; there's no node/playwright. Publish without a preview.
- `results/`, `checkpoints/`, `data/` are gitignored. Data rebuilds deterministically:
  `scripts/build_data.py --which nested_bal --out data` (~6 min), verify with `scripts/verify_data.py`.
- Push after committing: `git push` (origin = GitHub, SSH). Commit docs/code changes as you go; end commit messages with the attribution line in use.
