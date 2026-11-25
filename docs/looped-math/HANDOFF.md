# Handoff: read this first in a new session

_Rewritten 2026-11-24 19:55 EST (docs reorganized). Update it whenever the running state changes.
Older updates are in `status.md` and the git history._

## What this project is
Walter (software engineer, newer to ML research) is testing whether **looped transformers**
(the same layers run several times) add reasoning depth. Goal: a **mini paper** with an
evidence-backed opinion about loops. One RTX 5090 (32 GB, 450 W cap) at home.
Tiny transformers trained from scratch on synthetic PEMDAS arithmetic (`src/looptf/toy/`).
The first approach (Qwen retrofit) is stopped: `history/qwen-retrofit/`.

## Running now: H16 / D-015 (wider blocks), 9 runs, seed 0
- `scripts/arith_wide_chain.sh` runs all 9 runs, 2 at a time. Log: `results/toy/wide-chain.queue.log`.
  Order: all 9 to an 80k pause (`arith_wide.sh resume80`), then add/sub scoring
  (`scripts/wide_ops.sh 80k`), then all 9 to 120k (`arith_wide.sh resume`), then `wide_ops.sh 120k`.
  Walter approved both stages.
- State at 19:40: looped and deep at all 3 widths are paused at 80k. The 3 shallow runs train to
  80k now. ETA for 120k on all 9: about 03:00 on 25 Nov (rough).
- State is saved every 10k steps. If a run crashes, run the same mode again.
- Curves page auto-rebuild: Walter asked for a rebuild at 60k, 80k, 100k and 120k. A Claude session
  does this with a watcher in its scratchpad (not durable). If no session does it, rebuild by hand:
  run `scripts/curves_export.py` and `scripts/breakdown_export.py` (CPU only), then republish
  `docs/visual/training-curves.html` with `curves/*.json` as files.
- Do not edit `scripts/arith_wide_chain.sh` while it runs. Bash reads the file as it runs.
- Do not move `results/toy/`, `checkpoints/toy/`, `src/looptf/toy/` or `scripts/` while it runs.

## When the chain finishes
1. Fix the wait loop (also listed in `docs/housekeeping.md`) in `scripts/arith_wide_chain.sh`: `pgrep -f "pause_at 40000"` matches any shell
   whose command line contains that text. On 24 Nov it hung the chain from 13:45 to 15:29.
2. Rerun `scripts/wide_ops.sh 40k` with the GPU free. `c2k8-w512-s0@40k-sub` FAILED. The cause is
   not known, because the script sends its output to /dev/null.
3. Score 120k against the D-015 rules: `scripts/wide_report.py 120k`. Show Walter. Write the log
   entry, `results/`, and the verdict in `hypotheses.md`.

## Where things are

| what | file |
|---|---|
| Code on GitHub (public, branch `main`) | https://github.com/waltertan12/looped-transformer |
| Public docs site (GitHub Pages from `main:/docs`, updates on push) | https://waltertan12.github.io/looped-transformer/ |
| Forward plan, with decision rules written **before** results | `docs/looped-math/plan.md` |
| Hypotheses (status + Walter's ranking) | `docs/looped-math/hypotheses.md` |
| Idea backlog (short / medium / long) | `docs/looped-math/ideas.md` |
| Closest papers, compared with ours | `docs/looped-math/prior-art/related-work.md` |
| Every decision (cards D-001…D-015), who made it, outcome | `docs/looped-math/decisions.md` |
| Reasoning and results, chronological | `docs/looped-math/research-log.md` |
| State snapshots (newest first) | `docs/looped-math/status.md` |
| Glossary: plain (for Walter) / terse (for agents) | `docs/glossary.md`, `docs/glossary-agent.md` |
| Visual results page (claude.ai artifact, private to Walter; republish to keep the URL) | https://claude.ai/artifact/BBw1f6Y9HGTgb75UBM1Nfj, source `docs/visual/loop-depth-results.html` |
| Attention viewer (artifact, private; publish from a scratchpad copy with `attn/*.json` as files) | https://claude.ai/artifact/BrSUsELtTfMcQZrj6rLxbp, source `docs/visual/attention.html`, data `scripts/attn_export.py` |
| Training curves page (artifact, private) | https://claude.ai/artifact/3gsN2gCjago7zG2AQs1eFA, source `docs/visual/training-curves.html`, data `scripts/curves_export.py` and `scripts/breakdown_export.py` (add/sub scores: `scripts/wide_ops.sh 40k|80k|120k`) → `docs/visual/curves/` (re-run to refresh H16). Local only: the JSON is gitignored |
| Skills | `.claude/skills/status-update`, `.claude/skills/decision-brief` (has a pre-launch checklist) |
| Working rules | `CLAUDE.md`. How the docs work: `docs/README.md` |

## Main results (full list in `hypotheses.md`)
Shallow 2×1 (4.4M), looped 2×8 (4.4M, 8 passes), deep 16×1 (19M, same compute as looped).
120k steps, 2 seeds, place-value tags, normal answer order (D-012).
- **Loops buy steps:** looped beats shallow by +0.11 (depth 4) to +0.25 (depth 8) and matches
  deep within ±0.03, also past training depth (13).
- **Weights buy number size:** deep beats looped by +0.05…+0.15 on big numbers, with or without `*` (H13).
- **Operator mix doesn't matter** at fixed depth and number size (H14).
- **Tags break longer numbers** (H18): the random tag offset turns nonsense into near-misses
  (`results/tags.md`). Exact 4-digit accuracy is still ~0.
- **H19 not shown:** probes read only levels 1–3; no one-level-per-pass staircase (`results/h19.md`).
- Fixed-K models fail at other loop counts. Reversed order: fails the rule, helps only deep.
- **H16 so far (one seed, unfinished):** held-out accuracy at 78k, looped vs deep: width 256
  0.328 vs 0.419, 384 0.358 vs 0.414, 512 0.435 vs 0.431. Looped closes the gap as width grows.
- Prior work: Saunshi et al. ICLR 2025, McLeish et al. NeurIPS 2024 (`prior-art/related-work.md`).

## Next actions after H16, in order
1. **Walter decides:** adopt `--tag_offset 8` as the tag default? (The D-014 rule says "approximate
   only, don't adopt", but offset matches plain tags in range and fixes out-of-range nonsense.)
   If yes: card for the long probe set with it (`scripts/arith_long_arm.sh "--place_emb --tag_offset 8" offset`).
2. Update the results page (`docs/visual/loop-depth-results.html` + artifact BBw1f6Y9HGTgb75UBM1Nfj)
   with H19, H18 and H16. Style: what / why / result / meaning, no semicolons, no H-codes in the body.
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
