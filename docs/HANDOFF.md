# Handoff: read this first in a new session

_Written 2026-11-22 21:25 EST, updated 21:45. Update it whenever the state below changes._

## What this project is
Walter (software engineer, newer to ML research) is testing whether **looped transformers**
(the same layers run several times) add reasoning depth. Goal: a **mini paper** with an
evidence-backed opinion about loops. One RTX 5090 (32 GB, 450 W cap) at home.

Current track: **tiny transformers trained from scratch on synthetic PEMDAS arithmetic**
(`src/looptf/toy/`). The earlier Qwen-retrofit track (exp1–3) is paused: it didn't work.

## Where things are
| what | file |
|---|---|
| Forward plan, with decision rules written **before** results | `docs/plan.md` |
| Hypotheses (status + Walter's ranking) | `docs/hypotheses.md` |
| Idea backlog (short / medium / long) | `docs/ideas.md` |
| Closest papers, compared with ours | `docs/related-work.md` |
| Every decision (cards D-001…D-012), who made it, outcome | `docs/decisions.md` |
| Reasoning and results, chronological | `docs/research-log.md` |
| State snapshots (newest first) | `docs/status.md` |
| Glossary: plain (for Walter) / terse (for agents) | `docs/glossary.md`, `docs/glossary-agent.md` |
| Visual results page (published) | https://claude.ai/artifact/BBw1f6Y9HGTgb75UBM1Nfj, source `docs/visual/loop-depth-results.html` |
| Skills | `.claude/skills/status-update`, `.claude/skills/decision-brief` (has a pre-launch checklist) |
| Working rules | `CLAUDE.md` (bottom section "Working with Walter") |

## Main result so far (seed 0, 40k steps, `data/nested_bal`)
Shallow 2×1 (4.4M), looped 2×8 (same 4.4M, 8 passes), deep 16×1 (19M, same compute as looped):
- **Loops buy depth:** on many-step problems with small answers, looped beats shallow by
  +0.16…+0.29 and matches deep with 4× fewer weights. Generalises past training depth.
- **Weights buy arithmetic:** on 3-digit answers deep beats looped by +0.08…+0.30; loops add ~+0.03.
- **Reversed answers** hurt all three (D-007 → normal order).
- A fixed-8-loop model is useless at other loop counts (≤ 0.11).
- Close prior work: Saunshi et al., ICLR 2025 (arXiv 2502.17416): looped ≈ deeper models on
  reasoning; loops help reasoning more than memorization. This project is a replication +
  a clean depth-vs-number-size decomposition + negative results.

## In flight right now
- **D-012 long probe set**, launched 2026-11-22 21:40: shallow / looped / deep × seeds 0,1,
  120k steps, `--place_emb`, normal order. Results `results/toy/long/`; queue log
  `results/toy/long.queue.log` (6 "done" lines = finished; order shallow → looped → deep,
  2 at a time). Shallow ~40 min, looped and deep ~2.5 h each → done ~03:30–04:00.
- Looped pair done (00:30): looped beats shallow at 120k by +0.11 (depth 4) … +0.25 (depth 8),
  both seeds; also +0.13/+0.15 on big answers. Deep pair running → ~02:50.
- **Normal arm done (02:55):** looped = deep on step count (±0.03, both seeds); deep beats
  looped by +0.05…+0.15 on big answers / multi-digit inputs. Research log 2026-11-23 02:55.
- **D-012 reversed arm** (running since 02:55) (Walter, 22:20): same 6 runs with `--place_emb --reverse_answer`,
  queued behind the normal arm (`scripts/arith_long_arm.sh`, WAIT_PID on the normal queue),
  names `long/<model>-rev-s<seed>`, queue log `results/toy/long-rev.queue.log`; starts ~02:50,
  done ~08:15. Rule on the D-012 card.
- Shallow pair done (21:55): 120k ≫ 40k (+0.08…+0.26), seeds agree within 0.03. Shallow at
  120k ≈ looped at 40k, so the looped 120k pair decides whether the loop gap is real (research log).
- D-011 screen is finished: **tags on (looped big answers +0.055), normal order** (reversed
  +0.033 on multi-digit inputs but −0.110 on 1-digit inputs). See research log and D-011 card.
  Re-run the comparison: `.venv/bin/python scripts/d011_compare.py`.

## Next actions, in order
1. When both D-012 arms finish (12 runs): compare normal vs reversed by the D-012 card rule; report with `scripts/arith_report.py --runs results/toy/long --out
   docs/results-arith-long.md`, plus seed-to-seed spread (seed 0 vs 1) on every headline
   number. Log in research log and on the D-012 card. Check whether 120k beats 40k (vs
   `results/toy/place/` and `results/toy/bal/`).
2. Update the visual page (republish the same file path or pass the URL), with seed spread.
   This is the write-up checkpoint.
3. Offered, not yet accepted: scaffold `docs/paper/outline.md` (sections, numbers, figures,
   checked related work). Walter writes the prose.
4. Later (plan steps 5–8): variable loop control (deep supervision → confidence stop →
   learned stop head); deferred step 6 "who decides to stop"; loop placement; chain task.

## Practical notes
- Run Python as `/mnt/data/walter/loop-transformer/.venv/bin/python`. Tests: `pytest tests -q -k toy`.
- Run at most **2 trainings at a time**, each with `--mem_fraction 0.4`. GPU at ~94% use is normal.
- Don't use `pkill -f <pattern>`: it kills the calling shell. Get PIDs from
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader` or `ps -eo pid,args | grep "[p]attern"`.
- `rm -rf` on globbed paths is blocked by a safety check; move things to `results/toy/stopped/` instead.
- Headless Firefox screenshots hang on this machine; there's no node/playwright. Publish without a preview.
- `results/`, `checkpoints/`, `data/` are gitignored. Data rebuilds deterministically:
  `scripts/build_data.py --which nested_bal --out data` (~6 min), verify with `scripts/verify_data.py`.
- Commit docs/code changes as you go; end commit messages with the attribution line in use.
