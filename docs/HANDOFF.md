# Handoff: read this first in a new session

_Written 2026-11-22 21:25 EST. Update it whenever the state below changes._

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
- **D-011 screen** (place-value tags, `scripts/arith_place.sh`, results `results/toy/place/`):
  shallow pair done; **looped pair (c2k8-normal-pv, c2k8-rev-pv) finishing ~21:35**
  (queue log `results/toy/place-s0.queue.log`, 4 "done" lines = finished).
  Shallow result: no effect on the rule's original metric (1-digit inputs carry no place
  information: Claude's metric mistake); exploratory: tags + reversed lift depth-1 3-digit-input
  problems 0.14 → 0.43 (3-digit answers 0.38 → 0.89); tags + normal best at depth 2.

## Next actions, in order
1. When the looped D-011 pair finishes: compare against `results/toy/bal/c2k8-normal-s0` (no
   tags) on (a) big answers on 1-digit-input problems and (b) multi-digit-input problems,
   depths 1–3 (co-primary since the shallow results). Log results (research-log, D-011 card).
2. **Apply D-012's recipe rule** (in `docs/decisions.md`): tags on; reversed only if looped +
   tags + reversed beats looped + tags + normal by ≥ 0.05 on multi-digit-input problems and is
   ≤ 0.03 worse on 1-digit-input problems; else normal.
3. **Launch D-012** (approved by Walter): `nohup scripts/arith_long.sh "--place_emb" >
   results/toy/long.queue.log 2>&1 &` (add `--reverse_answer` to the flags if the rule says so).
   6 runs, 120k steps, seeds 0 and 1, ~6 h. Two at a time (Walter's preference).
4. After D-012: update the visual page (republish the same file path or pass the URL), with
   seed-to-seed spread on every number. This is the write-up checkpoint.
5. Offered, not yet accepted: scaffold `docs/paper/outline.md` (sections, numbers, figures,
   checked related work). Walter writes the prose.
6. Later (plan steps 5–8): variable loop control (deep supervision → confidence stop →
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
