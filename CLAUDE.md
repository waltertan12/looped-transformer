# loop-transformer

Research repo on looped (weight-shared) transformers. Several projects share it. The docs are the
project's memory: read `docs/README.md` for how they work.

## Start of a session
1. Run `scripts/lock.py show`. It lists every queue running on this machine, from any project.
2. Read `docs/README.md` (the harness), then the `HANDOFF.md` of the project you work on.
   Active project: `docs/looped-math/` (code `src/looptf/toy/`). Idea stage: `docs/protein/`,
   `docs/virtual-cell/`. Stopped: the Qwen retrofit (`docs/looped-math/history/qwen-retrofit/`).
3. Repo-wide structure and tech debt: `docs/housekeeping.md`. Add debt you notice there.

## Machine
- Repo at `/mnt/data/walter/loop-transformer`. Python venv at `.venv`. Run `.venv/bin/python`.
- GPU: one RTX 5090, 32 GB, power-limited to 450 W. Everything must fit on that card.
- Run at most 2 trainings at a time. Register every queue with `scripts/lock.py add`.
- Do not run tests or evals on the GPU while trainings run.
- Leave 25–50% of CPU cores free (Walter, 2026-11-24): use at most 12 of 24 threads (`torch.set_num_threads`, `OMP_NUM_THREADS`, `threadpoolctl`) and `nice -n 10` for CPU-heavy work.
- `results/`, `checkpoints/`, `data/` are gitignored.

## Working with Walter
- Walter is newer to ML research and wants to see the key calls, not have them made silently.
  Before any experiment or design change that alters what a result means (or spends > 30 min of
  GPU), write a decision card with the `decision-brief` skill and ask. Routine execution proceeds
  without asking.
- Write to the docs as you go (`docs/README.md` lists what goes where). Status snapshots:
  `status-update` skill. Check results against the pre-registered rules in the project's `plan.md`.
- Human-facing docs (ideas, hypotheses, plan, decisions, specs, log, results) use plain language
  and glossary terms. Agent-facing docs (`plans/impl/`, HANDOFF how-tos, skills) use ASD-STE100
  style: one imperative instruction per sentence, exact paths and commands.
- Superpowers specs and plans go in `docs/<project>/plans/specs/` and `docs/<project>/plans/impl/`,
  not in `docs/superpowers/`.
- Keep `docs/glossary.md` (plain) and `docs/glossary-agent.md` (terse) in sync when a new term
  comes up.
