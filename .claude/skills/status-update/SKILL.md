---
name: status-update
description: Write a structured project status update (aim, done, doing, blocked, next decisions) for the loop-transformer research repo, from live evidence (running jobs, latest evals, git log, research log), and append it durably to docs/looped-math/status.md. Use when the user asks "how are we doing", "status", "where are we", "status update", or invokes /status-update.
---

# Status update

Produce a short, structured snapshot of the project **from evidence, not memory**, show it to
the user, and prepend it to `docs/looped-math/status.md` (newest first). The research log
(`docs/looped-math/research-log.md`) holds reasoning; `docs/looped-math/status.md` holds state.

## 1. Gather evidence (run these; don't guess)

Repo root: `/mnt/data/walter/loop-transformer`.

- **Running jobs:** `pgrep -af "looptf.toy.train|build_data|arith_grid"` (ignore the pgrep line itself).
- **Queues:** `cat results/toy/*.queue.log` (lines `done <name>` / `FAILED <name>`).
- **Per-run progress:** for each active or recent `results/toy/**/*.out`: last `^step` line
  (step, train acc, current curriculum depth), `grep mastered` (depths mastered and when), and
  the last eval table (after the last `--- step` line). Check for `Traceback`.
- **Finished results:** if `docs/looped-math/results/*.md` exists or runs have `per_example.npz`,
  regenerate the report: `.venv/bin/python scripts/arith_report.py --runs <dir> --out <md>`.
- **What changed:** `git log --oneline -15`, `git status --short`.
- **Context:** the last 1–2 entries of `docs/looped-math/research-log.md` and the previous entry in
  `docs/looped-math/status.md` (to report what moved since then).
- **Resources if relevant:** `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`,
  `df -h /mnt/data`.

## 2. Write the update in this exact shape

```
## YYYY-MM-DD HH:MM <TZ>

**Aim.** 1–3 sentences: the question the project is answering right now and the stage.

**Done** (since the last status entry; bullets, each with the concrete result/number)

**Doing** (what is running now, progress, ETA; a small table of current numbers if useful)

**Blocked / problems** (anything stopping progress or making results untrustworthy; say
what it invalidates)

**Next decisions** (numbered; options with your recommendation; mark which need the user)
```

Rules:
- Every claim comes from step 1's evidence. Numbers are copied, not rounded into adjectives.
  If something couldn't be checked, say so.
- Say what a result does *not* show yet (one seed, run unfinished, confound).
- Keep it to roughly one screen. Plain language; the user is newer to ML research, so name
  things concretely ("depth-3 test accuracy 0.81") rather than with jargon.
- Bad news goes in **Blocked**, not buried in **Doing**.

## 3. Save it

Prepend the entry to `docs/looped-math/status.md` directly below the file's header block (the first
`---` line), so the newest entry is first. Create the file with that header if it doesn't
exist. Commit only if the user asks, or if the session's convention is to commit log updates
(it currently is: commit with a short message like "Status update YYYY-MM-DD").

## 4. Show it

Print the same entry to the user, then one line with the single most important decision
pending.
