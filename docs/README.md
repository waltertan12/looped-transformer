# How the docs work

The docs are the project's memory. Two jobs:

1. **Restart.** Any new session (person or agent) can pick up the work from the docs alone.
2. **Record.** Every idea, hypothesis, decision and result is written down, so we can see what
   happened, what changed and why.

If it is not in the docs, it did not happen. Write as you go, not at the end.

## Layout

```
docs/
  README.md            this file: how the harness works
  index.md             landing page of the public site (GitHub Pages serves docs/)
  housekeeping.md      repo structure and tech-debt to-do list, across projects
  glossary.md          shared terms, plain language (for Walter)
  glossary-agent.md    shared terms, terse (for agents)
  visual/              HTML pages published as claude.ai artifacts (looped-math; see below)
  looped-math/         ACTIVE project: do looped transformers add reasoning depth? (tiny arithmetic models)
  protein/             IDEA stage: the same loop question on proteins
  virtual-cell/        IDEA stage: toy gene networks shaped like the arithmetic task
```

Each project folder starts with a `README.md`: intent, current question, status, file map.

**Handoffs.** Each project has its own `HANDOFF.md` with its research state: what runs, how to
resume, next actions. There is no top-level handoff. Repo-wide structure and tech debt go in
`housekeeping.md` instead.

## The files in an active project

| file | holds | audience | when to write |
|---|---|---|---|
| `README.md` | intent, current question, status, file map | human | when the question or status changes |
| `HANDOFF.md` | what is running, how to resume it, next actions | agent and human | whenever the running state changes |
| `ideas.md` | idea backlog (short / medium / long) | human | when an idea comes up, even half-formed |
| `hypotheses.md` | every hypothesis (H-number), its test, its verdict | human | when an idea becomes testable; when a verdict lands |
| `plan.md` | next steps, each with decision rules written **before** results | human | before a run; revise when a step finishes, say why |
| `decisions.md` | decision cards (D-number): question, options, who decided, outcome | human | before any run that changes what a result means or uses > 30 min GPU |
| `plans/specs/` | design of one experiment (superpowers brainstorming) | human | before building it |
| `plans/impl/` | step-by-step build plan (superpowers writing-plans) | agent | after the spec is agreed |
| `research-log.md` | reasoning and results, in time order | human | after every result, surprise or change of mind |
| `results/` | one write-up per finished experiment | human | when an experiment is scored |
| `status.md` | dated state snapshots, newest first (`status-update` skill) | human | when asked "where are we", and at milestones |
| `prior-art/` | closest papers, compared with our results | human | when a paper bears on a hypothesis |
| `history/` | how the direction changed, and stopped approaches | human | when the direction shifts |

## How an idea moves to a verdict

```
ideas.md  ->  hypotheses.md (H-number)  ->  plans/specs + plans/impl  ->  decisions.md (D-number, Walter decides)
   ->  plan.md (rules written before the run)  ->  run  ->  research-log.md  ->  results/<name>.md
   ->  verdict in hypotheses.md  ->  status.md + HANDOFF.md  ->  history/ if the direction changed
```

Small or obvious steps can skip `plans/`. A decision card is never skipped for a run that changes
what a result means or uses more than 30 minutes of GPU.

## Two writing styles

**Human-readable** (README, ideas, hypotheses, plan, decisions, specs, research log, results,
status, history). Plain language, terms from `glossary.md`. Say what was done, why, the result
with its number, and what it means. Say what a result does *not* show yet (one seed, unfinished run).

**Agent-readable** (`plans/impl/`, the "how to" parts of `HANDOFF.md`, `CLAUDE.md`, skills,
`glossary-agent.md`). Write in the style of ASD-STE100 Simplified Technical English:
- One instruction per sentence. Use the imperative: "Run X." "Do not edit Y while Z runs."
- Give exact paths, commands, flags and expected output.
- Use one word for one thing. Do not use synonyms.
- State conditions before actions: "If the run crashes, run the same mode again."

## Rules for the record

- `research-log.md`, `decisions.md` and `status.md` are append-only. Do not rewrite old entries.
  To correct one, add a dated note below it.
- H-numbers and D-numbers are never reused.
- Use absolute dates with a time zone (`2026-11-24 15:30 EST`), never "yesterday".
- When a new term comes up, add it to both glossaries.
- Commit docs as you go. Push after committing.

## Starting a new project

1. Create `docs/<project>/README.md` (intent, status, file map) and `notes.md`. That is enough at
   the idea stage.
2. When the first experiment is planned, add the files in the table above.
3. Add the project to `index.md` and to the root `README.md`.

## One machine, shared by all projects

One GPU. Before launching anything, run `scripts/lock.py show`. Register a new queue with
`scripts/lock.py add`. The GPU rules are in the root `CLAUDE.md`.
