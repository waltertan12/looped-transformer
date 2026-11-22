---
name: decision-brief
description: Surface a key research decision to the user as a short decision card (question, options with what each outcome would mean, recommendation, who decides) and record it in docs/decisions.md. Use BEFORE launching any experiment or making any design change that alters what a result means or spends significant compute, and whenever the user asks "what are the key decisions", "what do I need to decide", or invokes /decision-brief.
---

# Decision brief

The user is newer to ML research and wants to see the calls that matter instead of having an
agent make them silently ("vibe research"). Your job: separate **decisions** from
**execution**, and put every real decision in front of them in a form they can judge.

## What counts as a decision (needs a card)
- Choosing what an experiment tests or how success is measured (metric, threshold, baseline).
- Design choices that could make a result mean something else (data format, curriculum,
  model variants, fixed vs random loops, what to hold out).
- Spending more than ~30 min of GPU, or stopping/restarting runs.
- Anything that gives the model a way to "opt out" or a shortcut (lesson D-001).

Not decisions: fixing a bug, writing a test, launching a run whose card is already decided,
formatting, refactors that don't change behaviour.

## The card (append to `docs/decisions.md`, newest first, next ID)

```
## D-NNN · <one-line question> (open)

**Question.** What's being decided, and why now (the evidence that triggered it).

**Options.** 2–3 real options. For each: what it costs, and what each possible outcome
would tell us.

**What would change our minds.** The result that would make us pick differently or abandon
the line of work. (This is the pre-registration.)

**Recommendation.** Your pick and the one-line reason.

**Needs Walter?** Yes / No (No only for calls that don't change what results mean).
**Status.** Open | Decided: <option>, by <who>, <date>.
```

## How to present it
- In chat: the question, the options in plain language (use `docs/glossary.md` terms; add new
  terms to both glossary files), your recommendation, then ask. One decision per question
  when possible; most important first.
- Don't bury a decision inside a progress update; give it its own short block.
- If the user says "just go", record the decision as decided by Claude with the user's
  go-ahead, and proceed.
- When a decided card later turns out wrong, add a line to it saying so and why (as in D-001).

## Also
- Keep `docs/glossary.md` (plain, for the user) and `docs/glossary-agent.md` (terse, for
  agents) in sync whenever a new term appears in a card.
