# Hypotheses

What the project has tested, what's running, and what's next. Each row is a short summary;
follow **Details** for the full write-up. Numbers are the average of two runs at 120k training
steps unless noted. To re-rank what's next, reorder its rows.

## Running now

Nothing. The last batch (the long runs and the add/subtract tests) finished on 23 Nov.

## Next, in priority order

| # | Hypothesis | How we'll test it | Cost | Details |
|---|---|---|---|---|
| H9 | A looped model can choose its own loop count: few on easy problems, more on hard ones | Train it to answer after every pass, then let it stop early | A few runs | [plan](plan.md#plan-step-5) |
| H10 | A stop signal learned from the model's internal state beats a simple rule or a separate checker | Compare the three on the H9 model | After H9 | [plan](plan.md#plan-step-6) |
| H21 | The number of attention heads matters: 1 vs 2 vs 4 (today) vs 8, same weights | Start with 2 vs 4 on shallow, looped and deep | 6 runs (one new head count × 3 models × 2 seeds) | [idea](ideas.md#idea-heads) |
| H15 | Letting question tokens see each other helps. Today the `+` in `3+4*5` can't see the `*` | One attention-mask change, three models | 6 runs | [idea](ideas.md#idea-attention) |
| H11 | Looping only the middle layers beats looping all of them | Five layer layouts at the same compute | 5 runs | [plan](plan.md#plan-step-7) |
| H16 | Wider models fix number size, and loops still add steps | Width 512–1024 | Many hours | [idea](ideas.md#idea-wide) |
| H12 | Loops help follow chained definitions like `a=3*4; b=a+2` | Same comparison on the chain data (already built) | 6 runs | [plan](plan.md#plan-step-8) |
| H20 | Teaching each pass to output one intermediate step (one bracket level done) helps loops more than deep | Extra loss per pass on a readout of the half-solved problem | 4–6 runs, needs a target generator | [idea](ideas.md#idea-intermediate) |
| H17 | Step skill carries over to algebra like `3*x+5=20` | New data generator first | Later | [idea](ideas.md#idea-algebra) |

## Done

| # | Hypothesis | Result | Details |
|---|---|---|---|
| H1 | Loops buy steps: looped beats shallow (same weights), more on deeper problems | **Yes.** +0.11 at depth 4, +0.25 at depth 8 | [log](research-log.md#log-long-looped) |
| H2 | Looped matches deep on steps with a quarter of the weights | **Yes.** Within 0.03 at every depth | [log](research-log.md#log-long-normal) |
| H3 | Extra weights, not loops, buy number size | **Yes.** Deep beats looped by +0.05 to +0.15 on big numbers | [log](research-log.md#log-long-normal) |
| H4 | The step skill reaches past the training depth | **Yes.** At depth 13 (trained up to 8), looped 0.61, shallow 0.42, deep 0.62 | [log](research-log.md#log-long-looped) |
| H5 | A model trained at one loop count works at others | **No.** At most 0.11 at other counts | [log](research-log.md#log-grid) |
| H6 | Answering smallest digit first helps | **No, by our rule.** Hurts small models, helps deep (+0.11 on big answers) | [log](research-log.md#log-reversed) |
| H7 | Place-value tags help | **A little.** +0.055 on big answers for looped (40k steps, one run). No effect on shallow | [log](research-log.md#log-tags) |
| H8 | The first models (40k steps) were undertrained | **Yes.** 120k steps raised every result by +0.08 to +0.26 | [log](research-log.md#log-long-shallow) |
| H13 | Loops match deep on pure addition and subtraction, so the number gap is about times tables | **No.** Deep still wins by about +0.10 on unseen problems. The gap is number size | [log](research-log.md#log-h13-h14) |
| H18 | Longer numbers work if the place-value tags are shifted randomly in training, or kept separate for question and answer | **Partly.** The random shift turns nonsense into near-misses (within 5%: 0.97, was 0.03). Separate tags don't help. Exact 4-digit still ~0, but so is most in-range exactness at 40k steps | [results](results-tags.md), [log](research-log.md#log-h18) |
| H19 | Each loop pass works through one level of brackets, innermost first | **Not shown.** Probes only read levels 1–3: level 1 before the loop, levels 2–3 together and gradually. Same in deep | [results](results-h19.md), [log](research-log.md#log-h19) |
| H14 | Problems with more multiplication are harder at the same size | **No.** Within 0.015 once depth and number size are matched | [log](research-log.md#log-h13-h14) |
