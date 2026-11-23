# Hypothesis log

One line per claim, with its status and evidence. Walter ranks the open ones; rank goes in the
first column. Update the status when a result comes in (link the research-log entry).
Evidence is seed 0 / seed 1 at 120k steps (D-012 normal arm) unless noted.

## Tested

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Loops add step-count capacity: looped beats shallow (same weights), more so on deeper problems | **Supported** | 1-digit answers: +0.11 at depth 4 → +0.25 at depth 8, both seeds |
| H2 | Looped matches deep at equal compute on step count, with 4× fewer weights | **Supported** | within ±0.03 at every depth, both seeds |
| H3 | Loops don't buy number size; unique weights do | **Supported** | deep − looped +0.05…+0.15 on big answers / big inputs; same size on pure add/sub (H13), so it's number size, not lookup |
| H4 | Looped generalizes past training depth (trained ≤ 8) | **Supported** | depth 13: looped 0.62/0.60 vs shallow 0.42/0.43, ≈ deep |
| H5 | A model trained at K loops works at other K | **Rejected** | ≤ 0.11 at other K (40k grid) |
| H6 | Reversed answer order helps | **Rejected by the rule; depends on capacity** | fails the rule on all 3 models. Big answers: shallow −0.07, looped −0.03, deep +0.13/+0.10. Helps shallow on one big multiplication (+0.12…+0.15) |
| H7 | Place-value tags help | **Weakly supported** | looped 40k: big answers +0.055; shallow: no effect. Only one seed of tags-vs-no-tags |
| H13 | Loops match deep on pure add/sub (so H3 is about multiplication lookup) | **Rejected** | depths 1–3, 2–3-digit inputs, unseen prompts only: deep − looped +0.05…+0.15 (add and sub, both seeds), same as mixed data. Pre-set rule said "unclear" only because of 0%-floor cells (rule-design mistake) |
| H14 | Operator mix predicts errors beyond depth and number size | **Not supported** | matched cells: low − high `*` third −0.01…+0.015 |
| H8 | The 40k models were undertrained | **Supported** | 120k lifts everything +0.08…+0.26; train acc still 0.42–0.54 at 120k |

## Open (untested), for ranking

| Rank | # | Hypothesis | Plan step / idea |
|---|---|---|---|
| | H9 | A looped model can learn to use fewer loops on easy problems and more on hard ones, with no accuracy loss | plan step 5 |
| | H10 | A stop head on the hidden state beats a hand rule or an external checker ("the model knows it's done") | plan step 6 |
| | H11 | Which blocks loop matters (looping the middle beats looping everything) | plan step 7 |
| | H12 | Looping helps following references (chain task), not just nested evaluation | plan step 8 |


| | H15 | Letting the question tokens see each other (prefix attention) helps, because causal attention hides later operators | I-M3 |
| | H16 | Wider models fix number size; loops still add steps on top | I-L1 |
| | H17 | Step skill transfers to algebra (solving 3x+5=20) | I-L2 |
| | H18 | Tags + random offset let models handle longer numbers than trained (today: 4-digit inputs = 0% everywhere) | I-M2 |
