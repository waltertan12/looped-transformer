# Related work

What other people have published that bears on this project, and how our results compare.
Facts were gathered by Haiku agents and checked against the paper pages where marked.

**How sure each entry is:** ✓ read on the paper page · ~ seen in search results only ·
? not checked. Re-check anything marked ~ or ? before quoting it in the write-up.

## The short version

- **Two papers are very close to ours.** Saunshi et al. (Google) show loops help reasoning more
  than memorization. McLeish et al. get looped models to add 100-digit numbers. Our results fit
  both.
- **Nobody seems to have tested our H20 idea directly.** Teaching each loop pass to output one
  intermediate step would be a new experiment. Nearby work varies *which* pass gets the loss
  but never asks every pass to be a readable step.
- **Steering attention with an extra loss has been done**, mostly in translation, and it didn't hurt
  the main task. We found no one doing it for arithmetic. For arithmetic, people have had more
  luck fixing the attention structure than training it.
- **Letting a model pick its own loop count (H9) is known to be fiddly.** Learned stopping tends to
  collapse to always-short or always-long. Start simple.

## The two closest papers

<a id="rw-saunshi"></a>

### Saunshi et al., "Reasoning with Latent Thoughts" (Google, ICLR 2025) ✓

[arXiv 2502.17416](https://arxiv.org/abs/2502.17416)

- **Setup.** Loops a small block of layers a fixed number of times. No re-injection of the input,
  no early stopping, same loop count in training and testing. Their "middle looping" (own first
  and last layers) is our prelude / core / coda.
- **Toy tasks.** Adding up to 32 three-digit numbers, multi-hop lookups, and a small math
  word-problem set. A 2-layer block looped 6 times scores 99.5% on 32-number addition. The plain
  2-layer model scores 38.8%, and a 12-layer model scores 100%.
- **Language models (1B).** Looping closes 34–48% of the gap to the deep model on trivia
  (memorization), 56–61% on math word problems and over 100% on reasoning drills. They call this
  %Gap = (looped − shallow) / (deep − shallow). We use the same measure.
- **A trick.** Keep a normal deep model but add a loss that pulls blocks toward each other. It got
  the best of both: deep-model fluency with looped-model reasoning.
- **Not done:** other loop counts at test time, early stopping, number size.

<a id="rw-mcleish"></a>

### McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings" (NeurIPS 2024) ✓

[arXiv 2405.17399](https://arxiv.org/abs/2405.17399)

- **Abacus tags.** Each digit gets a tag for its place within its number, with digits written
  smallest first. Tags are shifted randomly during training so longer numbers still get familiar
  tags. Our H18 random offset is this idea.
- **Results.** Trained on numbers up to 20 digits, it adds 100-digit numbers at up to 99%. Looped
  models match or beat deep ones with up to 6× fewer weights (8 layers × 2 loops: 99.1%, 16 × 1: 97.9%).
- **Input injection.** The embedded input is added back before each loop pass. Ours concatenates it.
- **Progressive loss.** The loss mixes a full-depth run with a run at a random smaller depth. It is
  not a loss at every pass, and they don't measure its effect on its own.
- **Multiplication** worked only within the trained sizes and needed extra tweaks: no injection
  inside the loop, and loop gradients divided by the loop count.

## How our results line up

Our main runs: 120k steps, two seeds.

- **Same split as Saunshi.** Looped closes about all of the gap on long chains of steps (%Gap 0.91
  and 1.08 at depth 8) but only part of it on big numbers (0.23–0.63). Steps behave like their
  "reasoning" and number size like their "memorization".
- **Why McLeish's looped model matches deep on numbers and ours doesn't.** Their task is one big
  addition, where carrying is a step-by-step procedure that suits loops. Ours is mostly
  multiplication and nested results, which need lookup capacity, which means weights. The two
  results agree once you see that.
- **Reversed digit order.** McLeish uses it for single operations. We find it helps a single big
  multiplication but hurts multi-step problems for small models. Neither paper shows that.
- **What we add.** Exact control of steps versus number size, both seeds reported (Saunshi
  reports the best of 3), and negative results such as fixed-loop models failing at other loop counts.

## Prior art for what's next

<a id="rw-steps"></a>

### Teaching each pass a step (H20) and extra losses inside the network

| Paper | What they did | What happened | What it means for us |
|---|---|---|---|
| Deeply-Supervised Nets, Lee et al. 2014 ✓ [1409.5185](https://arxiv.org/abs/1409.5185) | Extra classifier and loss on each hidden layer of an image model | Better results on standard image benchmarks | The classic "loss at every stage". Image models, not loops |
| Looped Transformers for Length Generalization, Fan et al. 2024 ✓ [2409.15647](https://arxiv.org/abs/2409.15647) | Each training example runs a number of passes that fits its size, loss only at that pass | Generalizes to longer inputs on counting, copying, sorting | Closest in spirit. Varies the depth of the loss but never supervises every pass |
| Recurrent depth LM, Geiping et al. 2025 ~ [2502.05171](https://arxiv.org/abs/2502.05171) | 3.5B model trained with a random loop count, loop count picked at test time | Biggest gains on math and code | Loops scale without per-pass losses |
| Ouro looped LMs, Zhu et al. 2025 ✓ (abstract) [2510.25741](https://arxiv.org/abs/2510.25741) | Looped LMs that learn how deep to go | 1.4B–2.6B match up to 12B standard models | Large-scale proof that looping works. No per-pass step loss |
| Implicit chain of thought, Deng et al. 2024 ✓ [2405.14838](https://arxiv.org/abs/2405.14838) | Train with written steps, then remove them a few at a time | GPT-2 Small does 9×9-digit multiplication at 99%. Normal training fails past 4×4 | Step-by-step teaching works on arithmetic. It removes steps rather than adding a loss on them |
| Coconut, Hao et al. 2024 ✓ (abstract) [2412.06769](https://arxiv.org/abs/2412.06769) | Feed the last hidden state back in as the next input instead of writing words | Beats written steps on planning-heavy logic tasks | Like a loop that reasons in hidden states. Taught with a written-steps curriculum |
| Scratchpads, Nye et al. 2021 ✓ (abstract) [2112.00114](https://arxiv.org/abs/2112.00114) | Model writes its working out before the answer | Helps long addition and running programs | The "write the steps out" baseline H20 tries to move inside the loop |
| CLRS benchmark, Veličković et al. 2022 ? [2205.15659](https://arxiv.org/abs/2205.15659) | Algorithms with step-by-step traces ("hints") as optional extra targets | Not checked. From memory, follow-up work found hints help on some algorithms and not others | The most direct test of "supervise the intermediate state". Read the results before H20 |

<a id="rw-attention"></a>

### Steering attention

| Paper | What they did | What happened | What it means for us |
|---|---|---|---|
| Supervised attention for translation, Liu et al. 2016 ✓ (abstract) [1609.04186](https://arxiv.org/abs/1609.04186) | Extra loss pulls attention toward word alignments from an outside tool | Better alignments and better translations | Direct precedent for our idea (e) |
| Jointly learning to align and translate, Garg et al. 2019 ✓ (abstract) [1909.02074](https://arxiv.org/abs/1909.02074) | Train one attention head of a Transformer to follow alignments | Alignments beat the standard tool, translation not hurt | Supervising one head can work without hurting the main task |
| From interpolation to extrapolation, Duan et al. 2023 ~ [2310.11984](https://arxiv.org/abs/2310.11984) | Arithmetic with a built-in attention bias instead of a loss | Near-perfect length generalization | For arithmetic, fixing the structure may beat training it. Try a mask before a loss |
| Linguistically-informed self-attention, Strubell et al. 2018 ? | One head trained to follow sentence structure | Not checked | A name to look up |

One search found no paper that adds an attention loss for arithmetic. That means "not found",
not "doesn't exist".

<a id="rw-halting"></a>

### Letting the model choose its loop count (H9)

| Paper | What they did | What happened | What it means for us |
|---|---|---|---|
| Adaptive Computation Time, Graves 2016 ~ [1603.08983](https://arxiv.org/abs/1603.08983) | A stop signal decides how many steps to run, with a cost per step | Extra steps go to hard spots | The original learned-stopping method |
| Universal Transformers, Dehghani et al. 2018 ~ [1807.03819](https://arxiv.org/abs/1807.03819) | A looped Transformer where each token picks its own step count | Stopping helped accuracy as well as saving compute | Per-token stopping is an option for us |
| PonderNet, Banino et al. 2021 ~ [2107.05407](https://arxiv.org/abs/2107.05407) | Stopping as a probability with a prior on how long to think | Beat earlier methods on parity and extrapolated | A steadier version of learned stopping |
| Two-scale latent dynamics, Pappone et al. 2025 ✓ (abstract) [2509.23314](https://arxiv.org/abs/2509.23314) | Stop when the hidden state stops changing much between passes | Better than Geiping's stop rule | A no-training stop rule we can try on any looped model |
| LoopFormer 2026 ~ [2602.11451](https://arxiv.org/abs/2602.11451) and "When does adaptive halting win?" 2026 ~ [2611.02333](https://arxiv.org/abs/2611.02333) | Early stopping in looped models | Reported collapse: models learn to stop too early or never | Read before building H9 |

## What this means for our plans

- **H20 is new.** No one we found asks every pass to be a readable step. Measure it against two
  baselines: loss at the last pass only, and McLeish's progressive loss. The progressive loss is
  close to free and may give most of the benefit.
- **Read the CLRS hints results first.** They are the nearest real test of "supervise the
  intermediate state", and the answer there may be "it depends".
- **For attention, try a mask before a loss.** A fixed structure helped arithmetic. A loss on one
  head is the softer follow-up.
- **For H9, start with a no-training stop rule** (stop when the state stops changing) and a sweep of
  loop counts. Add a learned stop signal only after that.

## Ideas these papers suggest (not planned yet)

1. Report %Gap as a headline number, so we compare directly with Saunshi.
2. McLeish's progressive loss as a cheap first step toward H9 and H20.
3. Saunshi's pull-blocks-together loss on our deep model. Does it keep deep's arithmetic and gain looped's step skill?
4. Length generalization: train on inputs up to 2 digits, test 3–4, with the random tag offset.
5. Fit accuracy against log(depth) across our grid and compare the slope for loops and layers.
6. Added versus concatenated input injection, plus McLeish's multiplication tweaks.
7. One pure big-addition task, where looping should match deep, as a check on "weights buy arithmetic".
