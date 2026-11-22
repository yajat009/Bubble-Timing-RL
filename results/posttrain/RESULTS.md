# First post-training results: R2 against R3 (seed 0)

Jobs 59915750 (prepare and pretrain, 34 min), 59915751 (R2, 3.6 h) and 59915752 (R3, 2.3 h), one V100 each.
Code: `scripts/posttrain_*.sbatch`. Statement types: **measured**, **interpretation**, **recommendation**.

## Setup

- Backbone: Bubbleformer-S, frozen. Head: flow-matching U-Net on the 128 x 128 grid, 3.9M parameters.
- Head pretraining: 6 training files (Twall 90, 94, 98, 102, 106, 110), 379 windows each, 40 epochs.
  Validation loss fell from 1.97 to 0.59.
- Post-training: 60 iterations; each iteration = 8 rollouts of 600 frames from one start state of a training file.
  R2 rewards the heat-flux histogram and the vapor fraction. R3 rewards the departure rate, the waiting times and the
  lengths of quiet periods. Update rule: DiffusionNFT, with beta 0.1, lambda_sup 0.1, lambda_ref 0.01.
- Evaluation: one 1,895-frame rollout per head from frames 100-104 of Twall 97 and 107 (neither used in training),
  scored at full resolution by `compare_rollout_timing_v2.py`.
- **Check of the coarse detector (measured).** On the 6 training files, the 128 x 128 detector that the rewards use
  finds 90, 156, 188, 203, 202 and 223 departures; the full-resolution detector finds 88, 163, 191, 206, 208 and 228.

## Results

**Measured.** Departures per 100 frames, and distances from the simulation (noise floor = simulation second half
against first half). "S" is the backbone alone (sections 6 and 8 of `results/v2/RESULTS.md`); "head" is the pretrained
head before post-training.

| Twall 97 (simulation: 8.28 per 100 frames) | S | head | R2 | R3 | noise |
|---|---|---|---|---|---|
| Departures per 100 frames | 1.48 | 2.48 | 3.59 | 3.48 | - |
| Waiting-time W1 (frames) | 46.3 | 26.6 | 16.0 | 15.3 | 2.08 |
| Heat-flux histogram W1 | 3,100 | 859 | 5,820 | 16,800 | 409 |
| Heat-flux autocorrelation, RMS | 0.461 | 0.182 | 0.192 | 0.705 | 0.119 |
| Heat flux around departures, RMS | 2,340 | 1,770 | 6,680 | 15,300 | 840 |
| Vapor fraction (simulation 0.026) | 0.023 | 0.020 | 0.021 | 0.030 | - |

| Twall 107 (simulation: 11.7 per 100 frames) | S | head | R2 | R3 | noise |
|---|---|---|---|---|---|
| Departures per 100 frames | 7.18 | 5.96 | 5.38 | **8.97** | - |
| Waiting-time W1 (frames) | 5.41 | 8.11 | 10.0 | **2.65** | 1.73 |
| Heat-flux histogram W1 | 590 | 3,870 | 2,120 | 4,850 | 1,590 |
| Heat-flux autocorrelation, RMS | 0.057 | 0.110 | 0.066 | 0.320 | 0.066 |
| Heat flux around departures, RMS | 1,220 | 3,420 | 2,290 | 4,310 | 1,560 |
| Vapor fraction (simulation 0.035) | 0.037 | 0.026 | 0.022 | 0.040 | - |

**Training logs (measured, `train_log_R2s0.jsonl`, `train_log_R3s0.jsonl`).** On the training files, the rollout
departure rate stayed at about 0.6 of the simulation rate during training (R2: 0.59 in the first 10 iterations,
0.59 in the last 10; R3: 0.60, then 0.58). The mean group reward did not rise (R2: -6.98, then -6.70; R3: -16.0, then -21.7).

## What it means

**Interpretation.**

- The pretrained head alone already reduces the stall at Twall 97 (2.48 against 1.48 departures per 100 frames).
- At Twall 107, R3 gives the closest timing of all models (waiting-time W1 2.65, near the noise floor of 1.73).
  At the same time, R3 makes the snapshot statistics worse, much more at Twall 97. This is the trade-off that R4 is
  meant to prevent.
- **These results are not yet evidence that post-training works.** The training reward did not improve during the
  60 iterations, and each evaluation is one stochastic rollout. The differences between head, R2 and R3 at evaluation
  may come from the randomness of the head, not from learning.

## Recommendations

1. Measure the evaluation noise: 4-5 evaluation rollouts per head with different seeds.
2. Make the training signal stronger before more runs: a fixed set of start states, more iterations, and a check that
   the reward rises on the training files (for example, a higher learning rate or beta).
3. Run R4 (snapshot and timing together), and repeat R2, R3 and R4 with 3 seeds once the reward rises.
