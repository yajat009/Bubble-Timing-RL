# BubbleML 2.0 results: departures and the T3 observability probe

This file reports the first results on BubbleML 2.0, the data that Bubbleformer uses.
Data facts and Bubbleformer's settings are in `analysis/T0_DATA_ALIGNMENT.md`.
The test definitions and pass criteria are in `analysis/TEST_PLAN.md` (T3).

Statement types: **measured** (from a file in this folder), **interpretation**, **recommendation**.

## 1. Departures on all 17 files

**Measured.** `scripts/detect_departures.py`, with the debounce settings frozen after T0 (`DEBOUNCE_TIME = 1.0`, `DEBOUNCE_PX = 10`).
Counts and intervals use frames 100-2000 only, because Bubbleformer skips the first 100 frames.
One frame is 0.1 time units.

| Twall | Departures | Mean interval (frames) | CV |
|---|---|---|---|
| 86 | 33 | 26.5 | 0.77 |
| 88 | 46 | 39.4 | 0.71 |
| 90 | 88 | 21.1 | 0.92 |
| 92 | 103 | 18.2 | 0.93 |
| 94 | 163 | 11.5 | 0.77 |
| 96 | 150 | 12.7 | 0.83 |
| **97 (test)** | 157 | 12.0 | 0.80 |
| 98 | 192 | 9.9 | 0.81 |
| 100 | 188 | 10.1 | 0.84 |
| 102 | 206 | 9.2 | 0.83 |
| 104 | 218 | 8.7 | 0.83 |
| 106 | 208 | 9.1 | 0.70 |
| 108 | 231 | 8.2 | 0.80 |
| 110 | 229 | 8.3 | 0.76 |
| 112 | 233 | 8.2 | 0.86 |
| 114 | 243 | 7.8 | 0.74 |
| 116 | 248 | 7.6 | 0.71 |

- The mean interval falls as Twall rises: steeply up to Twall 98, then slowly. Twall 88 is an exception (39.4 frames, against 26.5 at Twall 86).
- The CV is between 0.70 and 0.93 at every Twall. A CV of 1 is what fully random (Poisson) timing gives. The pooled departures are close to random at every wall temperature.
- The debounce removes more raw events at high Twall: 6% at Twall 86, 38% at Twall 108. The detector has not yet been validated by eye (T1), and it does the most correction where bubbles are densest.

Files: `departures_Twall_<T>.npy` (frames), `departures_Twall_<T>_events.npy` (frame, x pixel), `departures_Twall_<T>.png`.

## 2. T3: do Bubbleformer's 5 input frames predict the time to the next departure?

### Method

`scripts/probe_v2.py`. All settings were fixed before the run.

- **Input**: 5 frames x 4 fields (`dfun`, `temperature`, `velx`, `vely`), the same fields Bubbleformer uses. Each field is averaged over 4 x 4 cells, from 512 x 512 to 128 x 128.
- **Target**: frames from the last input frame to the next departure, from any nucleation site.
- **Split**: Bubbleformer's own. Train on the 16 training files, test on Twall 97.
- **Windows**: frames 100-2000. A window is kept only if it has a previous and a next departure. Training uses every 5th window (5,811 windows), because neighbouring windows share 4 of 5 frames. Testing uses every window (1,881).
- **Training**: 4 convolution layers, absolute-error loss, Adam, 10 epochs, 5 seeds.
- **Irregular windows**: windows inside an interval more than 1 std from the mean training interval (582 of 1,881 test windows).
- **Median baseline**: added after the run by `scripts/median_baseline_v2.py`, for the reason given under the results.

### Results

**Measured.** Test MAE in frames on Twall 97. Models: mean ± sample std over 5 seeds.

| Model | All windows | Irregular windows |
|---|---|---|
| (a) mean only (training mean, 11.18) | 7.04 | 10.32 |
| (a') median only (training median, 7.00) | **6.52** | 12.27 |
| (b) time since last departure only | 6.56 ± 0.04 | 12.50 ± 0.10 |
| (c) 5 frames | 6.46 ± 0.47 | 9.45 ± 0.96 |
| (d) 5 frames + time since last departure | 6.55 ± 0.30 | 9.69 ± 1.74 |
| Best constant on the test file itself (not a fair baseline) | 6.49 | - |

For scale: the mean interval at Twall 97 is 12.0 frames.

Pass criteria from `analysis/TEST_PLAN.md`, with sd = 0.47 (seed std of model c):

| Criterion | Result | Values |
|---|---|---|
| 1. (c) beats (a) by more than 2 sd | **FAIL** | 7.04 - 6.46 = 0.58, less than 0.95 |
| 2. \|(d) - (c)\| within 1 sd | PASS | 0.09, less than 0.47 |
| 3. On irregular windows, (c) beats (b) by more than 2 sd | PASS | 12.50 - 9.45 = 3.05, more than 0.95 |

**T3 fails**, because the plan requires all three criteria.

### The median baseline changes the reading

A model trained with absolute-error loss that learns nothing useful predicts the median of the training targets.
The waiting times are skewed, so the median (7.00) has a lower absolute error than the mean (11.18).
The fair "no information" baseline is therefore the median, not the mean.
The test plan named only the mean baseline. This was a mistake in the plan.

Against the median:

- **All windows.** The frames model (6.46 ± 0.47) equals the median baseline (6.52). It also equals the best constant for the test file (6.49). The frames give no measurable information about the time to the next departure.
- **The true clock gives none either.** The time-only model (6.56) also equals the median baseline.
- **Criteria 2 and 3 do not show information.** Criterion 2 passes because neither the frames nor the clock helps. Criterion 3 passes because the irregular windows are mostly inside long intervals, where the targets are large. On those windows, a model that predicts larger values beats the time-only model. The frames model (9.45) beats the median (12.27) but not the mean baseline (10.32). This is a flaw in how criterion 3 was defined.

## 3. What this means

**Interpretation.**

1. At the pooled level (departures from any nucleation site; the metadata lists 15 sites at Twall 97 and up to 35 at higher Twall), the next departure is close to unpredictable. The CV is near 1, and neither the frames nor the true time since the last departure beats a constant guess.
2. This does not show that the frames hide a clock. It shows that, at this level, there is almost no clock to hide. A process with random (memoryless) timing has no phase.
3. For the RL plan, this is mixed news:
   - **Good**: if the pooled timing is memoryless, a Markovian 5-frame emulator does not need hidden memory to reproduce it. The worry behind T3 matters less at this level.
   - **Bad**: the pooled timing statistic mostly measures the departure **rate** and the **shape** of the waiting-time distribution. Those are what R3 can reward. "Departing at the right moment" is not learnable from this target.
4. The per-site timing may still have a clock. Each site waits `nucWaitTime = 0.4` (4 frames) and then grows a bubble over about 150 frames. This probe does not test that.

## 4. Limits of this result

- **One test file** (Twall 97), as in Bubbleformer's split. No other held-out Twall was tested.
- **Detector not validated** (T1). Every target depends on it.
- **128 x 128 input**, not Bubbleformer's 512 x 512. Small bubbles near the wall may be lost by the averaging.
- **Pooled target only.** A per-site target ("time until this site's next departure") might be predictable even though the pooled one is not.
- **Short training**: 10 epochs on 5,811 windows. A larger model might find information this one missed. The seed spread (0.47) is large compared with the differences being tested.

## 5. Recommendations

1. Validate the detector (T1) before drawing conclusions from any departure-based number.
2. Repeat T3 with a **per-site** target: for each nucleation site, frames until that site's next departure, using the input around that site.
3. Fix the T3 criteria: use the median as the no-information baseline, and define irregular windows so that both short and long intervals are included.
4. Revisit R3: with near-random pooled timing, a reward on the waiting-time distribution (rate and shape) is meaningful. A reward on the phase of individual departures is not, at the pooled level.

## Files

| File | Contents |
|---|---|
| `probe_v2.log` | Full printed output of `scripts/probe_v2.py`. |
| `probe_v2.json` | Per-seed MAE values and criteria results. |
| `probe_v2.png` | Bar plots of the results (mean baseline only). |
| `median_baseline_v2.log` | Output of `scripts/median_baseline_v2.py`. |
| `departures_Twall_<T>*` | Departures per file (section 1). |
