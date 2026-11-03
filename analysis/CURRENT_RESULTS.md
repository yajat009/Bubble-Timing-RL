# Current results: what the repo shows, checked

This document checks the results in this repository against the claims in the project brief.
It separates three kinds of statement:

- **Measured**: a number computed from a file in this repository.
- **Assumption**: a statement taken from the brief or a web page that this repository does not verify.
- **Recommendation**: what we suggest doing next.

Terms used throughout:

- **Departure**: the frame where a bubble leaves the heater wall. Section 3.3 gives the exact rule the code uses.
- **Interval**: the number of frames between two consecutive departures in one file.
- **CV (coefficient of variation)**: the standard deviation of the intervals divided by their mean. A low CV means a regular rhythm.
- **MAE (mean absolute error)**: the average of |prediction - truth|, in frames.
- **Probe**: a small neural network trained only to test what information its input contains.
- **Twall**: the heater wall temperature of a simulation. In this repo each file has one Twall.

## 0. The most important finding: the data is not the project's data

**Measured.** Every result in this repository comes from **BubbleML version 1**, downloaded from the S3 bucket
`bubble-ml-simulations` (archive `pool-boiling-subcooled-fc72-2d.tar.gz`).
These files have these properties (from `inspect_file.py` on `Twall-100.hdf5`):

- Fluid: FC-72. Geometry: 2D.
- Grid: 384 x 384 cells, domain x in [-6, 6], y in [0, 12].
- Frames: 201, saved every 1.0 non-dimensional time unit (`plotfileintervaltime = 1.0`, `tmax = 200`).
- Fields: `dfun`, `temperature`, `pressure`, `velx`, `vely`, `x`, `y`.

**Assumption (from the brief).** The project uses **Bubbleformer-S on BubbleML 2.0** subcooled pool boiling.

**Assumption (from the BubbleML 2.0 Hugging Face page).** The example metadata there lists `t_final = 100` and `dt = 0.1`,
and a per-example parameter `heater.nucWaitTime`.
The page does not say whether these values apply to the subcooled pool boiling subset.

**Consequence.** If BubbleML 2.0 saves frames every 0.1 time units instead of 1.0,
then a 5-frame window covers 0.5 time units instead of 5,
and one departure interval (about 12 time units here) spans about 120 frames instead of 12.
That change would alter every number below: departure counts per rollout, the probe result, and the rollout length R3 needs.
None of the results here can be applied to the project until this is checked (test T0 in `TEST_PLAN.md`).

## 1. Inventory

| # | Experiment | Script (commit) | Data | Twall | Frames per file | Departures | Seeds | Output |
|---|---|---|---|---|---|---|---|---|
| E1 | Inspect one file | `inspect_file.py` (`4cca98e`) | BubbleML v1 `Twall-100.hdf5` | 100 | 201 | - | - | Printed only. Not saved. |
| E2 | Detect departures | `detect_departures.py` (`d95301b`, comments changed in `4461f46`) | BubbleML v1, 4 files | 100, 103, 106, 110 | 201 | 15, 14, 17, 17 (total 63) | Deterministic | `departures_Twall-*.npy`, `departures_Twall-*.png` |
| E3 | Probe, first version (superseded) | `probe.py` at `309fb0e` | Same 4 files | Same | 201 | Same | 1 | Numbers only in the session transcript. Plot `probe.png` at `309fb0e`, later overwritten. |
| E4 | Probe, 5 seeds | `probe.py` at `4461f46` | Same 4 files | Same | 201 | Same | 5 | `RESULTS.md`, `probe.png`. **No log file was saved.** The raw printed output was in a temporary folder that no longer exists. |
| - | Toy tests (1D signal, CPU) | **Not found** | - | - | - | - | - | Expected a toy script and results in this repo. None exists. |

The HDF5 data and the cached vapor masks (`*_vapor.npy`) are in `/pub/ynagaraj/bubbleml/PoolBoiling-SubCooled-FC72-2D/`, not in the repository.

## 2. Check of the numbers in the brief

Departure statistics were recomputed from `departures_Twall-*.npy` for this document.
Probe numbers come from `RESULTS.md`; they could not be recomputed from a log because none was saved.

| Claim in the brief | Status | Value in this repo |
|---|---|---|
| Toy: timing statistics separate two signals at 15-67x the noise | Not found | No toy code or results in this repo. |
| Toy: hidden phase leaves a 5-frame model 8-12 frames off | Not found | Same. |
| Toy: R3 alone made the histogram about 10x worse | Not found | Same. |
| Toy: waiting-time statistic needs about 20 departures, autocorrelation about 50 | Not found | Same. |
| Mean-only baseline MAE 3.76 | Confirmed | 3.76 (`RESULTS.md`). The baseline has no seed, so it has no spread. |
| 5 vapor frames MAE 2.83 | Confirmed | 2.83 +- 0.24 |
| True time since last departure (with frames) MAE 2.68 | Confirmed | 2.68 +- 0.18. Note: this model receives the frames **and** the time since the last departure. The brief's wording "true time since last departure" omits the frames. |
| Time since last departure only MAE 2.81 | Confirmed | 2.81 +- 0.02 |
| 63 departures in total | Confirmed | 15 + 14 + 17 + 17 = 63 departures, giving 59 intervals. |
| The gaps are smaller than the seed noise | Confirmed | Frames vs frames + time: 2.83 - 2.68 = 0.15, below the seed spread of 0.18-0.24. Frames vs time only: 2.83 - 2.81 = 0.02. |
| 14-17 departures per file in 200 frames | Confirmed | 15, 14, 17, 17 in 201 frames. |
| Waiting-time CV 20-44% | Confirmed (rounding) | Sample std: 0.45, 0.35, 0.20, 0.31 for Twall 100, 103, 106, 110. With population std: 0.44, 0.34, 0.19, 0.30. |
| Mean interval 14.0 frames at Twall 103, 10.8 at Twall 110 | Confirmed | 14.00 and 10.81. |

The "+-" values in `RESULTS.md` are the population standard deviation (`np.std`, line 80 of `probe.py`) of the 5 per-seed means.
The sample standard deviation is about 12% larger (0.27, 0.20, 0.02).

Additional measured statistics (sample std, standard error SE = std / sqrt(number of intervals)):

| Twall | Departures | Intervals | Mean interval | Std | SE of mean |
|---|---|---|---|---|---|
| 100 | 15 | 14 | 12.50 | 5.68 | 1.52 |
| 103 | 14 | 13 | 14.00 | 4.92 | 1.36 |
| 106 | 17 | 16 | 11.44 | 2.28 | 0.57 |
| 110 | 17 | 16 | 10.81 | 3.31 | 0.83 |
| Pooled | 63 | 59 | 12.08 | 4.22 | - |

Of the 59 pooled intervals, 15 are more than 1 std from the pooled mean ("irregular" intervals).

## 3. Method risks

### 3.1 Leakage between train and test

**Present? No.** The probe splits by file, which is the same as splitting by Twall.

```python
# probe.py, lines 68-70
for f, held in enumerate(FILES):
    tr = [torch.cat([data[n][i] for n in FILES if n != held]) for i in range(3)]
    te = data[held]
```

All windows from one file are either in training or in testing, never both.
Windows inside one file overlap, but they stay in the same fold, so the overlap does not leak.
The baseline uses only the training mean (line 71).
No setting is chosen on test data: the epoch count is fixed at 40 (line 16) and there is no early stopping.

**A related problem is present.** The detector thresholds were chosen after the first version found 0 departures,
by looking at frames 40-55 of Twall-100. Twall-100 is also one of the four test files of the probe.
The thresholds were not set before running, which breaks the course rule "set thresholds before running".
This does not leak probe labels into the probe input, but it means the detector was tuned on data it is later evaluated on.

### 3.2 Seeds

**Present? Partly resolved.** E4 runs 5 seeds and reports mean +- std. E3 ran 1 seed.

```python
# probe.py, lines 15 and 78-80
SEEDS = range(5)
overall = {k: v.mean(axis=0) for k, v in mae.items()}        # one number per seed
    print(f"  {k:20s} {v.mean():.2f} +- {v.std():.2f}")
```

Remaining issues:

- The std is the population std (see section 2).
- The seed changes only the network initialization and batch order. The detector and the data split do not change across seeds, so the std does not include detector or data uncertainty.
- The single-seed result of E3 (2.55 for frames only) was lower than the 5-seed mean (2.83). Use only the 5-seed numbers.

### 3.3 Departure detector

**How it works.** The detector labels each connected vapor region (`dfun > 0`) in each frame, then tracks regions from frame t-1 to frame t.

```python
# detect_departures.py, lines 11-14
WALL_ROWS = 2    # a vapor blob that touches the bottom 2 pixel rows is attached to the heater
GROW = 6         # pixels. A bubble moves between frames, so we match it to the next frame with a grown (dilated) mask.
MIN_AREA = 200   # pixels. Smaller wall blobs are short-lived nucleation spots, not bubbles.
KEEP_FRAC = 0.3  # a detached blob counts as the same bubble only if it keeps at least this share of the area
```

```python
# detect_departures.py, lines 46-47
        main = np.bincount(over).argmax()
        if main not in att and (lab == main).sum() >= KEEP_FRAC * blob.sum():
```

- **Heater rows**: the bottom 2 pixel rows (`WALL_ROWS = 2`), which is y < 0.0625. The code treats the whole bottom edge as the heater. It does not read the heater's x-extent from the file.
- **Minimum vapor size**: 200 pixels (`MIN_AREA`), about 0.2 square units.
- **Merging bubbles**: not handled explicitly. Two attached bubbles that merge become one attached region and produce no event. Two bubbles that leave in the same frame produce one departure frame, because the code keeps each frame once (line 51, `sorted({t for t, _ in events})`).
- **Multiple nucleation sites**: not separated. The departure list pools all sites into one sequence. The intervals are therefore between departures from any site, not per-site waiting times.
- **Checked by eye**: one departure only (frame 44 of Twall-100), inspected by printing region extents for frames 40-55. No precision or recall has been measured.
- **Definition differs from the brief.** The brief defines a departure as "the frame where a vapor region stops touching the heater rows".
  The code instead requires the region's **main body** to leave, and ignores small remnant regions that stay on the wall.
  At frame 44 of Twall-100, two small remnants stay attached after the bubble lifts off. The brief's literal rule would not count this departure.

**Also observed.** In Twall-100 the bottom row is mostly vapor across the heater (see `departures_Twall-100.png`), so "one bubble leaves" is only partly well defined there.
Twall-100 has two intervals of 2 and 3 frames, which may be one event counted twice.

### 3.4 Frame spacing

**Present? Yes, a departure can fall between frames.**
Frames are saved every 1.0 time unit (`plotfileintervaltime = 1.0`). The solver step is much smaller (`dtmax = 4e-4`).
A departure is reported at the first saved frame after it happens, so its timing error is up to 1 frame.
The mean interval is about 12 frames, so this resolution error is about 8% of an interval.
The frame spacing in BubbleML 2.0 is not verified (section 0).

### 3.5 Probe inputs

**Present? Yes, the inputs do not match Bubbleformer's inputs.**

```python
# probe.py, lines 20 and 22
    v = np.load(f"{D}/{name}_vapor.npy").astype(np.float32)   # vapor mask, shape (T, 384, 384)
    v = v.reshape(T, 96, 4, 96, 4).mean(axis=(2, 4))          # shrink each frame 4x
```

- The probe sees **one field**: the vapor mask (`dfun > 0`, true or false), not `dfun` itself.
- It sees **5 frames** (`W = 5`, line 14), shrunk from 384 x 384 to 96 x 96 by averaging 4 x 4 blocks.
- **Assumption (from the brief)**: Bubbleformer sees all fields (for example temperature, velocity, and `dfun`) for 5 frames. The Bubbleformer model page does not list its input fields, so this is not verified.
- Consequence: the probe sees less than Bubbleformer would. A positive probe result is evidence that the information exists. A negative result would not show that Bubbleformer's inputs lack it.

### 3.6 Target definition at the ends of a file

**Present? Handled by dropping samples.**

```python
# probe.py, lines 25-31
    for t in range(W - 1, T):                                 # window = frames t-W+1 ... t
        past, future = deps[deps <= t], deps[deps > t]
        ...
        if len(past) == 0 or len(future) == 0:
            continue
        X.append(v[t - W + 1:t + 1]); since.append(t - past.max()); until.append(future.min() - t)
```

- Windows after the last departure are dropped, because they have no next departure.
- Windows before the first departure are also dropped, because they have no "time since last departure".
- With departures between frames 11-17 and 187-194, 14-24 of the 197 windows per file are dropped. 173-183 remain.
- Dropping both ends means the target is never right-censored, which avoids bias. It also means every model is scored on the same windows.

## 4. Claims table

| Claim | Evidence | Strength | What would make it stronger |
|---|---|---|---|
| 5 frames contain information about the time to the next departure | Frames-only MAE 2.83 +- 0.24 vs baseline 3.76, held-out Twall, `RESULTS.md` | Supported, for BubbleML v1 vapor masks | Repeat on BubbleML 2.0 with Bubbleformer's fields and frame spacing (T0, T3). |
| 5 frames contain timing information beyond a regular rhythm | Time-only model 2.81 +- 0.02 equals frames-only 2.83 | Unsupported | Score models on irregular intervals only (T3). |
| Adding the true clock to the frames does not help | 2.68 vs 2.83; difference 0.15 is below the seed spread 0.18-0.24 | Weak. The result is consistent with the claim, but the test cannot detect a difference of this size. | More departures and more files (T0, T3). |
| The detector finds real departures at the right frame | One event checked by eye | Unsupported | Label at least 20 departures by eye and measure precision, recall, and timing error (T1). |
| Timing differs between Twall values | Mean interval 14.0 (SE 1.36) at 103 vs 10.8 (SE 0.83) at 110. Adjacent 106 vs 110: 11.44 (SE 0.57) vs 10.81 (SE 0.83), a gap of 0.63 against a combined SE of about 1.0 | Weak for 103 vs 110 (about 2 SE). Unsupported for adjacent values. | Bootstrap test across all files (T2). |
| Intervals shrink as Twall rises | 14.0, 11.4, 10.8 for 103, 106, 110, but 12.5 at 100 | Weak. Twall-100 breaks the trend. | More files per Twall. |
| One 200-frame rollout has about 15 departures | 14-17 per file | Supported for BubbleML v1 | Recount on BubbleML 2.0 (T0). |
| Toy-test claims | None in this repo | Not verifiable here | Add the toy code and results to the repo. |

## 5. Verdict

**Conditional go.** The one supported result is that 5 vapor frames predict the time to the next departure about 25% better than the mean, at a wall temperature the probe never saw.
This is the minimum the RL plan needs.
Every other claim is weak or unsupported. The frames do no better than a regular-rhythm clock, the detector has not been validated, and adjacent Twall values cannot yet be told apart.
All evidence also comes from BubbleML v1, not the BubbleML 2.0 data that Bubbleformer uses.
**The single biggest open risk** is that one rollout contains too few departures for the R3 timing reward to rise above its own noise.
Here a 200-frame file has only 14-17 departures, below the roughly 20 the brief says the waiting-time statistic needs.
If BubbleML 2.0 saves frames 10 times more often, the count per 200-frame rollout could drop to 1-2.
Tests T0 and T5 measure this risk.

## 6. Recommendations

1. Confirm the BubbleML 2.0 subcooled frame spacing, fields, and Twall values before any further work (T0).
2. Save every run's printed output to a log file in the repo, so results can be checked later without trusting a write-up.
3. Freeze the detector thresholds, and validate them on files not used to choose them (T1).
4. Report sample std, and state which sources of variation the std includes.
