# T0: data alignment with BubbleML 2.0

This test checks whether the project's data (BubbleML 2.0, used by Bubbleformer) matches the BubbleML v1 data used for the first results.
It does not match. The results in `results/v1/RESULTS.md` do not apply directly to the project.

Statement types: **measured** (computed here), **source** (read from a primary file), **assumption**, **recommendation**.

## 1. Data facts

| Property | BubbleML v1 (first results) | BubbleML 2.0 (project data) | Type |
|---|---|---|---|
| Location | S3 `bubble-ml-simulations`, archive `pool-boiling-subcooled-fc72-2d.tar.gz` | Hugging Face `hpcforge/BubbleML_2`, folder `PoolBoiling-Subcooled-FC72-2D` | source |
| Fluid | FC-72 | FC-72 | source |
| Files | 4 used (Twall 100, 103, 106, 110) | 20 (Twall 85-117), 16.8 GB each | source |
| Grid | 384 x 384 | 512 x 512 | source |
| Domain | x in [-6, 6], y in [0, 12] | x in [-8, 8], y in [0, 16] | source |
| Cell size | 0.03125 | 0.03125 | source |
| Frames per file | 201 | 2001 | source (HDF5 shapes) |
| Time between frames | 1.0 | **0.1** (t = 0 to 200 over 2001 frames) | source |
| Fields | dfun, temperature, pressure, velx, vely | dfun, temperature, pressure, velx, vely, massflux, normx, normy | source |
| Nucleation wait time | not stored | `heater.nucWaitTime = 0.4` (4 frames), same in all 5 files checked | source (JSON) |
| Nucleation sites | not stored | fixed x positions; 3 (Twall 85), 15 (97), 18 (100), 24 (106), 28 (110), 35 (117) | source (JSON) |
| Heater x-extent | not stored | [-5.25, 5.25] | source (JSON) |

The JSON field `dt` is not the frame spacing: it is 0.1 in some files and 5.0 in others, while all files have 2001 frames over t = 0-200.

## 2. Bubbleformer inputs

Source: `bubbleformer/config/data_cfg/poolboiling_subcooled.yaml` in `github.com/HPCForge/Bubbleformer`.

| Setting | Value |
|---|---|
| Input and output fields | dfun, temperature, velx, vely |
| Time window | 5 frames in, 5 frames out |
| Downsample factor | 1 (full 512 x 512) |
| Start time | 100 (the first 100 frames are not used) |
| FC-72 training files | Twall 86, 88, 90, ..., 116 (16 files) |
| FC-72 validation file | Twall 97 |
| FC-72 files in neither list | Twall 85, 107, 117 |

**Consequence.** A 5-frame input window covers 0.5 time units, not 5 as in BubbleML v1.

## 3. Detector on BubbleML 2.0 (Twall 97 and 100)

### 3.1 A problem found and fixed

**Measured.** With the BubbleML v1 settings, the detector found 216 events (Twall 97) and 284 (Twall 100).
Printing the events in frames 200-300 of Twall 100 showed the same bubble counted several times.
For example, at x ≈ 147 px there were events at frames 216, 218, and 223.
In each case the bubble's bottom was at pixel row 2, one row above the 2 heater rows.
The bubble touched the wall, lifted by one row, and touched again.
At the v1 spacing of 1.0 time units a bubble moves about 40 px between frames, so this did not occur there.

**Fix.** A new event within `DEBOUNCE_TIME = 1.0` time units and `DEBOUNCE_PX = 10` px of an earlier event is treated as the same departure.
The window is set in time units, so it means the same thing for both datasets.
10 px (0.31 units) is about half of the spacing between nucleation sites at Twall 100 (about 0.6 units).

**Rule violation.** These two values were chosen after looking at Twall 100. T1 must validate them on other files, with the values frozen.

**Not re-run.** The BubbleML v1 departure files (`results/v1/departures_Twall-*.npy`) were made before the debounce was added (commit `4461f46`). They are unchanged.

### 3.2 Results

All numbers are measured. "After frame 100" applies Bubbleformer's start time.

| | Twall 97 | Twall 100 |
|---|---|---|
| Raw events, then after debounce | 216, then 170 | 284, then 199 |
| Departures after frame 100 | 162 | 190 |
| Departures in frames 200-300 | 11 | 13 |
| Share within 10 px of a known nucleation site | 97% | 98% |
| Pooled mean interval after frame 100 | 12.0 frames (1.2 time units) | 10.1 frames (1.0 time units) |
| Pooled CV | 0.80 | 0.84 |
| Sites with at least 3 departures | 12 of 15 | 16 of 18 |
| Departures per site, median (range) | 12 (4-27) | 11 (5-21) |
| Per-site mean interval, median (range) | 145 frames (70-329) | 153 frames (84-244) |
| Per-site CV, median | 0.97 | 0.93 |

Files (in `results/v2/`): `departures_Twall_97*.npy`, `departures_Twall_97.png`, `departures_Twall_100*.npy`, `departures_Twall_100.png`.
The `_events.npy` files hold (frame, x pixel) for each departure.

### 3.3 What the results mean

- **The events are probably real departures.** 97-98% of them start within 10 px of a nucleation site listed in the metadata. A detector that invented events would not line up with the sites this well. This is a sanity check, not a substitute for T1.
- **Timing is irregular, at both levels.** The CV is 0.8 for the pooled sequence and about 0.95 per site. A CV near 1 is what random (Poisson) timing gives. In BubbleML v1 the pooled CV was 0.20-0.45.
- **The wait time is short compared with a site's cycle.** `nucWaitTime` is 4 frames; a site's mean interval is about 150 frames. So the wait is not the main clock of a site.
- **Departure counts are higher than in v1.** About 10 pooled departures per 100 frames, so a 200-frame rollout contains about 20 departures. A v1 file of 201 frames contained 14-17.
- **A 5-frame window sees about 3% of one site's cycle** (5 of about 150 frames).

## 4. Effect on the other tests

| Test | Change |
|---|---|
| T1 | Validate the debounced detector on files other than Twall 100, for example Twall 97 and 110, with all thresholds frozen. |
| T2 | Use BubbleML 2.0 files. Report both pooled and per-site statistics. |
| T3 | The v1 result (frames beat the mean, but match a time-only model) does not transfer. With CV near 1, a regular-rhythm guess will fail, so criterion 3 is now a strong test. Use dfun, temperature, velx, vely, 5 frames, Bubbleformer's split (train on even Twall 86-116, test on 97). |
| T4 | Answered in part: the wait time is 0.4 (4 frames) in every file checked, much shorter than a site's cycle. It remains open whether the frames show the state of each site during its cycle. |
| T5 | Pooled statistics give about 20 departures per 200 frames. Per-site statistics give about 1-2 per site per 200 frames, so per-site timing statistics need much longer rollouts or pooling across sites. |

## 5. Recommendations

1. Run T3 on BubbleML 2.0 with Bubbleformer's fields and split. This needs 4 fields from 17 files (16 training files and Twall 97).
2. Decide whether R3 should use pooled or per-site departure statistics. The pooled sequence has enough departures per rollout. The per-site sequence is closer to the physics but has too few.
