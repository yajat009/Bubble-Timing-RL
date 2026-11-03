# Test plan: what each upcoming test must provide or prove

This plan follows from `CURRENT_RESULTS.md`.
Every pass criterion below is set now, before any of these tests runs.
A test that misses its criterion counts as failed, even if the miss is small.

Statement types:

- **Measured**: a number already in this repository.
- **Assumption**: something we believe but have not verified.
- **Recommendation**: a choice we propose.

Terms:

- **Departure, interval, CV, MAE, probe, Twall**: defined in `CURRENT_RESULTS.md`.
- **Rollout**: a sequence of frames that the emulator generates from one start state.
- **Group**: several rollouts from the same start state, scored together by the RL method.
- **R2, R3, R4**: the snapshot-statistics, timing-statistics, and combined rewards from the brief.
- **Noise floor**: how much a statistic varies between two real simulation segments that have the same true timing. A reward difference smaller than this tells us nothing.
- **Wasserstein-1 (W1) distance**: a distance between two distributions. For intervals it is the average amount, in frames, by which one set of intervals must shift to match the other.
- **Seed std**: the sample standard deviation over seeds of a model's mean test MAE.

## Model names

`RESULTS.md` and this plan use different letters for the same models. This plan uses the brief's letters.

| This plan (brief) | `RESULTS.md` | Input |
|---|---|---|
| (a) mean only | baseline | none |
| (b) time only | (c) since only | true time since the last departure |
| (c) 5 frames | (a) frames | 5 frames |
| (d) frames + time | (b) frames + since | 5 frames and the true time since the last departure |

## Changes to the proposed test list, and why

1. **New test T0 (data alignment), first.** Every result so far uses BubbleML v1 (1.0 time unit per frame). The project uses BubbleML 2.0, and its Hugging Face page lists `dt = 0.1` for its example. If that applies to subcooled pool boiling, departure counts, window coverage, and rollout lengths all change by a factor of 10.
2. **T4 moves earlier.** The BubbleML 2.0 page lists `heater.nucWaitTime`, a nucleation wait time per simulation. A wait time is a candidate hidden clock: while a site waits, the frames may show nothing at that site. T4 is cheap and changes how T3 should be read.
3. **T5 moves before T6.** T6's pass criterion uses the noise floor that T5 measures.
4. **T1 adds a rule.** The detector thresholds were chosen by looking at Twall-100. Validation must use at least one file that was not used to choose thresholds.
5. **T3 adds a criterion.** In the current data, the time-only model (b) matches the frames model (c), so beating the mean does not show that the frames contain more than a regular rhythm. T3 now also requires (c) to beat (b) on irregular intervals.
6. **T5 depends on a heat-flux calculation that does not exist yet.** Two of the three R3 statistics (heat-flux autocorrelation, heat flux around each departure) need wall heat flux. No code in this repo computes it.

## Order and check-in status

| Order | Test | Needed for Nov 12 check-in | Hardware | Blocks |
|---|---|---|---|---|
| 1 | T0 Data alignment | Yes | CPU | All other tests |
| 2 | T1 Detector validation | Yes | CPU | T2, T3, T5, T6 |
| 3 | T4 Hidden solver clock | Recommended (cheap) | CPU | How T3 is read |
| 4 | T2 Departure statistics per Twall | Yes | CPU | Whether a held-out-Twall test can show anything |
| 5 | T5 Rollout length and noise floor | Yes (T6 needs it) | CPU | T6, T7, GPU budget |
| 6 | T3 Observability probe, redone | Yes | 1 GPU, under 1 hour (estimate) | Choice of input window |
| 7 | T6 Baseline timing error | Yes | GPU, cost unknown; ask before running | Project motivation |
| 8 | T7 Spread within a group | No | GPU | RL training |

---

## T0. Data alignment (new; check-in; blocking)

**Question.** Do the BubbleML 2.0 subcooled pool boiling files that Bubbleformer uses have the same frame spacing, fields, and departure counts as the BubbleML v1 files used so far?

**Why the project needs it.** All current numbers come from BubbleML v1. If the frame spacing or fields differ, no current number applies to the project, and the rollout lengths in T5 and T6 are wrong.

**Method.**
1. Download the BubbleML 2.0 subcooled pool boiling metadata (`.json`) and one `.hdf5` file from `hpcforge/BubbleML_2`.
2. Record the fields, grid size, number of frames, time between saved frames, list of Twall values, and train/test split.
3. Read Bubbleformer-S's input fields, number of input frames, and frame spacing from its code or paper.
4. Run the detector on 2 files from 2 different Twall values.
5. Runtime estimate: under 1 hour, mostly download time.

**Output.**
- A table that compares BubbleML v1 and BubbleML 2.0: fluid, grid, frames per file, time per frame, fields, Twall values.
- Bubbleformer-S inputs: field names, number of frames, resolution.
- Departures per file and mean interval, in frames and in time units, for the 2 files.

**Pass criterion.** The frame spacing and the Bubbleformer input fields are known from a primary source (data file, code, or paper), not inferred.

**If it fails.** Do not run T3, T5, or T6 on BubbleML 2.0 until the spacing and fields are known. If the spacing is 0.1 time units, redo the numbers in `CURRENT_RESULTS.md` with all window and rollout lengths scaled by 10.

---

## T1. Departure detector validation (check-in)

**Question.** Does the detector find real departures, at the right frame?

**Why the project needs it.** Every timing statistic, every probe label, and the R3 reward depend on the detector. A detector that misses or invents departures makes all of them wrong.

**Method.**
1. Write the labeling rule before labeling: a departure is the first frame where a bubble's main vapor body has no pixel in the bottom 2 rows. Remnants smaller than 30% of the bubble are ignored.
2. Choose segments of at least 60 consecutive frames from at least 2 Twall values. At least one file must be a file not used to choose the thresholds (that is, not Twall-100).
3. Label **every** departure in those segments by eye, without looking at the detector output. Labeling every frame, not only detected events, is what allows recall to be measured.
4. Match labeled and detected departures within 2 frames. Count true positives, false positives, and false negatives.
5. Freeze `WALL_ROWS`, `GROW`, `MIN_AREA`, and `KEEP_FRAC` before labeling. Do not change them after seeing the result.
6. Runtime: about 2 hours of human labeling, plus minutes of CPU.

**Output.**
- Precision, recall, and mean absolute timing error (frames), per Twall and pooled.
- The number of labeled departures (at least 20).
- A figure of `dfun` frames around 3 example departures, with the detected frame and the labeled frame marked.
- A list of every disagreement, with its frame number.

**Pass criterion.** Precision ≥ 0.9, recall ≥ 0.9, and mean absolute timing error ≤ 1 frame, pooled over at least 20 labeled departures.

**If it fails.**
- If precision is low (extra events): look for double counts during bubble merging. Add a rule that merges events closer than a fixed number of frames, set before re-labeling.
- If recall is low (missed events): look for departures hidden by merged vapor on the wall. Consider detecting departures per nucleation site instead of over the whole wall.
- Re-validate on new segments. Do not re-use the failed segments for the new check.

---

## T4. Hidden solver clock (recommended for check-in)

**Question.** Is the departure interval controlled by a solver parameter, such as a nucleation wait time, that the frames do not show?

**Why the project needs it.** If a site waits a fixed time with no visible change before a new bubble forms, a 5-frame window cannot tell how far into the wait it is. Then no 5-frame model can learn the timing, and R3 cannot be learned either.

**Method.**
1. **Measured already:** the BubbleML v1 file `Twall-100.hdf5` stores 109 real and 79 integer runtime parameters. None of them is a wait time. The closest is `sim_nucseedradius = 0.1`.
2. **Assumption:** the BubbleML 2.0 page lists `heater.nucWaitTime` per example. Read its value for each subcooled file.
3. Convert the wait time to frames using the frame spacing from T0.
4. Plot the interval histogram, in frames, with the wait time and its multiples marked.
5. For each departure, count the frames during which the wall site shows no vapor before the next bubble forms. This measures how long the frames stay unchanged.
6. Runtime: under 1 hour on CPU.

**Output.**
- The parameter name and value for each file, or "none found".
- The interval histogram with the wait time marked.
- The distribution of the number of no-vapor frames before each new bubble forms.

**Pass criterion.** The median number of no-vapor frames before a new bubble forms is ≤ 1 frame. This means at most about 1 frame of the cycle is hidden from the input.

**If it fails.** The wait is a hidden clock. Choose one, in this order:
1. Give the model the elapsed time since the last departure as an extra input.
2. Lengthen the input window to cover the wait.
3. Report the failure as a result: a 5-frame Markovian emulator cannot learn this timing.

---

## T2. Departure statistics per Twall (check-in)

**Question.** Do the timing statistics differ between wall temperatures by more than their own noise?

**Why the project needs it.** The main claim is tested at a held-out Twall. If neighbouring Twall values have the same timing within noise, no reward can show a timing improvement there.

**Method.**
1. Run the frozen detector from T1 on every available subcooled file.
2. For each Twall, compute the departure count, mean interval, std, and CV.
3. Bootstrap: resample the intervals of each Twall with replacement, 2000 times.
4. **Within-Twall distance**: the W1 distance between two independent bootstrap resamples of the same Twall. Repeat 2000 times. Take the 95th percentile.
5. **Between-Twall distance**: the W1 distance between the observed intervals of two adjacent Twall values.
6. Also bootstrap the mean interval and report its 95% interval.
7. Runtime: minutes on CPU, once the departures are detected.

**Output.**
- A table per Twall: count, mean interval with 95% bootstrap interval, std, and CV.
- For each adjacent pair: between-Twall W1, and the within-Twall 95th percentile for each of the two values.
- A plot of the interval distributions per Twall.

**Pass criterion.** For every adjacent Twall pair, the between-Twall W1 is larger than the within-Twall 95th percentile of both Twall values.

**Expectation (not measured).** With the current v1 files, 106 vs 110 will probably fail. Their mean intervals differ by 0.63 frames with a combined standard error of about 1.0 frame.

**If it fails.**
- Hold out a Twall farther from the training values, so its timing differs more.
- Or pool several simulations per Twall to reduce noise.
- Or change the main claim from "adjacent held-out Twall" to "held-out Twall at the edge of the training range".

---

## T5. Rollout length and noise floor of the reward (check-in; needed by T6)

**Question.** How long must a rollout, or how large must a group be, for each R3 statistic to tell correct timing from wrong timing?

**Why the project needs it.** Group-relative RL ranks rollouts by reward. If a statistic ranks a correct-timing rollout below a wrong-timing one by chance, the RL step learns from noise. This test also sets the GPU budget, because rollout length drives cost.

**Method.**
1. Use real simulation segments as stand-ins for rollouts.
2. Segment lengths: 50, 100, 200, and 400 frames. A BubbleML v1 file has only 201 frames, so the 400-frame case needs BubbleML 2.0 files or concatenated files. T0 decides which.
3. Wrong-timing stand-ins:
   - (i) a segment of the same length from another Twall;
   - (ii) a time-stretched copy of a real segment, with every interval multiplied by 0.9 and by 1.1.
4. Compute each R3 statistic against a reference from the same Twall: waiting-time W1, heat-flux autocorrelation distance, and heat flux around each departure.
5. A trial is correct when the real segment's distance to the reference is smaller than the wrong-timing segment's distance.
6. Repeat over at least 200 random segment pairs per length.
7. **Prerequisite:** write and check a wall heat-flux calculation. None exists in this repo.
8. Runtime: under 2 hours on CPU, after the heat-flux code exists.

**Output.**
- For each statistic and each length: the fraction of correct rankings against (i) and against (ii).
- The minimum length, or the minimum group pooling size, at which the fraction reaches 80%.
- The noise floor: the 95th percentile of each statistic between two real segments of the same Twall.
- An updated GPU-hour estimate for RL training, based on that length and the per-frame rollout cost from T6.

**Pass criterion.** At least one R3 statistic reaches 80% correct ranking against both (i) and (ii) at a length of 400 frames or less, or at a pooled group of 8 rollouts or less.

**If it fails.**
- Pool timing statistics across the whole group instead of per rollout.
- Or use longer rollouts, if the GPU budget allows.
- Or drop the statistics that fail, and keep only the ones that pass.

---

## T3. Observability probe, redone properly (check-in)

**Question.** Do Bubbleformer's 5 input frames contain the cycle phase, or do they not?

**Why the project needs it.** Bubbleformer is Markovian: it predicts from its last 5 frames only (assumption from the brief). If those frames do not show where the system is in the cycle, no reward can teach it correct timing. The toy tests in the brief suggest this case leaves a model 8-12 frames off.

**Method.**
1. Data: all available subcooled files, using the T0 frame spacing.
2. Split: leave one Twall out. Report each fold.
3. Input: the same fields and resolution that Bubbleformer sees (from T0), for 5 frames.
4. Models: (a) mean only, (b) time only, (c) 5 frames, (d) 5 frames + time.
5. Seeds: 5. Report the mean and the seed std (sample std).
6. Irregular intervals: intervals more than 1 std from the **training-fold** mean. Set this rule now. In the current v1 data, 15 of 59 intervals meet it.
7. Save the full printed output to `logs/`.
8. Runtime estimate: the current v1 run (4 folds, 5 seeds, 3 networks) finished in under 1 hour on one V100. Most of that time was slow reads from the shared filesystem. More fields and more files may double it.

**Output.**
- A table of test MAE (mean ± seed std) for (a)-(d), per fold and pooled.
- The same table for windows inside irregular intervals only.
- A predicted-vs-true plot for (c) and (d).

**Pass criterion.** All three must hold:
1. (c) beats (a) by more than 2 seed std of (c).
2. |(d) - (c)| is within 1 seed std of (c).
3. On irregular intervals, (c) beats (b) by more than 2 seed std of (c).

**Current v1 values against these criteria (measured; not a substitute for T3).**
- Criterion 1: 3.76 - 2.83 = 0.93, which is more than 2 × 0.27 = 0.54. Pass.
- Criterion 2: 2.83 - 2.68 = 0.15, which is less than 0.27. Pass.
- Criterion 3: not measured.

**If it fails.** Choose one:
- If criterion 1 fails: the frames do not show the phase. Use a longer input window, or add the elapsed time as an input.
- If criterion 2 fails (the clock helps): the frames show only part of the phase. Use the same fixes as above.
- If criterion 3 fails: the frames show only a regular rhythm. Report this as a limit of the project, and test the main claim only where the rhythm is regular.

---

## T6. Does the baseline get timing wrong? (check-in)

**Question.** Does Bubbleformer-S, without RL, produce departure timing that is measurably wrong?

**Why the project needs it.** If the baseline already has correct timing, the RL step has nothing to fix, and the project's motivation fails.

**Method.**
1. **Recommendation:** start from the released Bubbleformer-S pool-boiling-subcooled checkpoint on Hugging Face (`hpcforge/Bubbleformer`, listed as "PB-Subcooled"). This avoids retraining.
2. Reproduce its reported one-step and 10-frame accuracy on the BubbleML 2.0 test split. Record the reported values and our values side by side.
3. Roll it out from real start states for the length found in T5, at the training Twall values and at the held-out Twall.
4. Detect departures in the rollouts with the frozen detector from T1.
5. Compute the R2 and R3 statistics against the real simulation from the same start.
6. Seeds: 5 start states per Twall (Bubbleformer-S is deterministic, so the start state is the source of variation).
7. **Cost: unknown.** First time one rollout, then give a GPU-hour estimate and ask before running more than about 1 GPU-hour.

**Output.**
- Reported vs reproduced accuracy.
- R2 and R3 errors per Twall, mean ± std over start states.
- Interval histograms of rollouts and real simulations, side by side.
- The time per rollout frame on one GPU.

**Pass criterion.**
1. Reproduced 10-frame accuracy is within 10% of the reported value.
2. At the held-out Twall, the baseline's R3 error is above the 95th-percentile noise floor from T5 for at least one R3 statistic.

**If it fails.**
- If accuracy does not reproduce: fix the evaluation before anything else. All later comparisons would be invalid.
- If timing is already correct: report that, and change the target. For example, target a regime where timing is wrong, such as a Twall outside the training range or longer rollouts.

---

## T7. Spread within a group

**Question.** Do rollouts from the flow-matching head differ in their timing statistics?

**Why the project needs it.** Group-relative RL compares rollouts in a group. If all rollouts get the same reward, the method learns nothing.

**Method.**
1. **Prerequisite:** the flow-matching head is built and trained. It does not exist yet.
2. From one start state, generate 8 rollouts of the T5 length.
3. Compute each R3 statistic for each rollout.
4. Repeat for 5 start states.
5. Cost: depends on T6's per-frame time. Ask before running.

**Output.**
- For each statistic: the std across the 8 rollouts, averaged over start states.
- The ratio of this std to the T5 noise floor (the std between real segments of the same Twall).

**Pass criterion.** For at least one R3 statistic, the within-group std is greater than zero and at least 0.5 times the T5 noise floor.

**If it fails.**
- Increase the head's sampling noise.
- Or use longer rollouts, so timing differences can build up.
- Or use pooled group statistics (from T5) as the reward.

---

## Summary

**Blocking tests.** T0 blocks everything, because all current results use BubbleML v1, not the project's BubbleML 2.0 data. T1 blocks every test that uses departure labels.

**Can run now, on CPU, with data already on disk.**
- T1 labeling on the 4 BubbleML v1 files. This checks the detector logic, even before T0.
- T2 as a preliminary check on the 4 v1 files.
- T4 step 1 (done: no wait-time parameter in the v1 files) and the no-vapor-frames count on v1.

**Needs GPU.** T3 (under 1 hour, estimated). T6 and T7 (cost unknown; ask first).

**Needs new code.** Wall heat flux (for T5). The flow-matching head (for T7).

**Suggested 5 check-in slides (Nov 12).**
1. Goal and plan, with the revised test order and the BubbleML v1 vs 2.0 issue.
2. Detector validation (T1): precision, recall, timing error, and the 3-departure figure.
3. Timing statistics per Twall (T2): intervals per Twall with bootstrap intervals, and which pairs can be told apart.
4. Observability probe (T3): MAE of (a)-(d) with seed std, on all and on irregular intervals.
5. Baseline gap (T6) against the noise floor (T5), and the revised plan if any test failed.
