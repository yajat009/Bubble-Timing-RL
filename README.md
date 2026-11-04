# Bubble-Timing-RL

Final project for UCI EECS 195/221 (ML for Science, Fall 2026).

## Goal

We want to post-train a boiling emulator with a reward, so that its long rollouts have the right bubble-departure timing.

- **Emulator**: Bubbleformer-S, a transformer that predicts the next 5 frames of a boiling simulation from the previous 5 frames.
- **Data**: BubbleML 2.0, subcooled pool boiling, FC-72 fluid.
- **Departure**: the frame where a vapor bubble leaves the heater wall.
- **Main claim to test**: a timing-based reward lowers timing errors at a held-out wall temperature, with no loss in short-term accuracy.

This repository currently holds the **feasibility checks** that come before any RL.

## Layout

| Folder | Contents |
|---|---|
| `scripts/` | All code. Run every script from the repository root. |
| `results/v1/` | First results, on **BubbleML v1** (1.0 time units per frame). Kept for the record. They do not transfer directly to BubbleML 2.0. |
| `results/v2/` | Results on **BubbleML 2.0**, the project's data (0.1 time units per frame). See `results/v2/RESULTS.md`. |
| `analysis/` | Write-ups: what the results show, and the plan for the next tests. |

## Scripts

| Script | Dataset | Purpose |
|---|---|---|
| `scripts/download_v2.py` | 2.0 | Reads the 4 Bubbleformer fields of chosen files from Hugging Face. Saves a vapor mask and a 128 x 128 copy of the fields. |
| `scripts/detect_departures.py` | both | Finds the frames where bubbles leave the wall. |
| `scripts/download_v1.py` | v1 | Downloads the first N files of the BubbleML v1 archive. |
| `scripts/inspect_v1.py` | v1 | Prints the fields and key parameters of one v1 file. |
| `scripts/probe_v1.py` | v1 | Tests whether 5 frames predict the time to the next departure. |
| `scripts/probe_v2.py` | 2.0 | T3 probe: the same test with Bubbleformer's 4 fields and train/test split. |
| `scripts/median_baseline_v2.py` | 2.0 | Median baseline for the T3 probe. |

## Documents

| File | Contents |
|---|---|
| `analysis/CURRENT_RESULTS.md` | Check of every result and method risk, as of the BubbleML v1 results. |
| `analysis/TEST_PLAN.md` | What each upcoming test (T0-T7) must show, with pass criteria set before running. |
| `analysis/T0_DATA_ALIGNMENT.md` | BubbleML v1 vs 2.0, Bubbleformer's inputs, and first departure results on 2.0. |
| `results/v1/RESULTS.md` | The BubbleML v1 observability probe. |
| `results/v2/RESULTS.md` | Departures on all 17 BubbleML 2.0 files and the T3 probe. |

## Data

Data files are not in this repository.

- BubbleML 2.0: `/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D/` (source: `huggingface.co/datasets/hpcforge/BubbleML_2`).
- BubbleML v1: `/pub/ynagaraj/bubbleml/PoolBoiling-SubCooled-FC72-2D/` (source: S3 bucket `bubble-ml-simulations`).

## Setup

The node's V100 GPU needs the CUDA 12.6 build of PyTorch:

```
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```
