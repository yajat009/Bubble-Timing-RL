# Find the frames where a bubble leaves the heater wall, using the vapor region of one BubbleML file.
# Writes departures_<name>.npy (frame indices) and departures_<name>.png.
import os, sys, json, h5py, numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = sys.argv[1] if len(sys.argv) > 1 else "/pub/ynagaraj/bubbleml/PoolBoiling-SubCooled-FC72-2D/Twall-100.hdf5"
NAME = os.path.basename(PATH).replace(".hdf5", "")
CACHE = PATH.replace(".hdf5", "_vapor.npy")  # vapor mask, bool (T, Y, X). Saved because reading dfun from the HDF5 file is slow.
WALL_ROWS = 2    # a vapor blob that touches the bottom 2 pixel rows is attached to the heater
GROW = 6         # pixels. A bubble moves between frames, so we match it to the next frame with a grown (dilated) mask.
MIN_AREA = 200   # pixels. Smaller wall blobs are short-lived nucleation spots, not bubbles.
KEEP_FRAC = 0.3  # a detached blob counts as the same bubble only if it keeps at least this share of the area
DEBOUNCE_TIME = 1.0  # time units. A new event within this time of an earlier event at the same place is the same departure.
DEBOUNCE_PX = 10     # pixels. "The same place" means within this x distance.

if os.path.exists(CACHE):
    vapor = np.load(CACHE)
else:
    with h5py.File(PATH) as f:
        vapor = f["dfun"][()] > 0
    np.save(CACHE, vapor)
T = len(vapor)

# Time between saved frames. BubbleML 2.0 files have a .json with the time range; BubbleML v1 files do not (1.0 per frame).
meta = PATH.replace(".hdf5", ".json")
FRAME_DT = 1.0
if os.path.exists(meta):
    j = json.load(open(meta))
    FRAME_DT = (j["t_final"] - j["t_initial"]) / (T - 1)
debounce_frames = round(DEBOUNCE_TIME / FRAME_DT)

def label(frame):
    lab, n = ndimage.label(frame)
    attached = np.unique(lab[:WALL_ROWS][lab[:WALL_ROWS] > 0])
    return lab, attached

# A departure at frame t means: a bubble was attached to the wall at frame t-1,
# and at frame t its main body is off the wall. The main body is the blob that overlaps
# the old bubble the most.
# - Small stubs left on the wall after the bubble lifts off do not count against the departure.
# - If the main body is much smaller than the old bubble, the bubble collapsed or broke up.
#   We do not count that as a departure.
events = []  # (t, x_pixel)
prev_lab, prev_att = label(vapor[0])
for t in range(1, T):
    lab, att = label(vapor[t])
    for a in prev_att:
        blob = prev_lab == a
        if blob.sum() < MIN_AREA:
            continue
        over = lab[ndimage.binary_dilation(blob, iterations=GROW) & (lab > 0)]
        if len(over) == 0:
            continue
        main = np.bincount(over).argmax()
        if main not in att and (lab == main).sum() >= KEEP_FRAC * blob.sum():
            events.append((t, int(np.nonzero(blob)[1].mean())))
    prev_lab, prev_att = lab, att

# At fine frame spacing (0.1 time units) a bubble can hover at the wall: it touches, lifts by one row,
# and touches again. Each lift looks like a departure. Keep only the first event of such a sequence.
# An event is dropped if any earlier event (kept or dropped) is within DEBOUNCE_PX in x and
# debounce_frames in time, so a bubble that bounces many times still gives one departure.
kept, seen = [], []
for t, x in sorted(events):
    if not any(t - ts <= debounce_frames and abs(x - xs) <= DEBOUNCE_PX for ts, xs in seen):
        kept.append((t, x))
    seen.append((t, x))
print(f"frame spacing {FRAME_DT} time units; debounce {debounce_frames} frames: {len(events)} raw events -> {len(kept)} departures")
events = kept
np.save(f"departures_{NAME}_events.npy", np.array(events))  # (frame, x pixel) per departure, for per-site analysis
times = np.array(sorted({t for t, _ in events}))  # keep each frame once, even if several bubbles leave in it
print(NAME)
print(f"{len(events)} departure events in {len(times)} distinct frames, T={T}")
print("frames:", times.tolist())
gaps = np.diff(times)
print("inter-departure intervals:", gaps.tolist())
print(f"interval mean={gaps.mean():.2f} median={np.median(gaps):.1f} min={gaps.min()} max={gaps.max()}")
np.save(f"departures_{NAME}.npy", times)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(vapor[:, 0, :].T, origin="lower", aspect="auto", cmap="gray_r", extent=[0, T, 0, vapor.shape[2]])
ax[0].vlines(times, 0, vapor.shape[2], colors="r", lw=0.6, alpha=0.6)
ax[0].set(title="wall-row vapor (black) + departures (red)", xlabel="frame", ylabel="x pixel")
ax[1].plot(vapor.mean(axis=(1, 2))); ax[1].vlines(times, 0, vapor.mean(), colors="r", lw=0.6)
ax[1].set(title="vapor fraction", xlabel="frame")
ax[2].hist(gaps, bins=range(1, gaps.max() + 2)); ax[2].set(title="inter-departure interval", xlabel="frames")
plt.tight_layout(); plt.savefig(f"departures_{NAME}.png", dpi=110)
