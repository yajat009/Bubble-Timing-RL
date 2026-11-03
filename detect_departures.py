import os, sys, h5py, numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = sys.argv[1] if len(sys.argv) > 1 else "/pub/ynagaraj/bubbleml/PoolBoiling-SubCooled-FC72-2D/Twall-100.hdf5"
CACHE = PATH.replace(".hdf5", "_vapor.npy")  # bool (T,Y,X); reading dfun from hdf5 is slow
WALL_ROWS = 2    # a blob touching the first rows is attached to the heater
GROW = 6         # px; blobs move between frames, so match with a dilated mask
MIN_AREA = 200   # px; smaller wall blobs are nucleation flicker, not bubbles
KEEP_FRAC = 0.3  # the detached successor must keep this share of the bubble's area

if os.path.exists(CACHE):
    vapor = np.load(CACHE)
else:
    with h5py.File(PATH) as f:
        vapor = f["dfun"][()] > 0
    np.save(CACHE, vapor)
T = len(vapor)

def label(frame):
    lab, n = ndimage.label(frame)
    attached = np.unique(lab[:WALL_ROWS][lab[:WALL_ROWS] > 0])
    return lab, attached

# Departure at frame t: a bubble attached at t-1 whose main body (the blob it overlaps most
# at t) is off the wall. Small stubs left on the wall are ignored; if the main body is
# tiny the bubble collapsed or fragmented, which is not a departure.
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

times = np.array(sorted({t for t, _ in events}))  # several blobs may leave in one frame
print(f"{len(events)} departure events in {len(times)} distinct frames, T={T}")
print("frames:", times.tolist())
gaps = np.diff(times)
print("inter-departure intervals:", gaps.tolist())
print(f"interval mean={gaps.mean():.2f} median={np.median(gaps):.1f} min={gaps.min()} max={gaps.max()}")
np.save("departures.npy", times)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(vapor[:, 0, :].T, origin="lower", aspect="auto", cmap="gray_r", extent=[0, T, 0, vapor.shape[2]])
ax[0].vlines(times, 0, vapor.shape[2], colors="r", lw=0.6, alpha=0.6)
ax[0].set(title="wall-row vapor (black) + departures (red)", xlabel="frame", ylabel="x pixel")
ax[1].plot(vapor.mean(axis=(1, 2))); ax[1].vlines(times, 0, vapor.mean(), colors="r", lw=0.6)
ax[1].set(title="vapor fraction", xlabel="frame")
ax[2].hist(gaps, bins=range(1, gaps.max() + 2)); ax[2].set(title="inter-departure interval", xlabel="frames")
plt.tight_layout(); plt.savefig("departures.png", dpi=110)
