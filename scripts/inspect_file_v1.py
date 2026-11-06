# Print the fields, shapes, and a few key parameters of one BubbleML HDF5 file.
# Usage: python scripts/inspect_file_v1.py [path]
import sys, h5py

path = sys.argv[1] if len(sys.argv) > 1 else "/pub/ynagaraj/bubbleml/PoolBoiling-SubCooled-FC72-2D/Twall-100.hdf5"
KEY_PARAMS = ("tmax", "plotfileintervaltime", "xmin", "xmax", "ymin", "ymax", "ht_tbulk", "mph_tsat", "ins_gravy")
with h5py.File(path) as f:
    for k, v in f.items():
        print(f"{k:22s} shape={v.shape} dtype={v.dtype}")
    params = {n.decode().strip(): v for n, v in f["real-runtime-params"][()]}
    print({k: params[k] for k in KEY_PARAMS})
    dfun = f["dfun"]
    print("frames =", dfun.shape[0])
    for i in (0, dfun.shape[0] // 2, dfun.shape[0] - 1):
        print(f"frame {i}: dfun min={dfun[i].min():.3g} max={dfun[i].max():.3g} vapor fraction={(dfun[i] > 0).mean():.4f}")
