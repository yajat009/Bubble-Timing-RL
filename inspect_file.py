import sys, h5py

path = sys.argv[1] if len(sys.argv) > 1 else "/pub/ynagaraj/bubbleml/PoolBoiling-SubCooled-FC72-2D/Twall-100.hdf5"
KEEP = ("tmax", "plotfileintervaltime", "xmin", "xmax", "ymin", "ymax", "ht_tbulk", "mph_tsat", "ins_gravy")
with h5py.File(path) as f:
    for k, v in f.items():
        print(f"{k:22s} shape={v.shape} dtype={v.dtype}")
    params = {n.decode().strip(): v for n, v in f["real-runtime-params"][()]}
    print({k: params[k] for k in KEEP})
    d = f["dfun"]
    print("timesteps T =", d.shape[0])
    for i in (0, d.shape[0] // 2, d.shape[0] - 1):
        print(f"t={i}: dfun min={d[i].min():.3g} max={d[i].max():.3g} vapor frac={(d[i] > 0).mean():.4f}")
