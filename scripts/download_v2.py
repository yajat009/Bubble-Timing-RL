# Download BubbleML 2.0 subcooled pool boiling data (FC-72) from Hugging Face.
# Usage: python scripts/download_v2.py 97 100 ...   (wall temperatures)
#
# Each HDF5 file is 16.8 GB with 8 fields, stored uncompressed and contiguous.
# This script reads only the 4 fields that Bubbleformer uses, with HTTP range requests,
# and saves per file:
#   Twall_<T>.json           metadata (nucleation sites, wait time, heater size)
#   Twall_<T>_vapor.npy      bool (frames, 512, 512): dfun > 0 at full resolution, for the departure detector
#   Twall_<T>_fields128.npy  float16 (frames, 4, 128, 128): dfun, temperature, velx, vely, each averaged over 4 x 4 cells, for the probe
# Outputs that already exist are skipped.
import os, sys, time, urllib.request, fsspec, h5py, numpy as np

BASE = "https://huggingface.co/datasets/hpcforge/BubbleML_2/resolve/main/PoolBoiling-Subcooled-FC72-2D"
OUT = "/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D"
FIELDS = ["dfun", "temperature", "velx", "vely"]   # Bubbleformer's input fields (poolboiling_subcooled.yaml)
POOL = 4                                           # 512 -> 128
BLOCK = 100                                        # frames per read
RETRIES = 5                                        # attempts per block before giving up

def shrink(a):
    n, h, w = a.shape
    return a.reshape(n, h // POOL, POOL, w // POOL, POOL).mean(axis=(2, 4))

os.makedirs(OUT, exist_ok=True)
for t in sys.argv[1:]:
    name = f"Twall_{t}"
    vap_path, fld_path = f"{OUT}/{name}_vapor.npy", f"{OUT}/{name}_fields128.npy"
    if not os.path.exists(f"{OUT}/{name}.json"):
        urllib.request.urlretrieve(f"{BASE}/{name}.json", f"{OUT}/{name}.json")
    if os.path.exists(vap_path) and os.path.exists(fld_path):
        print(name, "already done", flush=True)
        continue
    # Hugging Face sometimes closes the connection. On an error, reopen the file and retry the same block.
    def open_file():
        for attempt in range(RETRIES):
            try:
                fo = fsspec.open(f"{BASE}/{name}.hdf5", "rb", block_size=2**24).open()
                return fo, h5py.File(fo)
            except Exception as e:
                print(name, f"open: {type(e).__name__}, retry {attempt + 1} of {RETRIES}", flush=True)
                time.sleep(30)
        raise RuntimeError(f"{name}: could not open after {RETRIES} attempts")
    fo, f = open_file()
    T, H, W = f["dfun"].shape
    vapor = np.zeros((T, H, W), dtype=bool)
    fields = np.zeros((T, len(FIELDS), H // POOL, W // POOL), dtype=np.float16)
    for i in range(0, T, BLOCK):
        for attempt in range(RETRIES):
            try:
                for k, fld in enumerate(FIELDS):
                    block = f[fld][i:i + BLOCK]
                    if fld == "dfun":
                        vapor[i:i + BLOCK] = block > 0
                    fields[i:i + BLOCK, k] = shrink(block)
                break
            except Exception as e:
                print(name, f"frames {i}: {type(e).__name__}, retry {attempt + 1} of {RETRIES}", flush=True)
                time.sleep(30)
                fo, f = open_file()
        else:
            raise RuntimeError(f"{name}: frames {i} failed {RETRIES} times")
        print(name, f"frames {i}-{min(i + BLOCK, T) - 1} of {T}", flush=True)
    f.close(); fo.close()
    np.save(vap_path, vapor)
    np.save(fld_path, fields)
    print(name, "saved", fields.shape, flush=True)
