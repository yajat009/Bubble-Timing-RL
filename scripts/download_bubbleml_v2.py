# Download BubbleML 2.0 subcooled pool boiling data (FC-72) from Hugging Face.
# Usage: python scripts/download_bubbleml_v2.py 97 100 ...   (wall temperatures)
#
# Each HDF5 file is 16.8 GB with 8 fields, stored uncompressed and contiguous.
# This script reads only the 4 fields that Bubbleformer uses, with HTTP range requests,
# and saves per file:
#   Twall_<T>.json           metadata (nucleation sites, wait time, heater size)
#   Twall_<T>_vapor.npy      bool (frames, 512, 512): dfun > 0 at full resolution, for the departure detector
#   Twall_<T>_fields128.npy  float16 (frames, 4, 128, 128): dfun, temperature, velx, vely, each averaged over 4 x 4 cells, for the departure probe
# Outputs that already exist are skipped.
import os, sys, time, urllib.request, aiohttp, fsspec, h5py, numpy as np

BASE_URL = "https://huggingface.co/datasets/hpcforge/BubbleML_2/resolve/main/PoolBoiling-Subcooled-FC72-2D"
OUT_DIR = "/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D"
FIELDS = ["dfun", "temperature", "velx", "vely"]   # Bubbleformer's input fields (poolboiling_subcooled.yaml)
POOL = 4                                           # average 4 x 4 cells: 512 -> 128
BLOCK = 100                                        # frames per read
RETRIES = 5                                        # attempts per block before giving up

def average_cells(frames):
    n, height, width = frames.shape
    return frames.reshape(n, height // POOL, POOL, width // POOL, POOL).mean(axis=(2, 4))

os.makedirs(OUT_DIR, exist_ok=True)
for twall in sys.argv[1:]:
    name = f"Twall_{twall}"
    vapor_path, fields_path = f"{OUT_DIR}/{name}_vapor.npy", f"{OUT_DIR}/{name}_fields128.npy"
    if not os.path.exists(f"{OUT_DIR}/{name}.json"):
        urllib.request.urlretrieve(f"{BASE_URL}/{name}.json", f"{OUT_DIR}/{name}.json")
    if os.path.exists(vapor_path) and os.path.exists(fields_path):
        print(name, "already done", flush=True)
        continue
    # Hugging Face sometimes closes the connection. On an error, reopen the file and retry the same block.
    def open_file():
        for attempt in range(RETRIES):
            try:
                # A read that gets no data for 120 s raises an error, so the retry below starts again instead of waiting forever.
                remote_file = fsspec.open(f"{BASE_URL}/{name}.hdf5", "rb", block_size=2**24,
                                          client_kwargs={"timeout": aiohttp.ClientTimeout(sock_read=120)}).open()
                return remote_file, h5py.File(remote_file)
            except Exception as e:
                print(name, f"open: {type(e).__name__}, retry {attempt + 1} of {RETRIES}", flush=True)
                time.sleep(30)
        raise RuntimeError(f"{name}: could not open after {RETRIES} attempts")
    remote_file, f = open_file()
    T, H, W = f["dfun"].shape
    vapor = np.zeros((T, H, W), dtype=bool)
    fields = np.zeros((T, len(FIELDS), H // POOL, W // POOL), dtype=np.float16)
    for i in range(0, T, BLOCK):
        for attempt in range(RETRIES):
            try:
                for k, field in enumerate(FIELDS):
                    block = f[field][i:i + BLOCK]
                    if field == "dfun":
                        vapor[i:i + BLOCK] = block > 0
                    fields[i:i + BLOCK, k] = average_cells(block)
                break
            except Exception as e:
                print(name, f"frames {i}: {type(e).__name__}, retry {attempt + 1} of {RETRIES}", flush=True)
                time.sleep(30)
                remote_file, f = open_file()
        else:
            raise RuntimeError(f"{name}: frames {i} failed {RETRIES} times")
        print(name, f"frames {i}-{min(i + BLOCK, T) - 1} of {T}", flush=True)
    f.close(); remote_file.close()
    np.save(vapor_path, vapor)
    np.save(fields_path, fields)
    print(name, "saved", fields.shape, flush=True)
