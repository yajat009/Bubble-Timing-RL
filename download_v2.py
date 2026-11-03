# Download the vapor mask (dfun > 0) of BubbleML 2.0 subcooled pool boiling files (FC-72) from Hugging Face.
# Usage: python download_v2.py 97 100 ...   (wall temperatures)
# Each HDF5 file is 16.8 GB with 8 fields. The fields are stored uncompressed and contiguous,
# so this script reads only the dfun field (about 2 GB) with HTTP range requests.
# Output per file: <OUT>/Twall_<T>_vapor.npy (bool, frames x 512 x 512) and <OUT>/Twall_<T>.json (metadata).
import sys, urllib.request, fsspec, h5py, numpy as np

BASE = "https://huggingface.co/datasets/hpcforge/BubbleML_2/resolve/main/PoolBoiling-Subcooled-FC72-2D"
OUT = "/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D"
BLOCK = 100  # frames per read

for t in sys.argv[1:]:
    name = f"Twall_{t}"
    urllib.request.urlretrieve(f"{BASE}/{name}.json", f"{OUT}/{name}.json")
    with fsspec.open(f"{BASE}/{name}.hdf5", "rb", block_size=2**24) as fo, h5py.File(fo) as f:
        d = f["dfun"]
        vapor = np.zeros(d.shape, dtype=bool)
        for i in range(0, d.shape[0], BLOCK):
            vapor[i:i + BLOCK] = d[i:i + BLOCK] > 0
            print(name, f"frames {i}-{min(i + BLOCK, d.shape[0]) - 1} of {d.shape[0]}", flush=True)
    np.save(f"{OUT}/{name}_vapor.npy", vapor)
    print(name, "saved", vapor.shape, f"vapor fraction {vapor.mean():.4f}", flush=True)
