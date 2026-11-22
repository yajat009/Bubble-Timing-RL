# Download the 4 Bubbleformer fields of BubbleML 2.0 files at the full 512 x 512 resolution, in float16.
# Usage: python scripts/download_fullres_v2.py 90 94 ...   (wall temperatures)
#
# Saves <DATA>/Twall_<T>_fields512.npy, float16, shape (frames, 4, 512, 512), about 4.2 GB per file.
# The head-training data and the post-training start states need these frames (download_bubbleml_v2.py keeps
# only the 128 x 128 averages). The download can resume: finished blocks are listed in Twall_<T>_fields512.done.
import os, signal, sys, time, aiohttp, fsspec, h5py, numpy as np

BASE_URL = "https://huggingface.co/datasets/hpcforge/BubbleML_2/resolve/main/PoolBoiling-Subcooled-FC72-2D"
OUT_DIR = "/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D"
FIELDS = ["dfun", "temperature", "velx", "vely"]
BLOCK, RETRIES = 100, 5
BLOCK_TIMEOUT = 600   # seconds. A block that takes longer is read again. (The socket timeout alone did not stop every hang.)


def on_timeout(signum, frame):
    raise TimeoutError(f"no block finished in {BLOCK_TIMEOUT} s")


signal.signal(signal.SIGALRM, on_timeout)


def open_file(name):
    """Open the remote HDF5 file. A read that gets no data for 120 s raises an error, so a retry can start."""
    for attempt in range(RETRIES):
        try:
            remote = fsspec.open(f"{BASE_URL}/{name}.hdf5", "rb", block_size=2**24,
                                 client_kwargs={"timeout": aiohttp.ClientTimeout(sock_read=120)}).open()
            return remote, h5py.File(remote)
        except Exception as e:
            print(name, f"open: {type(e).__name__}, retry {attempt + 1} of {RETRIES}", flush=True)
            time.sleep(30)
    raise RuntimeError(f"{name}: could not open after {RETRIES} attempts")


for twall in sys.argv[1:]:
    name = f"Twall_{twall}"
    path, done_path = f"{OUT_DIR}/{name}_fields512.npy", f"{OUT_DIR}/{name}_fields512.done"
    done = set(int(l) for l in open(done_path)) if os.path.exists(done_path) else set()
    remote, h5_file = open_file(name)
    n_frames, height, width = h5_file["dfun"].shape
    output = np.lib.format.open_memmap(path, mode="r+" if os.path.exists(path) else "w+",
                                    dtype=np.float16, shape=(n_frames, len(FIELDS), height, width))
    for i in range(0, n_frames, BLOCK):
        if i in done:
            continue
        for attempt in range(RETRIES):
            try:
                signal.alarm(BLOCK_TIMEOUT)
                for k, field in enumerate(FIELDS):
                    output[i:i + BLOCK, k] = h5_file[field][i:i + BLOCK].astype(np.float16)
                signal.alarm(0)
                break
            except Exception as e:
                signal.alarm(0)
                print(name, f"frames {i}: {type(e).__name__}, retry {attempt + 1} of {RETRIES}", flush=True)
                time.sleep(30)
                signal.alarm(BLOCK_TIMEOUT)   # opening the file again can hang too
                remote, h5_file = open_file(name)
                signal.alarm(0)
        else:
            raise RuntimeError(f"{name}: frames {i} failed {RETRIES} times")
        output.flush()
        with open(done_path, "a") as fd:
            fd.write(f"{i}\n")
        print(name, f"frames {i}-{min(i + BLOCK, n_frames) - 1} of {n_frames}", flush=True)
    h5_file.close(); remote.close()
    print(name, "saved", output.shape, flush=True)
