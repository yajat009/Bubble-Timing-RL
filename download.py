# Download the first N HDF5 files (default 1) from the 10.5 GB BubbleML subcooled pool boiling archive.
# Usage: python download.py [N]
# The archive is one .tar.gz, so the script streams it and stops after N files instead of saving all of it.
# Files already on disk are skipped, but the stream still reads past them.
import os, sys, tarfile, urllib.request

URL = "https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d.tar.gz"
OUT = "/pub/ynagaraj/bubbleml"
n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 1

with urllib.request.urlopen(URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
    got = 0
    for m in tar:
        print(m.name, f"{m.size/1e6:.0f} MB", flush=True)
        if m.isfile() and m.name.endswith((".hdf5", ".h5")):
            if not os.path.exists(f"{OUT}/{m.name}"):
                tar.extract(m, OUT)
            got += 1
            if got == n_files:
                break
