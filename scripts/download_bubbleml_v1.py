# Download the first N HDF5 files (default 1) from the 10.5 GB BubbleML subcooled pool boiling archive.
# Usage: python scripts/download_bubbleml_v1.py [N]
# The archive is one .tar.gz, so the script streams it and stops after N files instead of saving all of it.
# Files already on disk are skipped, but the stream still reads past them.
import os, sys, tarfile, urllib.request

URL = "https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d.tar.gz"
OUT_DIR = "/pub/ynagaraj/bubbleml"
n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 1

with urllib.request.urlopen(URL) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
    n_saved = 0
    for member in tar:
        print(member.name, f"{member.size/1e6:.0f} MB", flush=True)
        if member.isfile() and member.name.endswith((".hdf5", ".h5")):
            if not os.path.exists(f"{OUT_DIR}/{member.name}"):
                tar.extract(member, OUT_DIR)
            n_saved += 1
            if n_saved == n_files:
                break
