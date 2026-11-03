# Stream the 10.5 GB BubbleML archive and extract only the first HDF5 file(s), then stop.
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
