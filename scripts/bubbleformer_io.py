# Shared helpers for running the published Bubbleformer model on BubbleML 2.0 files.
# Used by scripts/rollout_timing_v2.py and scripts/accuracy_check_v2.py.
import os
import numpy as np, torch, h5py, fsspec

HF = "https://huggingface.co/datasets/hpcforge/BubbleML_2/resolve/main/PoolBoiling-Subcooled-FC72-2D"
FIELDS = ["dfun", "temperature", "velx", "vely"]   # Bubbleformer's fields, in its order


def load_model(path, dev):
    """Build the model from a Bubbleformer checkpoint, as in Bubbleformer's scripts/inference.py."""
    from bubbleformer.models import get_model
    import importlib, pickle, types, torch._weights_only_unpickler as wu
    # Load safely: the file may create only allow-listed classes, so it cannot run other code while it is read.
    # torch's weights_only=True loader cannot be used, because it cannot fill the defaultdict in this checkpoint.
    # The allow list is torch's own safe list plus the classes this checkpoint needs for its omegaconf settings
    # (listed by torch.serialization.get_unsafe_globals_in_checkpoint).
    extra = ["omegaconf.listconfig.ListConfig", "omegaconf.dictconfig.DictConfig", "omegaconf.base.ContainerMetadata",
             "omegaconf.base.Metadata", "omegaconf.nodes.AnyNode", "collections.defaultdict", "typing.Any",
             "builtins.list", "builtins.dict", "builtins.int"]
    allowed = dict(wu._get_allowed_globals())
    for key in extra:
        mod, attr = key.rsplit(".", 1)
        allowed[key] = getattr(importlib.import_module(mod), attr)

    import _compat_pickle

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            # Old pickle protocols use Python 2 names (for example __builtin__.dict). Translate them first,
            # as the standard unpickler does.
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
            key = f"{module}.{name}"
            if key not in allowed:
                raise pickle.UnpicklingError(f"checkpoint uses a class that is not allowed: {key}")
            return allowed[key]

    restricted = types.SimpleNamespace(Unpickler=Restricted, load=pickle.load, __name__="restricted_pickle")
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=restricted)
    hp = ck["hyper_parameters"]
    params = dict(hp["model_cfg"]["params"])
    params["input_fields"] = len(hp["data_cfg"]["input_fields"])
    params["output_fields"] = len(hp["data_cfg"]["output_fields"])
    params["time_window"] = hp["data_cfg"]["time_window"]
    model = get_model(hp["model_cfg"]["name"], **params)
    model.load_state_dict({k[len("model."):]: v for k, v in ck["state_dict"].items()})  # drop the Lightning prefix
    diff, div = hp["normalization_constants"]
    return model.to(dev).eval(), diff, div


def fluid_params(j):
    """The 9 conditioning values, in the order of Bubbleformer's BubbleForecast dataset."""
    h = j["heater"]
    return torch.tensor([[j["inv_reynolds"], j["cpgas"], j["mugas"], j["rhogas"], j["thcogas"],
                          j["stefan"], j["prandtl"], h["nucWaitTime"], h["wallTemp"]]], dtype=torch.float32)


def read_frames(data, name, start, stop):
    """Frames start ... stop-1 of the 4 fields at 512 x 512, shape (frames, 4, 512, 512).

    Reads DATA/<name>.hdf5 if it exists, else reads only these frames from Hugging Face.
    """
    local = f"{data}/{name}.hdf5"
    fo = None if os.path.exists(local) else fsspec.open(f"{HF}/{name}.hdf5", "rb", block_size=2**24).open()
    with h5py.File(local if fo is None else fo) as f:
        x = np.stack([f[k][start:stop] for k in FIELDS], axis=1)
    if fo is not None:
        fo.close()
    return torch.from_numpy(x.astype(np.float32))
