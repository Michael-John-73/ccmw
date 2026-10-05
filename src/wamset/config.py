"""Configuration, artifact paths, split-access guards and stage logging.

Pure standard library: importable without torch so CPU tests and planning stay light.
"""
from contextlib import contextmanager
from pathlib import Path
import gzip
import hashlib
import json
import math
import os
import time

STUDY = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = STUDY / "configs" / "candidate_set.json"
SPLITS = ("train", "dev_tune", "dev_check", "cal", "test")


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _atomic(path, opener, mode, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with opener(tmp, mode, encoding="utf-8", newline="\n") as stream:
        writer(stream)
    os.replace(tmp, path)


def write_json(path, value):
    def writer(stream):
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    _atomic(path, open, "w", writer)


def write_jsonl(path, rows, compress=False):
    def writer(stream):
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    _atomic(path, gzip.open if compress else open, "wt" if compress else "w", writer)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_config(path=None, output=None):
    path = Path(path or DEFAULT_CONFIG).resolve()
    cfg = read_json(path)
    cfg["_path"] = str(path)
    cfg["_study"] = str(path.parent.parent)
    cfg["_hash"] = sha_file(path)
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else (Path(cfg["_study"]) / p).resolve())
    if output is not None:
        cfg["paths"]["output"] = str(Path(output).resolve())
    validate(cfg)
    return cfg


def validate(cfg):
    c = cfg["counts"]
    if c["dev_tune"] + c["dev_check"] != c["dev"]:
        raise ValueError("dev must equal dev_tune + dev_check")
    if any(int(c[k]) <= 0 for k in ("train", "dev_tune", "dev_check", "cal", "test")):
        raise ValueError("All split counts must be positive")
    if cfg["extract"]["candidates"] != 3:
        raise ValueError("The protocol fixes top-3 candidates")
    if min(cfg["registry"]["train_sizes"]) < cfg["extract"]["candidates"]:
        raise ValueError("Registries must hold at least the candidate count")


def root(cfg):
    return Path(cfg["paths"]["output"])


def hypothesis_alpha(cfg):
    s = cfg["statistics"]
    return s["family_alpha"] / s["hypotheses"]


def neg_inf_to_none(values):
    return [None if (v is None or not math.isfinite(v)) else float(v) for v in values]


def none_to_neg_inf(values):
    return [-math.inf if v is None else float(v) for v in values]


# ---------------------------------------------------------------- artifact layout
def manifest_dir(cfg):
    return root(cfg) / "manifests"


def cache_path(cfg, split, scene_id, condition, boundary):
    return root(cfg) / "cache" / split / f"{scene_id}_{condition}_b{boundary}.npz"


def tuning_dir(cfg):
    return root(cfg) / "tuning_candidate_set"


def progress(iterable=None, **kwargs):
    """Terminal progress bar for long stages (tqdm). WAMSET_PROGRESS=0 disables it (unit tests)."""
    import sys
    from tqdm import tqdm
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("file", sys.stderr)
    kwargs["disable"] = os.environ.get("WAMSET_PROGRESS", "1") == "0"
    return tqdm(iterable, **kwargs)


def job_id(method, trial, seed):
    return f"{method}_{trial}_s{seed}"


def job_dir(cfg, method, trial, seed):
    return tuning_dir(cfg) / "jobs" / job_id(method, trial, seed)


# ---------------------------------------------------------------- split access guard
GATES = {
    # split -> (file that must exist under root, explanation)
    "train": (None, ""),
    "dev_tune": (None, ""),
    "dev_check": ("tuning_candidate_set/selection_candidate_set.json", "W06d selection must be locked before dev_check"),
    "cal": ("protocol_lock.json", "W07 protocol lock must exist before calibration data are read"),
    "test": ("calibration_candidate_set.json", "W09 calibration must exist before test data are read"),
}


class SplitAccessError(RuntimeError):
    pass


def require_split(cfg, split, allowed):
    """Raise unless `split` is permitted for the calling command and its gate file exists."""
    if split not in SPLITS:
        raise ValueError(split)
    if split not in allowed:
        raise SplitAccessError(f"Command may not read split '{split}' (allowed: {sorted(allowed)})")
    gate, why = GATES[split]
    if gate and not (root(cfg) / gate).is_file():
        raise SplitAccessError(why)


# ---------------------------------------------------------------- stage log
@contextmanager
def stage(cfg, name):
    folder = root(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "config_lock.json"
    current = {"protocol": cfg["protocol"], "config_sha256": cfg["_hash"]}
    if lock.exists():
        if read_json(lock) != current:
            raise RuntimeError("Output directory belongs to a different configuration; use a new output")
    else:
        write_json(lock, current)
    record = {"stage": name, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "config_sha256": cfg["_hash"]}
    started = time.monotonic()
    try:
        yield
    except Exception as error:
        record.update(status="failed", error=repr(error))
        raise
    else:
        record["status"] = "complete"
    finally:
        record["seconds"] = round(time.monotonic() - started, 3)
        with open(folder / "stage_log.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
