from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 100_000_000_000:
        raise OSError("Less than 100 GB disk reserve remains; collection stopped")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def shard_committed(path, branch_ids):
    meta = Path(str(path) + ".json")
    if not meta.is_file() or not Path(path).is_file():
        return False
    info = json.loads(meta.read_text())
    if info["branch_ids"] != branch_ids:
        raise ValueError(f"Existing shard belongs to a different plan: {path}")
    if sha256(path) != info["sha256"]:
        raise IOError(f"Committed shard checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        if data["branch_ids"].tolist() != branch_ids or info["branch_count"] != len(branch_ids):
            raise ValueError(f"Committed shard payload IDs/count differ from plan: {path}")
    return True
