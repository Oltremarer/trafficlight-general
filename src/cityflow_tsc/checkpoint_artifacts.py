"""Read retained checkpoint identities or identities in completed cleanup audits.

Archived metadata supports reporting only; it can never restore model weights.
"""
from __future__ import annotations

import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text())


class CheckpointCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._removed = None

    def _load_removed(self):
        removed = {}
        for manifest_path in sorted((self.root / "control").glob("checkpoint_retention_*/manifest.json")):
            audit_dir = manifest_path.parent
            summary_path, log_path = audit_dir / "summary.json", audit_dir / "deleted.jsonl"
            if not summary_path.is_file() or not log_path.is_file():
                continue
            summary, manifest = _read(summary_path), _read(manifest_path)
            if summary.get("status") != "complete":
                continue
            if Path(manifest["run_root"]).resolve() != self.root:
                raise ValueError(f"cleanup manifest belongs to a different run: {manifest_path}")
            deleted = {row["checkpoint"]: row for row in
                       (json.loads(line) for line in log_path.read_text().splitlines() if line.strip())}
            entries = manifest["delete"]
            if (len(deleted) != len(entries)
                    or len(entries) != summary["deleted_intermediate_checkpoints"]):
                raise ValueError(f"incomplete checkpoint cleanup journal: {audit_dir}")
            for entry in entries:
                path = Path(entry["checkpoint"]).resolve()
                if self.root not in path.parents:
                    raise ValueError("archived checkpoint is outside its run")
                record = deleted.get(str(path))
                sidecar = str(path.with_suffix(".protocol.json"))
                if record is None or record["sidecar"] != sidecar or entry["sidecar"] != sidecar:
                    raise ValueError(f"checkpoint deletion is not confirmed: {path}")
                protocol = entry["protocol"]
                if protocol["completed_episodes"] != entry["completed_episodes"]:
                    raise ValueError(f"archived episode identity disagrees: {path}")
                if path in removed and removed[path] != protocol:
                    raise ValueError(f"conflicting archived checkpoint identities: {path}")
                removed[path] = protocol
        self._removed = removed

    def protocol(self, checkpoint, *, require_weights=False):
        path = Path(checkpoint).resolve()
        if self.root not in path.parents:
            raise ValueError(f"checkpoint outside run: {path}")
        if require_weights and (not path.is_file() or path.stat().st_size == 0):
            raise FileNotFoundError(f"required checkpoint weights are missing: {path}")
        sidecar = path.with_suffix(".protocol.json")
        if sidecar.is_file():
            return _read(sidecar)
        # A live weight file without its sidecar is an incomplete artifact.
        if path.exists():
            raise ValueError(f"checkpoint has no protocol sidecar: {path}")
        if self._removed is None:
            self._load_removed()
        if path not in self._removed:
            raise FileNotFoundError(f"checkpoint metadata missing without a completed cleanup record: {path}")
        return self._removed[path]

    def validate_training(self, output, summary, config):
        output = Path(output)
        completed = summary["completed_episodes"]
        final_path = Path(summary["checkpoint"])
        final = self.protocol(final_path, require_weights=True)
        if final["completed_episodes"] != completed:
            raise ValueError("final checkpoint does not match completed training")
        interval = config.get("checkpoint_every", 1)
        if interval <= 0:
            raise ValueError("checkpoint cadence must be positive")
        baseline = config["profile"]["baseline_id"]
        expected = {row["episode"] + 1 for row in summary["training"]
                    if (row["episode"] + 1) % interval == 0}
        expected.add(completed)
        paths = {output / "checkpoints" / f"{baseline}.episode_{n - 1:04d}.pt": n
                 for n in expected}
        if final_path.resolve() != (output / "checkpoints" / f"{baseline}.episode_{completed - 1:04d}.pt").resolve():
            raise ValueError("final checkpoint points outside this training attempt")
        for path, episode in paths.items():
            protocol = self.protocol(path)
            if protocol["completed_episodes"] != episode:
                raise ValueError(f"checkpoint episode mismatch: {path}")
        retained = list((output / "checkpoints").glob("*.pt"))
        for path in retained:
            if path not in paths:
                raise ValueError(f"unexpected checkpoint: {path}")
            self.protocol(path, require_weights=True)
        # Missing weights with an orphan sidecar must also have a cleanup audit.
        for path in paths:
            if not path.exists():
                if self._removed is None:
                    self._load_removed()
                if path.resolve() not in self._removed:
                    raise FileNotFoundError(f"checkpoint weights missing without cleanup: {path}")
        return len(retained)
