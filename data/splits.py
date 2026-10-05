"""Splits (§6).

Three levels, deliberately separated so that nothing can leak:

* ``model_split(held_out_metro)`` -- the four training metros, 85/15
  train/validation by route id hash (stable across reruns and across machines);
* ``held_out_metro_split`` -- the fifth metro, never seen during Stage-1
  training; rotates over all five for the leave-one-metro-out protocol;
* ``official_eval_split`` -- the 3,072-route challenge evaluation set, which the
  pipeline is allowed to touch exactly once, for the headline number.  Loading
  it writes a use-record to ``results/official_eval_uses.jsonl`` so that
  "touched only once" is auditable rather than aspirational.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from core.config import METROS, Paths
from core.io import fs_exists, fs_path, list_route_ids, read_instance, metro_slug
from core.logging_utils import get_logger
from core.types import Instance

log = get_logger(__name__)

VALIDATION_FRACTION = 0.15
HIGH_QUALITY = "High"


def _hash_unit(route_id: str, salt: str = "mtp") -> float:
    h = hashlib.sha256(f"{salt}:{route_id}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


@dataclass
class Split:
    name: str
    route_paths: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.route_paths)

    def instances(self, limit: int | None = None) -> list[Instance]:
        paths = self.route_paths if limit is None else self.route_paths[:limit]
        return [read_instance(p) for p in paths]

    def iter_instances(self, limit: int | None = None):
        for k, p in enumerate(self.route_paths):
            if limit is not None and k >= limit:
                return
            yield read_instance(p)

    def route_ids(self) -> list[str]:
        return [p.stem for p in self.route_paths]


def _paths_for(processed: Path, metros: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for m in metros:
        d = Path(processed) / metro_slug(m)
        if not d.exists():
            log.warning("no processed routes for metro %s (%s)", m, d)
            continue
        out.extend(sorted(d.glob("*.parquet")))
    return out


def _filter_quality(paths: Sequence[Path], high_only: bool) -> list[Path]:
    if not high_only:
        return list(paths)
    keep = []
    for p in paths:
        meta_path = p.with_name(p.stem + "_meta.json")
        if not fs_exists(meta_path):
            continue
        with open(fs_path(meta_path), "r") as fh:
            meta = json.load(fh)
        if meta.get("quality") == HIGH_QUALITY and meta.get("actual_sequence"):
            keep.append(p)
    return keep


def model_split(
    processed: str | Path,
    held_out_metro: str,
    high_quality_only: bool = True,
) -> tuple[Split, Split]:
    """Train/validation split over the four non-held-out metros."""
    if held_out_metro not in METROS:
        raise ValueError(f"{held_out_metro!r} is not one of {METROS}")
    metros = [m for m in METROS if m != held_out_metro]
    paths = _filter_quality(_paths_for(Path(processed), metros), high_quality_only)
    train, val = [], []
    for p in paths:
        (val if _hash_unit(p.stem) < VALIDATION_FRACTION else train).append(p)
    log.info(
        "model split (held out %s): %d train, %d val routes",
        held_out_metro, len(train), len(val),
    )
    return Split(f"train_ho_{metro_slug(held_out_metro)}", train), Split(
        f"val_ho_{metro_slug(held_out_metro)}", val
    )


def held_out_metro_split(
    processed: str | Path,
    held_out_metro: str,
    high_quality_only: bool = True,
) -> Split:
    paths = _filter_quality(
        _paths_for(Path(processed), [held_out_metro]), high_quality_only
    )
    log.info("held-out metro %s: %d routes", held_out_metro, len(paths))
    return Split(f"test_{metro_slug(held_out_metro)}", paths)


def leave_one_metro_out(processed: str | Path, high_quality_only: bool = True):
    """Yield ``(fold_metro, train, val, test)`` for all five folds."""
    for m in METROS:
        train, val = model_split(processed, m, high_quality_only)
        test = held_out_metro_split(processed, m, high_quality_only)
        yield m, train, val, test


def official_eval_split(
    processed_eval: str | Path,
    paths: Paths | None = None,
    reason: str = "unspecified",
    high_quality_only: bool = False,
) -> Split:
    """The official 3,072-route evaluation set -- headline result only.

    Every call is appended to ``results/official_eval_uses.jsonl`` with the
    reason given, so the thesis can state exactly how often it was used.
    """
    root = Path(processed_eval)
    all_paths = _filter_quality(
        [p for d in sorted(root.iterdir()) if d.is_dir() for p in sorted(d.glob("*.parquet"))],
        high_quality_only,
    )
    paths = paths or Paths()
    Path(paths.results).mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "n_routes": len(all_paths),
    }
    with (Path(paths.results) / "official_eval_uses.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    log.warning("OFFICIAL EVALUATION SET accessed (%s): %d routes", reason, len(all_paths))
    return Split("official_eval", all_paths)


def summarise(processed: str | Path) -> dict:
    out = {}
    for m in METROS:
        ids = list_route_ids(processed, m)
        out[m] = len(ids)
    return out
