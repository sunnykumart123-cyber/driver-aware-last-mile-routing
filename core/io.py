"""Reading/writing processed instances and the experiment log.

Processed layout (per §6 of the build spec)::

    data/processed/<metro_slug>/<route_id>.parquet        stops + features + t_ij
    data/processed/<metro_slug>/<route_id>_meta.json      quality label, sigma*

The travel-time matrix lives in the same parquet as columns ``tt_0 ... tt_{n-1}``
so that one instance is one file plus its metadata sidecar.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from core.config import METRO_SLUGS
from core.types import Instance, Stop

STOP_COLS = [
    "stop_id",
    "zone",
    "lat",
    "lng",
    "pkg_count",
    "pkg_volume",
    "tw_start",
    "tw_end",
    "service_time",
    "is_depot",
]


def metro_slug(metro: str) -> str:
    return METRO_SLUGS.get(metro, metro.strip().lower().replace(" ", "_"))


def fs_exists(path: str | Path) -> bool:
    """``Path.exists()`` that survives Windows long paths.

    ``Path.exists()`` goes through the non-extended Win32 API and returns
    *False* for a path longer than ``MAX_PATH`` even when the file is there.
    That failure mode is silent and nasty: it made every metadata sidecar look
    absent, so instances loaded with no ``actual_sequence``, no quality label
    and the metro inferred from the directory name.
    """
    return os.path.exists(fs_path(path))


def fs_path(path: str | Path) -> str:
    """A path safe to open on this platform.

    Real LMRRC route ids are UUIDs (``RouteID_00143bdd-0a6b-...``), so a
    processed filename is ~50 characters before the directory prefix.  On
    Windows that blows past the 260-character ``MAX_PATH`` limit inside any
    moderately deep working directory, and the failure surfaces as a baffling
    ``FileNotFoundError`` *on write*.  Prefixing the absolute path with
    ``\\\\?\\`` opts into the extended-length API; ``open()`` honours it, so the
    parquet writers get a file object rather than a path string.
    """
    p = Path(path).resolve()
    s = str(p)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


# --------------------------------------------------------------------- write
def write_instance(inst: Instance, processed_root: str | Path) -> Path:
    out_dir = Path(processed_root) / metro_slug(inst.metro)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "stop_id": s.stop_id,
            "zone": s.zone,
            "lat": s.lat,
            "lng": s.lng,
            "pkg_count": s.pkg_count,
            "pkg_volume": s.pkg_volume,
            "tw_start": s.tw_start,
            "tw_end": s.tw_end,
            "service_time": s.service_time,
            "is_depot": s.is_depot,
        }
        for s in inst.stops
    ]
    df = pd.DataFrame(rows, columns=STOP_COLS)
    tt = pd.DataFrame(
        inst.travel_time, columns=[f"tt_{j}" for j in range(inst.n)]
    )
    df = pd.concat([df, tt], axis=1)

    for extra_name, extra in inst.meta.get("features", {}).items():
        df[f"feat_{extra_name}"] = np.asarray(extra)

    path = out_dir / f"{inst.route_id}.parquet"
    with open(fs_path(path), "wb") as fh:
        df.to_parquet(fh, index=False)

    meta = {
        "route_id": inst.route_id,
        "metro": inst.metro,
        "station": inst.station,
        "quality": inst.quality,
        "actual_sequence": inst.actual_sequence,
        "executor_capacity": (
            None if not np.isfinite(inst.executor_capacity) else inst.executor_capacity
        ),
        "n_stops": inst.n,
        "meta": {k: v for k, v in inst.meta.items() if k != "features"},
    }
    with open(fs_path(out_dir / f"{inst.route_id}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return path


# ---------------------------------------------------------------------- read
def read_instance(parquet_path: str | Path) -> Instance:
    parquet_path = Path(parquet_path)
    with open(fs_path(parquet_path), "rb") as fh:
        df = pd.read_parquet(fh)
    meta_path = parquet_path.with_name(parquet_path.stem + "_meta.json")
    meta = {}
    if fs_exists(meta_path):
        with open(fs_path(meta_path), "r") as fh:
            meta = json.load(fh)

    n = len(df)
    tt = df[[f"tt_{j}" for j in range(n)]].to_numpy(dtype=float)
    stops = [
        Stop(
            stop_id=str(r.stop_id),
            zone=str(r.zone),
            lat=float(r.lat),
            lng=float(r.lng),
            pkg_count=int(r.pkg_count),
            pkg_volume=float(r.pkg_volume),
            tw_start=float(r.tw_start),
            tw_end=float(r.tw_end),
            service_time=float(r.service_time),
            is_depot=bool(r.is_depot),
        )
        for r in df.itertuples(index=False)
    ]
    features = {
        c[len("feat_"):]: df[c].to_numpy()
        for c in df.columns
        if c.startswith("feat_")
    }
    inner = dict(meta.get("meta", {}))
    if features:
        inner["features"] = features
    cap = meta.get("executor_capacity")
    return Instance(
        route_id=meta.get("route_id", parquet_path.stem),
        metro=meta.get("metro", parquet_path.parent.name),
        station=meta.get("station", ""),
        stops=stops,
        travel_time=tt,
        actual_sequence=meta.get("actual_sequence"),
        quality=meta.get("quality"),
        executor_capacity=float("inf") if cap is None else float(cap),
        meta=inner,
    )


def iter_instances(
    processed_root: str | Path,
    metros: Iterable[str] | None = None,
    route_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> Iterator[Instance]:
    root = Path(processed_root)
    slugs = None if metros is None else {metro_slug(m) for m in metros}
    wanted = None if route_ids is None else set(route_ids)
    count = 0
    for metro_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if slugs is not None and metro_dir.name not in slugs:
            continue
        for pq in sorted(metro_dir.glob("*.parquet")):
            if wanted is not None and pq.stem not in wanted:
                continue
            yield read_instance(pq)
            count += 1
            if limit is not None and count >= limit:
                return


def list_route_ids(processed_root: str | Path, metro: str) -> list[str]:
    d = Path(processed_root) / metro_slug(metro)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


# ------------------------------------------------------------ experiment log
def append_results(rows: list[dict], log_path: str | Path) -> Path:
    """Append experiment rows to the parquet log, creating it if absent."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if log_path.exists():
        old = pd.read_parquet(log_path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(log_path, index=False)
    return log_path


def load_results(log_path: str | Path) -> pd.DataFrame:
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(log_path)
