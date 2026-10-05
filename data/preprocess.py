"""Parse the raw LMRRC JSON into one normalised instance per route (§6).

    python -m data.preprocess --raw data/raw --out data/processed --split training
    python -m data.preprocess --raw data/raw --out data/processed --split evaluation

Steps, in the order §6 prescribes:

1. parse the per-route JSON into a stop-level table;
2. keep the route-quality label (``High`` routes are the sigma* ground truth
   under the official LMRRC protocol -- filtering happens in
   :mod:`data.splits`, not here, so the other routes stay available for the
   uncertainty model);
3. build the per-route zone structure from the sort-zone ids;
4. engineer features; frequency tables that pool across routes are fit in
   :mod:`models.uncertainty.quantile_model` and
   :mod:`models.behavior.gbm_transition` **on the training split only**;
5. merge the supplied travel-time matrix directly as ``t_ij`` -- no geocoding,
   no external map lookups (the GPS coordinates are obfuscated).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np

from core.io import write_instance
from core.logging_utils import get_logger
from core.types import Instance, Stop

log = get_logger(__name__)

# station_code[1:3] -> metro.  The 2021 release covers exactly these five.
STATION_PREFIX_TO_METRO = {
    "LA": "Los Angeles",
    "SE": "Seattle",
    "CH": "Chicago",
    "BO": "Boston",
    "AU": "Austin",
}

FILE_PATTERNS = {
    "training": {
        "route": ["route_data.json"],
        "package": ["package_data.json"],
        "travel": ["travel_times.json"],
        "actual": ["actual_sequences.json"],
        "invalid": ["invalid_sequence_scores.json"],
    },
    "evaluation": {
        "route": ["new_route_data.json", "eval_route_data.json"],
        "package": ["new_package_data.json", "eval_package_data.json"],
        "travel": ["new_travel_times.json", "eval_travel_times.json"],
        "actual": ["new_actual_sequences.json", "eval_actual_sequences.json"],
        "invalid": ["new_invalid_sequence_scores.json"],
    },
}


def metro_of(station_code: str) -> str:
    return STATION_PREFIX_TO_METRO.get(str(station_code)[1:3].upper(), "Unknown")


def find_file(raw_root: Path, split: str, kind: str) -> Path | None:
    for name in FILE_PATTERNS[split][kind]:
        hits = sorted(raw_root.rglob(name))
        if hits:
            return hits[0]
    return None


def _parse_dt(date_str: str, time_str: str) -> datetime:
    d = datetime.strptime(str(date_str), "%Y-%m-%d")
    parts = str(time_str).split(":")
    h, m, s = (int(parts[0]), int(parts[1]), int(float(parts[2])) if len(parts) > 2 else 0)
    return d + timedelta(hours=h, minutes=m, seconds=s)


def _parse_window(value, departure: datetime) -> float | None:
    """A package time-window bound, in minutes from departure."""
    if value in (None, "", "nan") or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        dt = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    return (dt - departure).total_seconds() / 60.0


@dataclass
class RawBundle:
    """Lazy, memory-bounded handles on the five raw files of one split.

    Each field is a :class:`~data.jsonstream.JsonObjectIndex` (or an
    :class:`~data.jsonstream.InMemoryIndex` for small files and tests), so a
    route's payload is parsed on demand by seek.  ``travel_times.json`` is
    1.8 GB on the training split; reading it whole would need tens of GB of RAM,
    which is why nothing here ever calls ``json.load`` on a raw file.
    """

    route: object
    package: object
    travel: object | None
    actual: object
    invalid: object
    paths: dict[str, Path] = field(default_factory=dict)
    travel_path: Path | None = None

    # Files below this size are simply parsed into memory: the index machinery
    # costs more than it saves for the sequence and score files.
    SMALL_FILE_BYTES = 64 * 1024 * 1024

    @classmethod
    def load(cls, raw_root: Path, split: str, cache_dir: Path | None = None,
             stream_travel: bool = True) -> "RawBundle":
        """Open the five files with the cheapest access pattern for each.

        ``travel_times.json`` is the big one and preprocessing visits every
        route exactly once, so it is **streamed** rather than indexed: on the
        1.73 GB training file, streaming it with ijson's C backend takes about
        1.2 minutes, against ~3 minutes to build an offset index and then seek
        back through it.  The smaller files keep the index, because the join is
        driven by the travel stream and their key order need not agree with it.
        """
        from data.jsonstream import InMemoryIndex, JsonObjectIndex

        paths: dict[str, Path] = {}

        def _load(kind: str, required: bool):
            p = find_file(raw_root, split, kind)
            if p is None:
                if required:
                    raise FileNotFoundError(
                        f"could not find the {kind} file for the {split} split "
                        f"under {raw_root}; run `python -m data.download` first"
                    )
                log.warning("%s file absent for the %s split", kind, split)
                return InMemoryIndex({})
            paths[kind] = p
            size = p.stat().st_size
            if size <= cls.SMALL_FILE_BYTES:
                log.info("reading %s (%.1f MB)", p.name, size / 1048576)
                return InMemoryIndex(json.loads(p.read_text()))
            log.info("indexing %s (%.1f MB)", p.name, size / 1048576)
            return JsonObjectIndex.build_cached(p, cache_dir)

        route = _load("route", True)
        package = _load("package", True)
        travel_file = find_file(raw_root, split, "travel")
        if travel_file is None:
            raise FileNotFoundError(
                f"could not find the travel file for the {split} split under "
                f"{raw_root}; run `python -m data.download` first"
            )
        paths["travel"] = travel_file
        big = travel_file.stat().st_size > cls.SMALL_FILE_BYTES
        travel = None if (stream_travel and big) else _load("travel", True)
        if travel is None:
            log.info("streaming %s (%.1f MB)", travel_file.name,
                     travel_file.stat().st_size / 1048576)

        return cls(
            route=route,
            package=package,
            travel=travel,
            actual=_load("actual", False),
            invalid=_load("invalid", False),
            paths=paths,
            travel_path=travel_file,
        )

    def route_ids(self) -> list[str]:
        return sorted(self.route.keys())

    def iter_travel(self):
        """``(route_id, travel_matrix)`` for every route, streamed if possible."""
        if self.travel is None:
            from data.jsonstream import iter_top_level_objects

            yield from iter_top_level_objects(self.travel_path)
        else:
            yield from self.travel.items()

    def close(self) -> None:
        for handle in (self.route, self.package, self.travel, self.actual,
                       self.invalid):
            if handle is None:
                continue
            close = getattr(handle, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> "RawBundle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_instance(
    route_id: str,
    route: dict,
    packages: dict,
    travel: dict,
    actual: dict | None,
    quality_override: float | None = None,
) -> Instance | None:
    """Assemble one :class:`Instance`; returns ``None`` if the route is unusable."""
    stops_raw = route.get("stops", {})
    if not stops_raw:
        return None
    departure = _parse_dt(route["date_YYYY_MM_DD"], route["departure_time_utc"])

    depot_ids = [k for k, v in stops_raw.items() if str(v.get("type")) == "Station"]
    if len(depot_ids) != 1:
        log.debug("%s: %d stations, skipping", route_id, len(depot_ids))
        return None
    depot_id = depot_ids[0]
    other_ids = sorted(k for k in stops_raw if k != depot_id)
    ordered_ids = [depot_id] + other_ids
    index = {sid: k for k, sid in enumerate(ordered_ids)}

    stops: list[Stop] = []
    for sid in ordered_ids:
        raw = stops_raw[sid]
        is_depot = sid == depot_id
        pkgs = packages.get(sid, {}) if not is_depot else {}
        svc, vol, count = 0.0, 0.0, 0
        tw_lo, tw_hi = -np.inf, np.inf
        for pkg in pkgs.values():
            count += 1
            svc += float(pkg.get("planned_service_time_seconds") or 0.0) / 60.0
            dim = pkg.get("dimensions") or {}
            vol += (
                float(dim.get("depth_cm") or 0.0)
                * float(dim.get("height_cm") or 0.0)
                * float(dim.get("width_cm") or 0.0)
            ) / 1000.0                                    # cm^3 -> litres
            win = pkg.get("time_window") or {}
            lo = _parse_window(win.get("start_time_utc"), departure)
            hi = _parse_window(win.get("end_time_utc"), departure)
            # A stop inherits the tightest window among its packages.
            if lo is not None:
                tw_lo = max(tw_lo, lo)
            if hi is not None:
                tw_hi = min(tw_hi, hi)
        zone = raw.get("zone_id")
        if zone in (None, "", "nan") or (isinstance(zone, float) and np.isnan(zone)):
            # Unzoned stops (a known quirk of the release) form their own block.
            zone = "UNZONED" if not is_depot else "DEPOT"
        stops.append(
            Stop(
                stop_id=sid,
                zone=str(zone),
                lat=float(raw["lat"]),
                lng=float(raw["lng"]),
                pkg_count=count,
                pkg_volume=vol,
                tw_start=float(tw_lo),
                tw_end=float(tw_hi),
                service_time=float(svc),
                is_depot=is_depot,
            )
        )

    n = len(ordered_ids)
    tt = np.zeros((n, n), dtype=float)
    for a, row in travel.items():
        ia = index.get(a)
        if ia is None:
            continue
        for b, secs in row.items():
            ib = index.get(b)
            if ib is None:
                continue
            tt[ia, ib] = float(secs) / 60.0
    np.fill_diagonal(tt, 0.0)

    seq = None
    if actual:
        pos = actual.get("actual", actual)
        if isinstance(pos, dict) and len(pos) == n:
            try:
                seq = [index[sid] for sid, _ in sorted(pos.items(), key=lambda kv: kv[1])]
            except KeyError:
                seq = None
            if seq is not None and (seq[0] != 0 or sorted(seq) != list(range(n))):
                seq = None

    quality = route.get("route_score")
    if quality is None and quality_override is not None:
        quality = "High" if quality_override <= 0 else "Low"

    cap = route.get("executor_capacity_cm3")
    return Instance(
        route_id=route_id,
        metro=metro_of(route.get("station_code", "")),
        station=str(route.get("station_code", "")),
        stops=stops,
        travel_time=tt,
        actual_sequence=seq,
        quality=str(quality) if quality is not None else None,
        executor_capacity=float(cap) / 1000.0 if cap else float("inf"),
        meta={
            "date": route.get("date_YYYY_MM_DD"),
            "departure_time": route.get("departure_time_utc"),
            "departure_minute_of_day": departure.hour * 60.0 + departure.minute,
            "weekday": departure.weekday(),
            "n_zones": len({s.zone for s in stops if not s.is_depot}),
        },
    )


def iter_parsed(bundle: RawBundle, limit: int | None = None) -> Iterator[Instance]:
    """Build instances one route at a time; only one route is ever in memory.

    The travel-time file drives the loop (it is the one that is streamed) and
    the other four are joined by route id.  Routes present in ``route_data`` but
    absent from ``travel_times`` are reported at the end rather than ignored
    silently -- a mismatch there would mean the download is incomplete.
    """
    n_bad = 0
    n_unknown = 0
    seen: set[str] = set()
    yielded = 0
    for route_id, travel in bundle.iter_travel():
        if limit is not None and yielded >= limit:
            break
        route = bundle.route.get(route_id)
        if route is None:
            n_unknown += 1
            continue
        seen.add(route_id)
        inst = build_instance(
            route_id,
            route,
            bundle.package.get(route_id, {}) or {},
            travel or {},
            bundle.actual.get(route_id),
            bundle.invalid.get(route_id),
        )
        if inst is None:
            n_bad += 1
            continue
        yielded += 1
        yield inst

    if n_bad:
        log.warning("skipped %d unusable routes", n_bad)
    if n_unknown:
        log.warning("%d routes in the travel file are absent from route_data",
                    n_unknown)
    if limit is None:
        missing = set(bundle.route.keys()) - seen
        if missing:
            log.error(
                "%d routes have route_data but no travel matrix (e.g. %s); the "
                "download may be incomplete",
                len(missing), sorted(missing)[:3],
            )


def run(raw_root: Path, out_root: Path, split: str, limit: int | None = None) -> dict:
    counts: dict[str, int] = {}
    n_seq = 0
    sizes: list[int] = []
    with RawBundle.load(raw_root, split) as bundle:
        log.info("%d routes in the %s split", len(bundle.route), split)
        for k, inst in enumerate(iter_parsed(bundle, limit)):
            write_instance(inst, out_root)
            counts[inst.metro] = counts.get(inst.metro, 0) + 1
            n_seq += int(inst.actual_sequence is not None)
            sizes.append(inst.n)
            if (k + 1) % 500 == 0:
                log.info("written %d routes", k + 1)
    summary = {
        "split": split,
        "routes": int(sum(counts.values())),
        "with_actual_sequence": n_seq,
        "per_metro": counts,
        "stops_mean": float(np.mean(sizes)) if sizes else 0.0,
        "stops_min": int(np.min(sizes)) if sizes else 0,
        "stops_max": int(np.max(sizes)) if sizes else 0,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"preprocess_summary_{split}.json").write_text(
        json.dumps(summary, indent=2)
    )
    log.info("%s", json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--split", choices=["training", "evaluation"], default="training")
    ap.add_argument("--limit", type=int, default=None,
                    help="parse only the first N routes (smoke runs)")
    args = ap.parse_args(argv)
    run(args.raw, args.out, args.split, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
