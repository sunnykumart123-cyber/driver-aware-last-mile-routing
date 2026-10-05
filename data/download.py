"""Fetch the 2021 Amazon Last Mile Routing Research Challenge data.

Source: AWS Registry of Open Data, bucket ``amazon-last-mile-challenges``
(https://registry.opendata.aws/amazon-last-mile-challenges/).  The bucket is
public, so no credentials are needed.

    python -m data.download --plan                  # sizes only, downloads nothing
    python -m data.download --dest data/raw         # fetch both splits
    python -m data.download --dest data/raw --split training
    python -m data.download --verify-only           # check what is on disk

Two transports.  HTTPS (the default) uses only the standard library and fetches
the ten files this project actually reads; it resumes partial downloads with a
Range request.  ``--transport aws`` shells out to ``aws s3 sync`` instead, which
is faster on a good connection but needs the AWS CLI installed.

Total download: about 3.2 GB (training 2.2 GB, evaluation 1.0 GB).  All of it is
open and publicly available, which is the hard data constraint in §1 of the
build spec; nothing in this repository generates substitute data for it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from core.logging_utils import get_logger

log = get_logger(__name__)

BUCKET_HTTPS = "https://amazon-last-mile-challenges.s3.amazonaws.com"
BUCKET_S3 = "s3://amazon-last-mile-challenges/almrrc2021/"
PREFIXES = {
    "training": "almrrc2021/almrrc2021-data-training/",
    "evaluation": "almrrc2021/almrrc2021-data-evaluation/",
}

# The files the pipeline reads, with the sizes published by the bucket listing
# on 2026-10-01.  Sizes are advisory: they are used for the download plan and to
# flag a truncated file, never to reject an otherwise-valid download.
EXPECTED: dict[str, list[tuple[str, int]]] = {
    "training": [
        ("model_build_inputs/route_data.json", 78_972_162),
        ("model_build_inputs/package_data.json", 375_437_806),
        ("model_build_inputs/travel_times.json", 1_817_146_363),
        ("model_build_inputs/actual_sequences.json", 9_665_078),
        ("model_build_inputs/invalid_sequence_scores.json", 414_742),
    ],
    "evaluation": [
        ("model_apply_inputs/eval_route_data.json", 37_777_768),
        ("model_apply_inputs/eval_package_data.json", 166_201_035),
        ("model_apply_inputs/eval_travel_times.json", 843_104_906),
        ("model_score_inputs/eval_actual_sequences.json", 4_625_218),
        ("model_score_inputs/eval_invalid_sequence_scores.json", 207_138),
    ],
}

CHUNK = 1 << 20


def have_aws_cli() -> bool:
    return shutil.which("aws") is not None


# ---------------------------------------------------------------------- plan
def plan(splits: list[str]) -> dict:
    rows = []
    total = 0
    for split in splits:
        for rel, size in EXPECTED[split]:
            rows.append({"split": split, "file": rel, "bytes": size,
                         "mb": round(size / 1048576, 1)})
            total += size
    return {"files": rows, "total_bytes": total,
            "total_gb": round(total / 1073741824, 2)}


def remote_size(url: str, timeout: float = 30.0) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError) as exc:
        log.warning("HEAD %s failed: %s", url, exc)
        return None


# ------------------------------------------------------------------ transport
def download_https(url: str, dest: Path, expected_size: int | None = None,
                   timeout: float = 60.0, retries: int = 4) -> bool:
    """Fetch ``url`` to ``dest``, resuming a partial file if one is present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    total = expected_size if expected_size is not None else remote_size(url)

    if dest.exists() and total is not None and dest.stat().st_size == total:
        log.info("%s already complete (%.1f MB)", dest.name, total / 1048576)
        return True

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have == total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, \
                    part.open("ab" if have else "wb") as fh:
                t0 = time.time()
                done = have
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if total and done % (64 * CHUNK) < CHUNK:
                        pct = 100.0 * done / total
                        rate = (done - have) / max(time.time() - t0, 1e-6) / 1048576
                        log.info("%s %5.1f%% (%.1f MB/s)", dest.name, pct, rate)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            log.warning("%s attempt %d/%d failed: %s", dest.name, attempt,
                        retries, exc)
            if attempt == retries:
                return False
            time.sleep(min(2 ** attempt, 30))

    size = part.stat().st_size if part.exists() else 0
    if total is not None and size != total:
        log.error("%s is %d bytes, expected %d", dest.name, size, total)
        return False
    part.replace(dest)
    log.info("wrote %s (%.1f MB)", dest, size / 1048576)
    return True


def sync_aws(split: str, dest: Path, dry_run: bool = False) -> int:
    src = BUCKET_S3 + PREFIXES[split].split("almrrc2021/", 1)[1]
    out = dest / split
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", src, str(out), "--no-sign-request"]
    if dry_run:
        cmd.append("--dryrun")
    log.info("%s", " ".join(cmd))
    return subprocess.call(cmd)


def fetch_split(split: str, dest: Path) -> dict:
    ok, failed = [], []
    for rel, size in EXPECTED[split]:
        url = f"{BUCKET_HTTPS}/{PREFIXES[split]}{rel}"
        target = dest / split / rel
        (ok if download_https(url, target, size) else failed).append(rel)
    return {"split": split, "downloaded": ok, "failed": failed}


# -------------------------------------------------------------------- verify
def verify(dest: Path, checksums: bool = True) -> dict:
    """Report which expected files are present, with sizes and checksums."""
    report = {}
    for split, files in EXPECTED.items():
        for rel, size in files:
            p = dest / split / rel
            entry: dict = {"present": p.exists(), "expected_bytes": size}
            if p.exists():
                actual = p.stat().st_size
                entry["size_bytes"] = actual
                entry["size_matches"] = actual == size
                if checksums:
                    h = hashlib.sha256()
                    with p.open("rb") as fh:
                        for chunk in iter(lambda: fh.read(CHUNK), b""):
                            h.update(chunk)
                    entry["sha256"] = h.hexdigest()
            report[f"{split}/{rel}"] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=Path("data/raw"), type=Path)
    ap.add_argument("--split", choices=["training", "evaluation", "both"],
                    default="both")
    ap.add_argument("--transport", choices=["https", "aws"], default="https")
    ap.add_argument("--plan", action="store_true",
                    help="print what would be downloaded, and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-checksums", action="store_true")
    args = ap.parse_args(argv)

    splits = ["training", "evaluation"] if args.split == "both" else [args.split]

    if args.plan:
        p = plan(splits)
        for row in p["files"]:
            print(f"{row['split']:<11} {row['file']:<48} {row['mb']:>9.1f} MB")
        print(f"{'TOTAL':<11} {'':<48} {p['total_gb']:>9.2f} GB")
        return 0

    if args.verify_only:
        report = verify(args.dest, checksums=not args.no_checksums)
        print(json.dumps(report, indent=2))
        missing = [k for k, v in report.items() if not v["present"]]
        wrong = [k for k, v in report.items()
                 if v["present"] and not v.get("size_matches", True)]
        if missing:
            log.warning("%d expected files missing", len(missing))
        if wrong:
            log.error("%d files have an unexpected size", len(wrong))
        return 1 if (missing or wrong) else 0

    if args.transport == "aws":
        if not have_aws_cli():
            log.error("the AWS CLI is not installed; use --transport https")
            return 2
        rc = 0
        for s in splits:
            rc |= sync_aws(s, args.dest, args.dry_run)
        results = [{"split": s, "transport": "aws"} for s in splits]
    else:
        if args.dry_run:
            log.info("dry run: %s", json.dumps(plan(splits), indent=2))
            return 0
        rc = 0
        results = []
        for s in splits:
            res = fetch_split(s, args.dest)
            results.append(res)
            rc |= 1 if res["failed"] else 0

    manifest = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "verification": verify(args.dest, checksums=not args.no_checksums),
    }
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("manifest written to %s", args.dest / "download_manifest.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
