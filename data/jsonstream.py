"""Streaming access to the LMRRC JSON files, which are too big to load whole.

The release ships one JSON object per file, keyed by route id::

    {"RouteID_0000...": { ... }, "RouteID_0001...": { ... }, ...}

``travel_times.json`` alone is 1.8 GB on the training split (0.8 GB on the
evaluation split), and as a Python dict of dicts of floats it would need
something like an order of magnitude more than that in RAM.  ``json.load`` on it
is not an option on an ordinary machine.

Two tools here, both dependency-free:

:func:`iter_top_level_objects`
    stream ``(key, value)`` pairs, parsing one route's payload at a time;

:class:`JsonObjectIndex`
    one cheap scan records the byte offset of every top-level key, after which
    any route can be parsed on demand by seek.  Memory is a few bytes per route
    regardless of file size, and routes can be visited in any order -- which is
    what :mod:`data.preprocess` needs, since it has to join four files on route
    id and their key orders are not guaranteed to agree.

Both are deliberately specialised to this shape: a single top-level object whose
values are self-contained JSON values.  They do not attempt to be a general
incremental JSON parser.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterator

from core.logging_utils import get_logger

log = get_logger(__name__)

CHUNK = 1 << 22            # 4 MiB reads
_WS = b" \t\r\n"
# Structurally interesting bytes outside a string, and inside one.  Commas only
# matter at depth 0, to terminate a scalar; inside a container they are noise,
# and in these files they outnumber everything else by orders of magnitude --
# stopping on every one of them is what made the first version crawl.
_TOKEN_DEPTH0 = re.compile(rb'["{}\[\],]')
_TOKEN_NESTED = re.compile(rb'["{}\[\]]')
_IN_STRING = re.compile(rb'["\\]')


class JsonStreamError(ValueError):
    """The file is not a single top-level JSON object of the expected shape."""


class _Scanner:
    """Byte-level cursor over a file, with a sliding buffer."""

    def __init__(self, fh, chunk: int = CHUNK):
        self.fh = fh
        self.chunk = chunk
        self.buf = b""
        self.base = 0          # file offset of buf[0]
        self.pos = 0           # cursor within buf

    @property
    def offset(self) -> int:
        return self.base + self.pos

    def _fill(self) -> bool:
        data = self.fh.read(self.chunk)
        if not data:
            return False
        # Drop the consumed prefix so the buffer does not grow without bound.
        self.buf = self.buf[self.pos:] + data
        self.base += self.pos
        self.pos = 0
        return True

    def peek(self) -> int | None:
        while self.pos >= len(self.buf):
            if not self._fill():
                return None
        return self.buf[self.pos]

    def next_byte(self) -> int | None:
        b = self.peek()
        if b is not None:
            self.pos += 1
        return b

    def skip_ws(self) -> int | None:
        while True:
            b = self.peek()
            if b is None:
                return None
            if b not in _WS:
                return b
            self.pos += 1

    def read_string(self) -> str:
        """Read a JSON string; the cursor must sit on the opening quote."""
        if self.next_byte() != 0x22:
            raise JsonStreamError("expected a string key")
        out = bytearray()
        escaped = False
        while True:
            b = self.next_byte()
            if b is None:
                raise JsonStreamError("unterminated string")
            if escaped:
                out.append(b)
                escaped = False
                continue
            if b == 0x5C:              # backslash
                out.append(b)
                escaped = True
                continue
            if b == 0x22:
                break
            out.append(b)
        return json.loads(b'"' + bytes(out) + b'"')

    def scan_value(self, capture: bool) -> bytes | None:
        """Advance past one JSON value; return its bytes when ``capture``.

        Containers are tracked by depth with string awareness; scalars end at
        the first delimiter at depth zero.

        The inner loop uses compiled-regex jumps rather than a per-byte Python
        loop.  That is not micro-optimisation: on the 1.73 GB training
        ``travel_times.json`` the byte-at-a-time version indexed at ~4 MB/s
        (about 7 minutes); jumping to the next structurally interesting byte
        runs an order of magnitude faster, and the vast majority of these files
        is digits and commas inside number-valued leaves.
        """
        # Skip leading whitespace *before* anchoring the capture: skip_ws may
        # refill the buffer, which resets `pos` and would invalidate an offset
        # taken earlier.  This is the subtle one -- it only shows up when a
        # value straddles a buffer boundary.
        if self.skip_ws() is None:
            raise JsonStreamError("expected a value")
        start_pos = self.pos
        chunks = bytearray() if capture else None
        depth = 0
        in_str = False
        escaped = False

        def _flush() -> None:
            if capture:
                chunks.extend(self.buf[start_pos:self.pos])

        skip_next = False          # an escape straddled a buffer boundary
        while True:
            if self.pos >= len(self.buf):
                _flush()
                if not self._fill():
                    if depth != 0 or in_str:
                        raise JsonStreamError("truncated value")
                    break
                start_pos = self.pos
            if skip_next:
                self.pos += 1
                skip_next = False
                continue

            if in_str:
                m = _IN_STRING.search(self.buf, self.pos)
                if m is None:
                    self.pos = len(self.buf)
                    continue
                self.pos = m.end()
                if m.group() == b"\\":
                    if self.pos >= len(self.buf):
                        skip_next = True        # resolve after the next refill
                    else:
                        self.pos += 1
                else:
                    in_str = False
                continue

            pattern = _TOKEN_DEPTH0 if depth == 0 else _TOKEN_NESTED
            m = pattern.search(self.buf, self.pos)
            if m is None:
                self.pos = len(self.buf)
                continue
            self.pos = m.end()
            b = m.group()
            if b == b'"':
                in_str = True
            elif b in b"{[":
                depth += 1
            elif b in b"}]":
                if depth == 0:
                    self.pos -= 1               # leave the closing brace unread
                    break
                depth -= 1
                if depth == 0:
                    break
            elif b == b"," and depth == 0:
                self.pos -= 1                   # leave the comma unread
                break

        _flush()
        return bytes(chunks) if capture else None


def _open_top_level(scanner: _Scanner) -> None:
    if scanner.skip_ws() != 0x7B:
        raise JsonStreamError("file does not start with a JSON object")
    scanner.next_byte()


def _ijson_kvitems(path: Path):
    """``(key, value)`` iterator over the top-level object via ijson's C backend.

    Returns ``None`` when ijson (or its C backend) is unavailable, so the pure
    Python reader stays the always-present fallback.
    """
    try:
        import ijson
    except ImportError:
        return None
    if not getattr(ijson, "backend", "python").startswith("yajl"):
        return None            # the pure-Python ijson backend buys nothing

    def _gen():
        with Path(path).open("rb") as fh:
            try:
                yield from ijson.kvitems(fh, "", use_float=True)
            except ijson.common.JSONError as exc:
                # Present one error type to callers whichever backend ran.
                raise JsonStreamError(str(exc)) from exc

    return _gen()


def _first_meaningful_byte(path: Path, probe: int = 4096) -> int | None:
    with path.open("rb") as fh:
        while True:
            block = fh.read(probe)
            if not block:
                return None
            stripped = block.lstrip(_WS)
            if stripped:
                return stripped[0]


def iter_top_level_objects(
    path: str | Path, chunk: int = CHUNK
) -> Iterator[tuple[str, Any]]:
    """Yield ``(key, parsed_value)`` for each member of the top-level object.

    Uses ijson's C backend when it is installed and falls back to the pure
    Python scanner otherwise; ``tests/test_jsonstream.py`` asserts the two
    produce identical output.
    """
    path = Path(path)
    # ijson silently yields nothing for a non-object document; the contract here
    # is that the file *is* a single top-level object, so check it explicitly.
    if _first_meaningful_byte(path) != 0x7B:
        raise JsonStreamError("file does not start with a JSON object")
    fast = _ijson_kvitems(path)
    if fast is not None:
        yield from fast
        return
    yield from _iter_top_level_python(path, chunk)


def _iter_top_level_python(
    path: str | Path, chunk: int = CHUNK
) -> Iterator[tuple[str, Any]]:
    with Path(path).open("rb") as fh:
        sc = _Scanner(fh, chunk)
        _open_top_level(sc)
        while True:
            b = sc.skip_ws()
            if b is None:
                raise JsonStreamError("unterminated top-level object")
            if b == 0x7D:               # '}'
                return
            if b == 0x2C:               # ','
                sc.next_byte()
                continue
            key = sc.read_string()
            if sc.skip_ws() != 0x3A:
                raise JsonStreamError(f"expected ':' after key {key!r}")
            sc.next_byte()
            raw = sc.scan_value(capture=True)
            yield key, json.loads(raw)


class JsonObjectIndex:
    """Random access by top-level key, via a byte-offset index.

    ::

        with JsonObjectIndex("travel_times.json") as idx:
            print(len(idx))
            payload = idx.get("RouteID_0000...")

    The index is built once per file (a single sequential scan that parses no
    values) and can be cached next to the file with :meth:`build_cached`, which
    matters because the pipeline reads these files repeatedly.
    """

    def __init__(self, path: str | Path, chunk: int = CHUNK,
                 offsets: dict[str, tuple[int, int]] | None = None):
        self.path = Path(path)
        self.chunk = chunk
        self._fh = None
        self.offsets = offsets if offsets is not None else self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> dict[str, tuple[int, int]]:
        return self._build_python()

    def _build_python(self) -> dict[str, tuple[int, int]]:
        """Offset index by direct byte scan.

        ijson is *not* usable here: ``ijson.parse`` yields ``(prefix, event,
        value)`` where ``prefix`` is a JSON path string, not a byte position, so
        it cannot report where a value starts and ends.  It is used for
        sequential streaming instead -- see :func:`iter_top_level_objects`.
        """
        offsets: dict[str, tuple[int, int]] = {}
        with self.path.open("rb") as fh:
            sc = _Scanner(fh, self.chunk)
            _open_top_level(sc)
            while True:
                b = sc.skip_ws()
                if b is None:
                    raise JsonStreamError("unterminated top-level object")
                if b == 0x7D:
                    break
                if b == 0x2C:
                    sc.next_byte()
                    continue
                key = sc.read_string()
                if sc.skip_ws() != 0x3A:
                    raise JsonStreamError(f"expected ':' after key {key!r}")
                sc.next_byte()
                sc.skip_ws()
                start = sc.offset
                sc.scan_value(capture=False)
                offsets[key] = (start, sc.offset - start)
        log.info("indexed %d top-level keys in %s", len(offsets), self.path.name)
        return offsets

    @classmethod
    def build_cached(cls, path: str | Path, cache_dir: str | Path | None = None,
                     chunk: int = CHUNK) -> "JsonObjectIndex":
        """Load the offset index from a sidecar cache, building it if stale."""
        path = Path(path)
        cache_dir = Path(cache_dir) if cache_dir else path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f".{path.name}.offsets.pkl"
        stat = path.stat()
        if cache.exists():
            try:
                with cache.open("rb") as fh:
                    blob = pickle.load(fh)
                if (blob.get("size") == stat.st_size
                        and blob.get("mtime") == int(stat.st_mtime)):
                    log.info("reusing offset index %s", cache.name)
                    return cls(path, chunk, offsets=blob["offsets"])
            except Exception as exc:  # pragma: no cover - corrupt cache
                log.warning("ignoring unreadable index cache %s: %s", cache, exc)
        idx = cls(path, chunk)
        with cache.open("wb") as fh:
            pickle.dump(
                {"size": stat.st_size, "mtime": int(stat.st_mtime),
                 "offsets": idx.offsets},
                fh,
            )
        return idx

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.offsets)

    def __contains__(self, key: str) -> bool:
        return key in self.offsets

    def keys(self) -> list[str]:
        return list(self.offsets)

    def open(self) -> "JsonObjectIndex":
        if self._fh is None:
            self._fh = self.path.open("rb")
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonObjectIndex":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, key: str, default: Any = None) -> Any:
        loc = self.offsets.get(key)
        if loc is None:
            return default
        self.open()
        start, length = loc
        self._fh.seek(start)
        return json.loads(self._fh.read(length))

    def items(self) -> Iterator[tuple[str, Any]]:
        for k in self.offsets:
            yield k, self.get(k)


class InMemoryIndex:
    """The same interface over an already-parsed dict.

    Lets the preprocessing code treat small files (and the test fixtures)
    exactly like indexed big ones.
    """

    def __init__(self, data: dict | None = None):
        self.data = data or {}

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self) -> list[str]:
        return list(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def items(self):
        return self.data.items()

    def open(self) -> "InMemoryIndex":
        return self

    def close(self) -> None:
        pass

    def __enter__(self) -> "InMemoryIndex":
        return self

    def __exit__(self, *exc) -> None:
        pass
