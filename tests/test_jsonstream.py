"""Tests for the streaming JSON readers.

These matter more than they look: the training ``travel_times.json`` is 1.8 GB,
so if the streamer mis-parses a nested brace inside a string, or an escaped
quote, the whole dataset silently comes out wrong.  Every test therefore checks
the streamed result against ``json.loads`` of the same bytes.
"""
from __future__ import annotations

import json

import pytest

from data.jsonstream import (
    InMemoryIndex,
    JsonObjectIndex,
    JsonStreamError,
    iter_top_level_objects,
)

ADVERSARIAL = {
    "plain": {"a": 1, "b": 2.5},
    "nested": {"x": {"y": {"z": [1, 2, {"deep": True}]}}},
    "braces_in_string": {"s": "a { b } c [ d ] e", "t": "}}}{{{"},
    "escaped_quotes": {"s": "he said \"hi\" and \\ left", "u": "tail\"}"},
    "unicode": {"s": "zone P-13.2A éè 中文"},
    "empty_obj": {},
    "empty_list": [],
    "scalars": 42,
    "float_sci": 1.5e-7,
    "null_value": None,
    "true_value": True,
    "false_value": False,
    "string_value": "just a string",
    "list_of_objs": [{"k": 1}, {"k": 2}],
    "route-id/with:odd.chars": {"ok": 1},
    "newlines": {"s": "line1\nline2\ttabbed"},
}


@pytest.fixture
def adversarial_file(tmp_path):
    p = tmp_path / "adversarial.json"
    p.write_text(json.dumps(ADVERSARIAL, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def pretty_file(tmp_path):
    """Same content, but indented -- whitespace everywhere."""
    p = tmp_path / "pretty.json"
    p.write_text(json.dumps(ADVERSARIAL, indent=4, ensure_ascii=False),
                 encoding="utf-8")
    return p


# ------------------------------------------------------------------- streaming
@pytest.mark.parametrize("fixture", ["adversarial_file", "pretty_file"])
@pytest.mark.parametrize("chunk", [8, 64, 4096, 1 << 22])
def test_stream_matches_json_loads(fixture, chunk, request):
    """Identical output to json.loads, at every buffer size including tiny."""
    path = request.getfixturevalue(fixture)
    got = dict(iter_top_level_objects(path, chunk=chunk))
    assert got == ADVERSARIAL


@pytest.mark.parametrize("chunk", [8, 64, 4096])
def test_stream_preserves_key_order(adversarial_file, chunk):
    keys = [k for k, _ in iter_top_level_objects(adversarial_file, chunk=chunk)]
    assert keys == list(ADVERSARIAL)


def test_stream_handles_an_empty_object(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("{}")
    assert list(iter_top_level_objects(p)) == []
    p.write_text("  {  }  ")
    assert list(iter_top_level_objects(p)) == []


def test_stream_rejects_non_object_toplevel(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(JsonStreamError, match="does not start with a JSON object"):
        list(iter_top_level_objects(p))


def test_stream_rejects_truncation(tmp_path):
    p = tmp_path / "trunc.json"
    p.write_text('{"a": {"b": 1}, "c": {"d":')
    with pytest.raises(JsonStreamError):
        list(iter_top_level_objects(p))


# ----------------------------------------------------------------------- index
@pytest.mark.parametrize("chunk", [8, 64, 4096])
def test_index_random_access(adversarial_file, chunk):
    with JsonObjectIndex(adversarial_file, chunk=chunk) as idx:
        assert len(idx) == len(ADVERSARIAL)
        assert set(idx.keys()) == set(ADVERSARIAL)
        # deliberately out of file order
        for key in reversed(list(ADVERSARIAL)):
            assert idx.get(key) == ADVERSARIAL[key]
        assert idx.get("missing") is None
        assert idx.get("missing", {}) == {}
        assert "plain" in idx and "missing" not in idx


def test_index_items_matches_dict(pretty_file):
    with JsonObjectIndex(pretty_file) as idx:
        assert dict(idx.items()) == ADVERSARIAL


def test_index_cache_is_reused_and_invalidated(adversarial_file, tmp_path):
    cache_dir = tmp_path / "cache"
    first = JsonObjectIndex.build_cached(adversarial_file, cache_dir)
    cached = list(cache_dir.glob("*.offsets.pkl"))
    assert len(cached) == 1

    second = JsonObjectIndex.build_cached(adversarial_file, cache_dir)
    assert second.offsets == first.offsets
    with second:
        assert second.get("plain") == ADVERSARIAL["plain"]

    # Rewrite the file with different content: the stale cache must be rejected.
    import os
    import time

    new = {"only": {"k": 9}}
    adversarial_file.write_text(json.dumps(new))
    os.utime(adversarial_file, (time.time() + 10, time.time() + 10))
    third = JsonObjectIndex.build_cached(adversarial_file, cache_dir)
    with third:
        assert third.keys() == ["only"]
        assert third.get("only") == {"k": 9}


def test_corrupt_cache_is_ignored(adversarial_file, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / f".{adversarial_file.name}.offsets.pkl").write_bytes(b"not a pickle")
    idx = JsonObjectIndex.build_cached(adversarial_file, cache_dir)
    with idx:
        assert idx.get("plain") == ADVERSARIAL["plain"]


def test_in_memory_index_has_the_same_interface():
    idx = InMemoryIndex({"a": 1, "b": 2})
    with idx:
        assert len(idx) == 2
        assert idx.keys() == ["a", "b"]
        assert idx.get("a") == 1
        assert idx.get("z", 7) == 7
        assert "a" in idx and "z" not in idx
        assert dict(idx.items()) == {"a": 1, "b": 2}


# ------------------------------------------------- shape of the real LMRRC file
def test_index_on_an_lmrrc_shaped_travel_matrix(tmp_path):
    """A realistic travel_times.json: route -> from-stop -> to-stop -> seconds."""
    routes = {
        f"RouteID_{r:08d}": {
            f"S{i}": {f"S{j}": float(60 * (i + j)) for j in range(12)}
            for i in range(12)
        }
        for r in range(25)
    }
    p = tmp_path / "travel_times.json"
    p.write_text(json.dumps(routes))

    with JsonObjectIndex(p, chunk=1024) as idx:
        assert len(idx) == 25
        mid = idx.get("RouteID_00000012")
        assert mid == routes["RouteID_00000012"]
        assert mid["S3"]["S4"] == 420.0


def test_streamer_memory_is_bounded(tmp_path):
    """Peak memory must track the *buffer*, not the file size.

    This is the property that makes the 1.8 GB ``travel_times.json`` tractable:
    memory is O(chunk + one route's payload), independent of how long the file
    is.  The test therefore pins peak usage to a multiple of the chunk size
    rather than to a fraction of the file.
    """
    import tracemalloc

    routes = {
        f"R{r:05d}": {f"S{i}": {f"S{j}": float(i * j) for j in range(40)}
                      for i in range(40)}
        for r in range(400)
    }
    p = tmp_path / "big.json"
    p.write_text(json.dumps(routes))
    file_size = p.stat().st_size
    assert file_size > 4_000_000          # the test is meaningless if it is tiny

    chunk = 1 << 16                       # 64 KiB buffer
    tracemalloc.start()
    count = 0
    for _key, value in iter_top_level_objects(p, chunk=chunk):
        count += 1
        assert len(value) == 40
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert count == 400
    # Bounded by the buffer plus one parsed route, with slack for dict overhead.
    assert peak < 40 * chunk, f"peak {peak} vs chunk {chunk}"
    assert peak < file_size / 4, f"peak {peak} vs file {file_size}"


def test_index_memory_is_bounded(tmp_path):
    """Building the offset index must not materialise any value."""
    import tracemalloc

    routes = {
        f"R{r:05d}": {f"S{i}": {f"S{j}": float(i * j) for j in range(40)}
                      for i in range(40)}
        for r in range(400)
    }
    p = tmp_path / "big.json"
    p.write_text(json.dumps(routes))
    file_size = p.stat().st_size

    chunk = 1 << 16
    tracemalloc.start()
    idx = JsonObjectIndex(p, chunk=chunk)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(idx) == 400
    assert peak < 40 * chunk, f"peak {peak} vs chunk {chunk}"
    assert peak < file_size / 4
    with idx:
        assert idx.get("R00399")["S39"]["S39"] == float(39 * 39)


# --------------------------------------------- ijson streaming fast path
def _has_ijson_c() -> bool:
    try:
        import ijson
    except ImportError:
        return False
    return getattr(ijson, "backend", "python").startswith("yajl")


needs_ijson = pytest.mark.skipif(
    not _has_ijson_c(), reason="ijson C backend not installed"
)


@needs_ijson
@pytest.mark.parametrize("fixture", ["adversarial_file", "pretty_file"])
def test_ijson_and_python_streams_agree(fixture, request):
    """The fast and fallback streaming paths must produce identical output.

    This is what keeps ijson an optimisation rather than a second, subtly
    different implementation of the reader.
    """
    import data.jsonstream as js

    path = request.getfixturevalue(fixture)
    fast = dict(js.iter_top_level_objects(path))
    slow = dict(js._iter_top_level_python(path, chunk=64))
    assert fast == slow == ADVERSARIAL


def test_python_fallback_is_used_when_ijson_is_absent(adversarial_file, monkeypatch):
    """With ijson unavailable the reader must still work."""
    import data.jsonstream as js

    monkeypatch.setattr(js, "_ijson_kvitems", lambda path: None)
    assert dict(js.iter_top_level_objects(adversarial_file, chunk=64)) == ADVERSARIAL
    with JsonObjectIndex(adversarial_file, chunk=64) as idx:
        assert dict(idx.items()) == ADVERSARIAL
