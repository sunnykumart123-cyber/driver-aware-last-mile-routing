"""Parsing and IO tests, driven by a miniature record in the LMRRC JSON shape.

The fixture below mirrors the structure of ``route_data.json`` /
``package_data.json`` / ``travel_times.json`` / ``actual_sequences.json`` so the
parser is exercised end to end without the 20 GB download.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.io import iter_instances, read_instance, write_instance
from data.preprocess import build_instance, metro_of
from data.splits import _hash_unit, model_split
from data.preprocess import RawBundle, iter_parsed

ROUTE = {
    "station_code": "DLA7",
    "date_YYYY_MM_DD": "2018-08-06",
    "departure_time_utc": "08:30:00",
    "executor_capacity_cm3": 4247527,
    "route_score": "High",
    "stops": {
        "AB": {"lat": 34.05, "lng": -118.25, "type": "Station", "zone_id": None},
        "CD": {"lat": 34.06, "lng": -118.26, "type": "Dropoff", "zone_id": "P-13.2A"},
        "EF": {"lat": 34.07, "lng": -118.27, "type": "Dropoff", "zone_id": "P-13.2A"},
        "GH": {"lat": 34.09, "lng": -118.29, "type": "Dropoff", "zone_id": "P-14.1B"},
    },
}
PACKAGES = {
    "CD": {
        "p1": {
            "scan_status": "DELIVERED",
            "time_window": {"start_time_utc": None, "end_time_utc": None},
            "planned_service_time_seconds": 120,
            "dimensions": {"depth_cm": 10.0, "height_cm": 10.0, "width_cm": 10.0},
        },
        "p2": {
            "scan_status": "DELIVERED",
            "time_window": {
                "start_time_utc": "2018-08-06 09:00:00",
                "end_time_utc": "2018-08-06 11:00:00",
            },
            "planned_service_time_seconds": 60,
            "dimensions": {"depth_cm": 20.0, "height_cm": 10.0, "width_cm": 5.0},
        },
    },
    "EF": {
        "p3": {
            "time_window": {
                "start_time_utc": "2018-08-06 09:30:00",
                "end_time_utc": "2018-08-06 10:00:00",
            },
            "planned_service_time_seconds": 300,
            "dimensions": {"depth_cm": 30.0, "height_cm": 20.0, "width_cm": 10.0},
        }
    },
    "GH": {
        "p4": {
            "time_window": {"start_time_utc": None, "end_time_utc": None},
            "planned_service_time_seconds": 90,
            "dimensions": {"depth_cm": 5.0, "height_cm": 5.0, "width_cm": 5.0},
        }
    },
}
TRAVEL = {
    "AB": {"AB": 0, "CD": 300, "EF": 600, "GH": 900},
    "CD": {"AB": 300, "CD": 0, "EF": 180, "GH": 720},
    "EF": {"AB": 600, "CD": 180, "EF": 0, "GH": 420},
    "GH": {"AB": 900, "CD": 720, "EF": 420, "GH": 0},
}
ACTUAL = {"actual": {"AB": 0, "CD": 1, "EF": 2, "GH": 3}}


@pytest.fixture
def parsed():
    return build_instance("RouteID_x", ROUTE, PACKAGES, TRAVEL, ACTUAL)


def test_metro_is_derived_from_the_station_code():
    assert metro_of("DLA7") == "Los Angeles"
    assert metro_of("DSE4") == "Seattle"
    assert metro_of("DCH2") == "Chicago"
    assert metro_of("DBO1") == "Boston"
    assert metro_of("DAU1") == "Austin"
    assert metro_of("DZZ9") == "Unknown"


def test_depot_is_index_zero_and_stops_are_sorted(parsed):
    assert parsed.stops[0].stop_id == "AB"
    assert parsed.stops[0].is_depot
    assert [s.stop_id for s in parsed.stops] == ["AB", "CD", "EF", "GH"]


def test_travel_times_are_converted_to_minutes(parsed):
    # 300 s -> 5 min, 900 s -> 15 min
    assert parsed.travel_time[0, 1] == pytest.approx(5.0)
    assert parsed.travel_time[0, 3] == pytest.approx(15.0)
    assert np.allclose(np.diag(parsed.travel_time), 0.0)


def test_service_time_and_volume_aggregate_over_packages(parsed):
    # stop CD: 120 s + 60 s = 3 minutes; volumes 1000 + 1000 cm^3 = 2 litres
    assert parsed.stops[1].service_time == pytest.approx(3.0)
    assert parsed.stops[1].pkg_volume == pytest.approx(2.0)
    assert parsed.stops[1].pkg_count == 2
    # stop EF: 300 s = 5 minutes, 6000 cm^3 = 6 litres
    assert parsed.stops[2].service_time == pytest.approx(5.0)
    assert parsed.stops[2].pkg_volume == pytest.approx(6.0)


def test_stop_inherits_the_tightest_package_window(parsed):
    # departure 08:30; CD's only windowed package closes at 11:00 -> 150 min
    assert parsed.stops[1].tw_start == pytest.approx(30.0)
    assert parsed.stops[1].tw_end == pytest.approx(150.0)
    # EF: 09:30-10:00 -> [60, 90]
    assert parsed.stops[2].tw_start == pytest.approx(60.0)
    assert parsed.stops[2].tw_end == pytest.approx(90.0)
    # GH has no window at all
    assert not np.isfinite(parsed.stops[3].tw_end)
    assert not parsed.stops[3].has_window


def test_zones_and_unzoned_handling(parsed):
    assert parsed.stops[0].zone == "DEPOT"
    assert parsed.zones[1:] == ["P-13.2A", "P-13.2A", "P-14.1B"]
    groups = parsed.zone_groups()
    assert groups == {"P-13.2A": [1, 2], "P-14.1B": [3]}


def test_actual_sequence_is_parsed_as_indices(parsed):
    assert parsed.actual_sequence == [0, 1, 2, 3]
    assert parsed.quality == "High"


def test_departure_metadata(parsed):
    assert parsed.meta["departure_minute_of_day"] == pytest.approx(8 * 60 + 30)
    assert parsed.meta["n_zones"] == 2


def test_bad_route_without_a_station_is_skipped():
    bad = json.loads(json.dumps(ROUTE))
    bad["stops"]["AB"]["type"] = "Dropoff"
    assert build_instance("bad", bad, PACKAGES, TRAVEL, ACTUAL) is None


def test_write_then_read_round_trips(parsed, tmp_path):
    write_instance(parsed, tmp_path)
    again = read_instance(tmp_path / "los_angeles" / "RouteID_x.parquet")
    assert again.route_id == parsed.route_id
    assert again.metro == parsed.metro
    assert again.actual_sequence == parsed.actual_sequence
    assert again.quality == parsed.quality
    assert np.allclose(again.travel_time, parsed.travel_time)
    for a, b in zip(again.stops, parsed.stops):
        assert a == b


def test_iter_instances_filters_by_metro(parsed, tmp_path):
    write_instance(parsed, tmp_path)
    found = list(iter_instances(tmp_path, metros=["Los Angeles"]))
    assert len(found) == 1
    assert list(iter_instances(tmp_path, metros=["Austin"])) == []


def test_raw_bundle_iteration(tmp_path):
    build = tmp_path / "training" / "model_build_inputs"
    build.mkdir(parents=True)
    (build / "route_data.json").write_text(json.dumps({"R1": ROUTE}))
    (build / "package_data.json").write_text(json.dumps({"R1": PACKAGES}))
    (build / "travel_times.json").write_text(json.dumps({"R1": TRAVEL}))
    (build / "actual_sequences.json").write_text(json.dumps({"R1": ACTUAL}))
    bundle = RawBundle.load(tmp_path, "training")
    got = list(iter_parsed(bundle))
    assert len(got) == 1
    assert got[0].route_id == "R1"
    assert got[0].n == 4


def test_split_hash_is_stable_and_uniform():
    assert _hash_unit("abc") == _hash_unit("abc")
    vals = [_hash_unit(f"route_{k}") for k in range(2000)]
    assert 0.0 <= min(vals) and max(vals) < 1.0
    assert abs(np.mean(vals) - 0.5) < 0.05


def test_model_split_excludes_the_held_out_metro(parsed, tmp_path):
    write_instance(parsed, tmp_path)
    train, val = model_split(tmp_path, "Austin")
    assert len(train) + len(val) == 1
    train_ho_la, val_ho_la = model_split(tmp_path, "Los Angeles")
    assert len(train_ho_la) + len(val_ho_la) == 0
