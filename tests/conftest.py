"""Test fixtures: hand-built toy instances with hand-computed optima.

These are *unit-test fixtures*, not data.  They exist so the MILP encoding, the
CVaR/DRO algebra and the ALNS can each be checked against a value computed by
hand or by exhaustive enumeration, which is the Phase-5 definition of done in
§9.  No experimental result in this repository is ever produced from them -- all
reported numbers come from the real LMRRC routes under ``data/processed``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config, SolverConfig, UncertaintyConfig  # noqa: E402
from core.objective import CombinedObjective  # noqa: E402
from core.types import Instance, Stop  # noqa: E402
from models.uncertainty.ambiguity_set import AmbiguitySet  # noqa: E402
from solvers.reliability import ReliabilityEvaluator  # noqa: E402

# The 4-node toy used throughout: depot + three stops on a line.
TOY4_TT = np.array(
    [
        [0.0, 10.0, 20.0, 30.0],
        [10.0, 0.0, 10.0, 20.0],
        [20.0, 10.0, 0.0, 10.0],
        [30.0, 20.0, 10.0, 0.0],
    ]
)
TOY4_SERVICE = [0.0, 5.0, 5.0, 5.0]


def make_instance(
    travel: np.ndarray,
    service,
    zones,
    tw_end,
    route_id: str = "toy",
    metro: str = "Los Angeles",
    actual_sequence=None,
) -> Instance:
    n = len(service)
    stops = [
        Stop(
            stop_id=f"S{i}",
            zone=zones[i],
            lat=34.0 + 0.01 * i,
            lng=-118.0 - 0.01 * i,
            pkg_count=0 if i == 0 else 2,
            pkg_volume=0.0 if i == 0 else 12.0,
            tw_start=-np.inf,
            tw_end=tw_end[i],
            service_time=service[i],
            is_depot=(i == 0),
        )
        for i in range(n)
    ]
    return Instance(
        route_id=route_id,
        metro=metro,
        station="DLA9",
        stops=stops,
        travel_time=np.asarray(travel, dtype=float),
        actual_sequence=actual_sequence,
        quality="High" if actual_sequence else None,
        meta={"departure_minute_of_day": 480.0, "n_zones": len(set(zones[1:]))},
    )


def uniform_cost(n: int, good_arcs, good: float = 1.0, bad: float = 5.0) -> np.ndarray:
    """Behaviour cost matrix that makes ``good_arcs`` the plausible path."""
    c = np.full((n, n), bad, dtype=float)
    for i, j in good_arcs:
        c[i, j] = good
    np.fill_diagonal(c, 0.0)
    return c


def make_objective(inst: Instance, cost: np.ndarray, lam: float,
                   eps: float = 0.1, delta: float = 0.0,
                   geometry: str = "wasserstein") -> CombinedObjective:
    unc = UncertaintyConfig(eps=eps, delta=delta, geometry=geometry, n_scenarios=1)
    amb = AmbiguitySet.deterministic(inst, unc)
    return CombinedObjective(
        inst=inst, behavior_cost=cost,
        evaluator=ReliabilityEvaluator(inst, amb), lam=lam,
    )


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def toy4() -> Instance:
    """Depot + 3 stops, one zone, a single window at stop 3 (b = 35 minutes)."""
    return make_instance(
        TOY4_TT, TOY4_SERVICE,
        zones=["DEPOT", "Z1", "Z1", "Z1"],
        tw_end=[np.inf, np.inf, np.inf, 35.0],
        route_id="toy4",
        actual_sequence=[0, 1, 2, 3],
    )


@pytest.fixture
def toy4_loose() -> Instance:
    """The same route with a window that the natural order just meets (b = 40)."""
    return make_instance(
        TOY4_TT, TOY4_SERVICE,
        zones=["DEPOT", "Z1", "Z1", "Z1"],
        tw_end=[np.inf, np.inf, np.inf, 40.0],
        route_id="toy4_loose",
        actual_sequence=[0, 1, 2, 3],
    )


@pytest.fixture
def toy4_cost() -> np.ndarray:
    """``c^beh`` making 0-1-2-3 the unique behaviourally plausible path."""
    return uniform_cost(4, [(0, 1), (1, 2), (2, 3)])


@pytest.fixture
def toy_zones() -> Instance:
    """Depot + 4 stops in two zones, used for the zone-precedence constraints."""
    tt = np.array(
        [
            [0.0, 10.0, 12.0, 30.0, 32.0],
            [10.0, 0.0, 4.0, 22.0, 25.0],
            [12.0, 4.0, 0.0, 20.0, 23.0],
            [30.0, 22.0, 20.0, 0.0, 5.0],
            [32.0, 25.0, 23.0, 5.0, 0.0],
        ]
    )
    return make_instance(
        tt, [0.0, 4.0, 4.0, 4.0, 4.0],
        zones=["DEPOT", "A", "B", "A", "B"],
        tw_end=[np.inf, np.inf, np.inf, np.inf, 90.0],
        route_id="toy_zones",
        actual_sequence=[0, 1, 3, 2, 4],
    )


@pytest.fixture
def stochastic_toy4() -> tuple[Instance, AmbiguitySet]:
    """Toy4 with a three-scenario ``P_hat`` written out explicitly.

    Scenario travel times are 0.8x, 1.0x and 1.4x the planned matrix, so every
    CVaR in the tests can be checked by hand.
    """
    inst = make_instance(
        TOY4_TT, TOY4_SERVICE,
        zones=["DEPOT", "Z1", "Z1", "Z1"],
        tw_end=[np.inf, np.inf, np.inf, 35.0],
        route_id="toy4_stoch",
        actual_sequence=[0, 1, 2, 3],
    )
    factors = np.array([0.8, 1.0, 1.4])
    travel = (TOY4_TT[None, :, :] * factors[:, None, None]).astype(np.float32)
    service = np.tile(np.array(TOY4_SERVICE, dtype=np.float32), (3, 1))
    service[:, 0] = 0.0
    amb = AmbiguitySet(travel=travel, service=service, eps=1 / 3, delta=0.0,
                       ground_norm="linf", geometry="wasserstein",
                       route_id=inst.route_id)
    return inst, amb


@pytest.fixture
def solver_cfg() -> SolverConfig:
    return SolverConfig(
        time_limit_s=60.0, alns_iterations=600, alns_segment=50,
        zone_precedence=True, exact_backend="pulp", seed=7,
    )


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def stochastic_toy_zones() -> tuple[Instance, AmbiguitySet]:
    """A five-scenario zoned instance.

    Exists because a single-scenario ambiguity set hides broadcasting bugs: with
    ``S = 1`` a per-stop quantity of shape ``(1,)`` silently broadcasts against a
    per-position array of shape ``(1, P)``.  Every solver test that can run
    multi-scenario should use this fixture.
    """
    tt = np.array(
        [
            [0.0, 10.0, 12.0, 30.0, 32.0],
            [10.0, 0.0, 4.0, 22.0, 25.0],
            [12.0, 4.0, 0.0, 20.0, 23.0],
            [30.0, 22.0, 20.0, 0.0, 5.0],
            [32.0, 25.0, 23.0, 5.0, 0.0],
        ]
    )
    service = [0.0, 4.0, 4.0, 4.0, 4.0]
    inst = make_instance(
        tt, service,
        zones=["DEPOT", "A", "B", "A", "B"],
        tw_end=[np.inf, np.inf, 70.0, np.inf, 85.0],
        route_id="toy_zones_stoch",
        actual_sequence=[0, 1, 3, 2, 4],
    )
    factors = np.array([0.7, 0.9, 1.0, 1.2, 1.5])
    travel = (tt[None, :, :] * factors[:, None, None]).astype(np.float32)
    srv = (np.array(service, dtype=np.float32)[None, :]
           * factors[:, None].astype(np.float32))
    srv[:, 0] = 0.0
    amb = AmbiguitySet(travel=travel, service=srv, eps=0.2, delta=0.0,
                       ground_norm="linf", geometry="wasserstein",
                       route_id=inst.route_id)
    return inst, amb
