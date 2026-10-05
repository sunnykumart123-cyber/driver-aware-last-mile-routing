"""Stage-1 model tests: shapes, invariants, and the properties the MILP relies on.

The behaviour models feed ``c^beh_ij = -log p_ij`` into an optimizer, so the
invariants that matter are: every row of ``p`` is a distribution, no entry is
zero (or the cost would be infinite), and the diagonal and the depot column are
never attractive.  Those are asserted here rather than assumed downstream.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import BehaviorConfig, UncertaintyConfig
from core.types import is_valid_order
from models.behavior.cost import UniformBehavior, build_behavior_cost
from models.behavior.dataset import (
    N_STOP_FEATURES,
    collate,
    encode_instance,
    encode_many,
    zone_ranks,
)
from models.behavior.gbm_transition import GBMTransitionModel, TransitionTables
from models.uncertainty.ambiguity_set import AmbiguitySet
from models.uncertainty.quantile_model import (
    QuantileUncertaintyModel,
    build_training_tables,
)
from tests.conftest import TOY4_SERVICE, TOY4_TT, make_instance


def _training_set(k: int = 6):
    """A handful of toy routes with sigma*, enough to exercise the training code."""
    out = []
    rng = np.random.default_rng(5)
    for i in range(k):
        tt = TOY4_TT * (1.0 + 0.1 * rng.standard_normal())
        np.fill_diagonal(tt, 0.0)
        out.append(
            make_instance(
                tt, TOY4_SERVICE,
                zones=["DEPOT", "Z1", "Z1", "Z2"],
                tw_end=[np.inf, np.inf, np.inf, 40.0 + i],
                route_id=f"r{i}",
                metro="Seattle" if i % 2 else "Chicago",
                actual_sequence=[0, 1, 2, 3],
            )
        )
    return out


# ------------------------------------------------------------------- dataset
def test_encoding_shapes_and_zone_ranks(toy_zones):
    enc = encode_instance(toy_zones)
    assert enc.features.shape == (toy_zones.n, N_STOP_FEATURES)
    assert np.isfinite(enc.features).all()
    zr, zl = zone_ranks(toy_zones)
    assert zr[0] == 0                          # depot always rank 0
    assert set(zr[1:]) == {1, 2}               # two zones, ranked by depot time
    assert len(set(zl[1:])) == 2


def test_collate_pads_and_masks(toy4, toy_zones):
    batch = collate(encode_many([toy4, toy_zones]))
    assert batch["features"].shape == (2, 5, N_STOP_FEATURES)
    assert batch["pad_mask"][0].tolist() == [False] * 4 + [True]
    assert batch["pad_mask"][1].tolist() == [False] * 5
    assert batch["lengths"].tolist() == [4, 5]


# ------------------------------------------------------------- uniform baseline
def test_uniform_behavior_is_a_proper_distribution(toy_zones):
    model = UniformBehavior()
    p = model.transition_matrix(toy_zones)
    # The diagonal is floored, not zeroed: -log(0) would be infinite.
    assert np.allclose(np.diag(p), model.prob_floor)
    assert (p > 0).all()
    cost = build_behavior_cost(UniformBehavior(), toy_zones)
    assert np.isfinite(cost).all()
    # A uniform model must not prefer any delivery arc over another; the depot
    # column is deliberately unattractive, so it is excluded.
    off = ~np.eye(toy_zones.n, dtype=bool)
    off[:, 0] = False
    assert cost[off].std() == pytest.approx(0.0, abs=1e-9)
    assert (cost[:, 0][1:] > cost[off].max()).all()


# ------------------------------------------------------------------ pointer net
def test_pointer_net_forward_loss_and_matrix(toy_zones):
    import torch

    from models.behavior.pointer_net import PointerNet, count_parameters

    cfg = BehaviorConfig(embed_dim=16, hidden_dim=16, n_layers=1, epochs=1)
    model = PointerNet(cfg)
    assert count_parameters(model) > 0

    batch = collate(encode_many([toy_zones]))
    loss = model.loss(batch, teacher_forcing_p=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    p = model.transition_matrix(toy_zones)
    assert p.shape == (toy_zones.n, toy_zones.n)
    assert (p > 0).all()
    assert np.allclose(np.diag(p), 0.0, atol=cfg.prob_floor)
    cost = model.behavior_cost_matrix(toy_zones)
    assert np.isfinite(cost).all()


def test_pointer_net_greedy_order_is_a_permutation(toy_zones):
    from models.behavior.pointer_net import PointerNet

    model = PointerNet(BehaviorConfig(embed_dim=16, hidden_dim=16, n_layers=1))
    order = model.greedy_order(encode_instance(toy_zones))
    assert is_valid_order(order, toy_zones.n)


def test_pointer_net_can_overfit_one_route(toy_zones):
    """A capacity sanity check: the loss must be driveable down on one example."""
    import torch

    from models.behavior.pointer_net import PointerNet

    torch.manual_seed(0)
    model = PointerNet(BehaviorConfig(embed_dim=32, hidden_dim=32, n_layers=1))
    batch = collate(encode_many([toy_zones]))
    # lr must stay below the logit-clipping scale; see the note in pointer_net.
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    first = float(model.loss(batch).item())
    for _ in range(120):
        opt.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        opt.step()
    assert float(loss.item()) < first * 0.1
    assert model.greedy_order(encode_instance(toy_zones)) == toy_zones.actual_sequence


def test_pointer_net_round_trips_through_disk(toy_zones, tmp_path):
    from models.behavior.pointer_net import PointerNet

    model = PointerNet(BehaviorConfig(embed_dim=16, hidden_dim=16, n_layers=1))
    before = model.transition_matrix(toy_zones)
    path = model.save(tmp_path / "pointer.pt")
    again = PointerNet.load(path)
    assert np.allclose(before, again.transition_matrix(toy_zones))


def test_scheduled_sampling_curriculum():
    from models.behavior.pointer_net import scheduled_sampling_p

    cfg = BehaviorConfig(epochs=10, teacher_forcing_epochs=4,
                         scheduled_sampling_final_p=0.5)
    assert scheduled_sampling_p(0, cfg) == 1.0
    assert scheduled_sampling_p(3, cfg) == 1.0
    assert scheduled_sampling_p(4, cfg) == pytest.approx(1.0)
    assert scheduled_sampling_p(9, cfg) < 1.0
    assert scheduled_sampling_p(9, cfg) >= cfg.scheduled_sampling_final_p


# -------------------------------------------------------------------- GBM
def test_transition_tables_are_row_normalised():
    tables = TransitionTables.fit(_training_set())
    rows = tables.zone_rank_freq.sum(axis=1)
    nonzero = rows[rows > 0]
    assert np.allclose(nonzero, 1.0)
    assert sum(tables.zone_pair_freq.values()) == pytest.approx(1.0)


def test_gbm_transition_matrix_and_greedy(toy_zones):
    model = GBMTransitionModel.fit(_training_set(), n_negatives=2)
    p = model.transition_matrix(toy_zones)
    assert p.shape == (toy_zones.n, toy_zones.n)
    assert (p > 0).all()
    assert np.allclose(np.diag(p), 0.0, atol=model.prob_floor)
    order = model.greedy_order(toy_zones)
    assert is_valid_order(order, toy_zones.n)


def test_gbm_round_trips_through_disk(toy_zones, tmp_path):
    model = GBMTransitionModel.fit(_training_set(), n_negatives=2)
    path = model.save(tmp_path / "gbm.pkl")
    again = GBMTransitionModel.load(path)
    assert np.allclose(model.transition_matrix(toy_zones),
                       again.transition_matrix(toy_zones))


# ------------------------------------------------------------- uncertainty model
def test_quantile_model_produces_monotone_quantiles():
    insts = _training_set(8)
    cfg = UncertaintyConfig(n_scenarios=8)
    model = QuantileUncertaintyModel.fit(insts, cfg)
    tr, sv = build_training_tables(insts, model.stats)
    pred = model.predict_travel(tr)
    assert pred.shape == (len(tr), len(model.quantiles))
    assert (np.diff(pred, axis=1) >= -1e-9).all()
    assert (pred >= 0).all()
    pred_s = model.predict_service(sv)
    assert (np.diff(pred_s, axis=1) >= -1e-9).all()


def test_instance_quantiles_are_anchored_on_the_observed_times(toy_zones):
    """The median quantile must reproduce the route's own supplied t_ij and s_i."""
    model = QuantileUncertaintyModel.fit(_training_set(8), UncertaintyConfig())
    tq = model.instance_travel_quantiles(toy_zones)
    sq = model.instance_service_quantiles(toy_zones)
    mid = len(model.quantiles) // 2
    off = ~np.eye(toy_zones.n, dtype=bool)
    assert np.allclose(tq[:, :, mid][off], toy_zones.travel_time[off], atol=1e-3)
    assert np.allclose(sq[1:, mid], toy_zones.service_time[1:], atol=1e-3)
    assert np.allclose(np.diagonal(tq[:, :, mid]), 0.0)
    assert np.allclose(sq[0], 0.0)


def test_ambiguity_set_uses_common_random_numbers(toy_zones):
    model = QuantileUncertaintyModel.fit(_training_set(8), UncertaintyConfig())
    cfg = UncertaintyConfig(n_scenarios=16, seed=42)
    a = AmbiguitySet.from_model(toy_zones, model, cfg, np.random.default_rng(cfg.seed))
    b = AmbiguitySet.from_model(toy_zones, model, cfg, np.random.default_rng(cfg.seed))
    assert np.allclose(a.travel, b.travel)
    assert np.allclose(a.service, b.service)
    assert a.travel.shape == (16, toy_zones.n, toy_zones.n)
    assert np.allclose(a.service[:, 0], 0.0)
    for k in range(a.n_scenarios):
        assert np.allclose(np.diagonal(a.travel[k]), 0.0)


def test_degradation_hooks_change_the_fitted_model():
    insts = _training_set(10)
    full = QuantileUncertaintyModel.fit(insts, UncertaintyConfig())
    coarse = QuantileUncertaintyModel.fit(
        insts, UncertaintyConfig(degrade_quantiles=True)
    )
    few = QuantileUncertaintyModel.fit(
        insts, UncertaintyConfig(degrade_features=True)
    )
    assert len(coarse.quantiles) < len(full.quantiles)
    assert len(few.travel_head.features) < len(full.travel_head.features)


def test_scheduled_sampling_loss_stays_finite(toy_zones):
    """Regression: a diverged rollout must not explode the loss.

    With scheduled sampling the rollout can leave sigma*, after which the
    teacher's next stop may already be visited.  Its logit is then the mask
    value and the cross-entropy against it becomes astronomical -- a real run
    logged a train loss of 1.98e8 in the first scheduled-sampling epoch.  Those
    steps carry no teacher signal and must be masked out instead.
    """
    import torch

    from models.behavior.pointer_net import PointerNet

    torch.manual_seed(0)
    model = PointerNet(BehaviorConfig(embed_dim=16, hidden_dim=16, n_layers=1))
    batch = collate(encode_many([toy_zones, toy_zones]))
    for p in (1.0, 0.75, 0.5, 0.0):
        loss = model.loss(batch, teacher_forcing_p=p)
        assert torch.isfinite(loss), f"non-finite loss at teacher_forcing_p={p}"
        # ln(n) for the largest candidate set is the uniform-policy scale; a
        # mask leak shows up as something astronomically larger.
        assert float(loss.detach()) < 100.0, f"loss {float(loss.detach())} exploded at p={p}"


def test_target_reachable_mask_is_reported(toy_zones):
    import torch

    from models.behavior.pointer_net import PointerNet

    model = PointerNet(BehaviorConfig(embed_dim=16, hidden_dim=16, n_layers=1))
    batch = collate(encode_many([toy_zones]))
    out = model.forward(
        batch["features"], batch["zone_rank"], batch["pad_mask"],
        target=batch["target"], teacher_forcing_p=0.0,
        travel=batch.get("travel"), zone_of=batch.get("zone_of"),
    )
    assert "target_reachable" in out
    assert out["target_reachable"].dtype == torch.bool
    assert out["target_reachable"].shape == (1, toy_zones.n - 1)
    # Under full teacher forcing every target is reachable by construction.
    out_tf = model.forward(
        batch["features"], batch["zone_rank"], batch["pad_mask"],
        target=batch["target"], teacher_forcing_p=1.0,
        travel=batch.get("travel"), zone_of=batch.get("zone_of"),
    )
    assert bool(out_tf["target_reachable"].all())


def test_arc_features_are_what_make_it_learn(toy_zones):
    """The decoder needs current-stop-relative features, not just per-stop ones.

    Without them the policy on real routes sat exactly at the uniform-loss
    floor.  Here, on a single route, the arc head must at least change the
    logits -- i.e. it is wired into the forward pass at all.
    """
    import torch

    from models.behavior.pointer_net import PointerNet

    torch.manual_seed(0)
    model = PointerNet(BehaviorConfig(embed_dim=32, hidden_dim=32, n_layers=1))
    batch = collate(encode_many([toy_zones]))
    with torch.no_grad():
        with_arc = model.forward(
            batch["features"], batch["zone_rank"], batch["pad_mask"],
            travel=batch["travel"], zone_of=batch["zone_of"],
        )["logits"]
        model.arc_scale.zero_()
        without = model.forward(
            batch["features"], batch["zone_rank"], batch["pad_mask"],
            travel=batch["travel"], zone_of=batch["zone_of"],
        )["logits"]
    finite = with_arc > -1e8
    assert not torch.allclose(with_arc[finite], without[finite])
