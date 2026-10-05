"""Evaluate Stage-1a in isolation (§9, Phase 4 definition of done).

    python eval_behavior.py --held-out-metro Austin --model pointer
    python eval_behavior.py --held-out-metro Austin --model both --limit 200

Reported, separately from any optimisation result:

* **next-stop accuracy** (top-1 / top-3 / top-5) under teacher forcing -- how
  often the model puts the driver's actual next stop at the top of its
  distribution given the true prefix;
* **rollout agreement** -- the fraction of positions its own greedy rollout
  agrees with sigma*, plus Kendall's tau and the positional deviation;
* **zone-sequence accuracy** -- whether the *zone* order is reproduced, which is
  what the zone-sequencing literature (Wu et al. 2022) actually targets;
* **ERP** of the greedy rollout via the official scorer, when installed;
* on the held-out metro, all of the above again, which is the generalisation
  number §6 asks for;
* for the GBM, its gain importances -- the interpretability check on the pointer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import METROS, Config, Paths
from core.io import metro_slug
from core.logging_utils import get_logger
from data.splits import held_out_metro_split, model_split
from eval.erp_official import erp_score, kendall_tau, scorer_available, sequence_deviation
from models.behavior.dataset import encode_instance

log = get_logger(__name__)
TOPK = (1, 3, 5)


def _zone_sequence(inst, order) -> list[str]:
    seen: list[str] = []
    for i in list(order)[1:]:
        z = inst.zones[i]
        if not seen or seen[-1] != z:
            seen.append(z)
    return seen


def next_stop_accuracy(model, inst) -> dict:
    """Teacher-forced top-k accuracy, read off the masked pointer distribution."""
    import torch

    from models.behavior.pointer_net import _single

    enc = encode_instance(inst)
    if enc.target is None:
        return {}
    batch = _single(enc, model.device)
    with torch.no_grad():
        # travel/zone_of are not optional: the decoder's arc-feature head is
        # where most of its accuracy lives, and omitting them evaluates the
        # model blind (observed: top-1 collapsing from 0.71 to 0.04).
        out = model.forward(
            batch["features"], batch["zone_rank"], batch["pad_mask"],
            target=batch["target"], teacher_forcing_p=1.0,
            travel=batch.get("travel"), zone_of=batch.get("zone_of"),
        )
    logits = out["logits"][0]                      # (N-1, N)
    tgt = torch.from_numpy(enc.target[1:]).to(logits.device)
    ranks = (logits > logits.gather(1, tgt[:, None])).sum(dim=1)
    res = {f"top{k}": float((ranks < k).float().mean().item()) for k in TOPK}
    res["mean_rank"] = float(ranks.float().mean().item()) + 1.0
    res["nll"] = float(
        torch.nn.functional.cross_entropy(logits, tgt).item()
    )
    return res


def rollout_metrics(inst, order) -> dict:
    actual = inst.actual_sequence
    agree = float(np.mean([a == b for a, b in zip(order, actual)]))
    z_pred, z_true = _zone_sequence(inst, order), _zone_sequence(inst, actual)
    zone_agree = float(
        np.mean([a == b for a, b in zip(z_pred, z_true)]) if z_true else np.nan
    )
    out = {
        "position_agreement": agree,
        "zone_sequence_agreement": zone_agree,
        "zone_count_pred": len(z_pred),
        "zone_count_true": len(z_true),
        "seq_dev": sequence_deviation(inst, order),
        "kendall_tau": kendall_tau(inst, order),
    }
    if scorer_available():
        try:
            out["erp"] = erp_score(inst, order)
        except Exception as exc:  # pragma: no cover
            log.debug("ERP failed on %s: %s", inst.route_id, exc)
    return out


def evaluate_pointer(model, instances, split_name: str) -> pd.DataFrame:
    rows = []
    for inst in instances:
        if inst.actual_sequence is None:
            continue
        order = model.greedy_order(encode_instance(inst))
        row = {"route_id": inst.route_id, "metro": inst.metro, "n_stops": inst.n,
               "split": split_name, "model": "pointer"}
        row.update(next_stop_accuracy(model, inst))
        row.update(rollout_metrics(inst, order))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_gbm(model, instances, split_name: str) -> pd.DataFrame:
    rows = []
    for inst in instances:
        if inst.actual_sequence is None:
            continue
        order = model.greedy_order(inst)
        p = model.transition_matrix(inst)
        actual = inst.actual_sequence
        # Top-k on the static pairwise score, the analogue of the pointer's metric.
        ranks = []
        for i, j in zip(actual, actual[1:]):
            ranks.append(int((p[i] > p[i, j]).sum()))
        row = {"route_id": inst.route_id, "metro": inst.metro, "n_stops": inst.n,
               "split": split_name, "model": "gbm"}
        for k in TOPK:
            row[f"top{k}"] = float(np.mean([r < k for r in ranks]))
        row["mean_rank"] = float(np.mean(ranks)) + 1.0
        row.update(rollout_metrics(inst, order))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--held-out-metro", default="Austin", choices=list(METROS))
    ap.add_argument("--model", default="both", choices=["pointer", "gbm", "both"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.paths = Paths().ensure()
    tag = f"ho_{metro_slug(args.held_out_metro)}"
    art = Path(cfg.paths.artifacts)

    _, val = model_split(args.processed, args.held_out_metro)
    test = held_out_metro_split(args.processed, args.held_out_metro)
    val_i = val.instances(args.limit)
    test_i = test.instances(args.limit)
    log.info("validation %d routes, held-out metro %d routes", len(val_i), len(test_i))

    frames = []
    extras: dict = {}
    if args.model in ("pointer", "both"):
        from models.behavior.pointer_net import PointerNet

        path = art / f"pointer_net_{tag}.pt"
        if not path.exists():
            log.error("missing %s: run train_behavior.py first", path)
        else:
            m = PointerNet.load(path)
            frames.append(evaluate_pointer(m, val_i, "validation"))
            frames.append(evaluate_pointer(m, test_i, "held_out_metro"))
    if args.model in ("gbm", "both"):
        from models.behavior.gbm_transition import GBMTransitionModel

        path = art / f"gbm_transition_{tag}.pkl"
        if not path.exists():
            log.error("missing %s: run train_behavior.py first", path)
        else:
            m = GBMTransitionModel.load(path)
            frames.append(evaluate_gbm(m, val_i, "validation"))
            frames.append(evaluate_gbm(m, test_i, "held_out_metro"))
            extras["gbm_importances"] = dict(
                sorted(m.importances.items(), key=lambda kv: -kv[1])
            )

    if not frames:
        return 1
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = args.out or Path(cfg.paths.results) / f"behavior_eval_{tag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    metrics = [c for c in ("top1", "top3", "top5", "mean_rank", "nll",
                           "position_agreement", "zone_sequence_agreement",
                           "seq_dev", "kendall_tau", "erp") if c in df.columns]
    summary = df.groupby(["model", "split"])[metrics].agg(["mean", "std", "count"])
    print(summary.to_string())
    payload = {
        "held_out_metro": args.held_out_metro,
        "n_validation": len(val_i),
        "n_held_out": len(test_i),
        "erp_available": scorer_available(),
        "summary": json.loads(summary.reset_index().to_json(orient="records")),
        **extras,
    }
    (Path(cfg.paths.results) / f"behavior_eval_{tag}.json").write_text(
        json.dumps(payload, indent=2)
    )
    if not scorer_available():
        log.warning(
            "the official ERP scorer is not installed: seq_dev and kendall_tau "
            "are reported instead (see eval/erp_official.py). The Phase-3 gate "
            "of §9 cannot be closed without the official scorer."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
