"""Train Stage-1a (§4a).

    python train_behavior.py --held-out-metro Austin --model pointer --epochs 20
    python train_behavior.py --held-out-metro Austin --model gbm
    python train_behavior.py --all-folds --model both        # 5 LOMO folds

Artefacts land in ``results/artifacts/`` tagged by fold
(``pointer_net_ho_austin.pt``), which is what
``pipeline.run_pipeline`` and ``eval_behavior.py`` look for.  Training curves and
the config are written next to them so a run is reproducible from its artefact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from core.config import METROS, BehaviorConfig, Config, Paths
from core.io import metro_slug
from core.logging_utils import get_logger
from data.splits import model_split
from models.behavior.dataset import collate, encode_many
from models.behavior.gbm_transition import GBMTransitionModel

log = get_logger(__name__)


def _fold_tag(metro: str | None) -> str:
    return f"ho_{metro_slug(metro)}" if metro else ""


def _top1_accuracy(model, batch) -> float:
    """Teacher-forced next-stop accuracy -- the interpretable training signal.

    Cross-entropy alone is hard to read because the number of candidates shrinks
    along the route; top-1 accuracy against sigma* is not.
    """
    import torch

    out = model.forward(
        batch["features"], batch["zone_rank"], batch["pad_mask"],
        target=batch["target"], teacher_forcing_p=1.0,
        travel=batch.get("travel"), zone_of=batch.get("zone_of"),
    )
    logits = out["logits"]
    tgt = batch["target"][:, 1:]
    lengths = batch["lengths"]
    valid = torch.arange(tgt.shape[1])[None, :] < (lengths[:, None] - 1)
    if not valid.any():
        return float("nan")
    pred = logits.argmax(dim=-1)
    return float(((pred == tgt) & valid).sum().item() / valid.sum().item())


def _uniform_loss(batch) -> float:
    """Cross-entropy of the uniform-over-remaining policy, as a floor to beat."""
    import torch

    lengths = batch["lengths"]
    total, count = 0.0, 0
    for m in lengths.tolist():
        for remaining in range(m - 1, 0, -1):
            total += float(np.log(remaining))
            count += 1
    return total / max(count, 1)


# ------------------------------------------------------------------ pointer net
def train_pointer(
    train_instances,
    val_instances,
    cfg: BehaviorConfig,
    out_dir: Path,
    tag: str,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from models.behavior.pointer_net import (
        PointerNet,
        count_parameters,
        scheduled_sampling_p,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_enc = [e for e in encode_many(train_instances)
                 if e.target is not None and e.n <= cfg.max_stops]
    val_enc = [e for e in encode_many(val_instances)
               if e.target is not None and e.n <= cfg.max_stops]
    if not train_enc:
        raise SystemExit(
            "no training routes with an actual sequence: check the processed "
            "data and the quality filter in data/splits.py"
        )
    log.info("pointer net: %d train / %d val routes", len(train_enc), len(val_enc))

    model = PointerNet(cfg)
    log.info("pointer net has %d trainable parameters", count_parameters(model))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    def loader(data, shuffle):
        return DataLoader(data, batch_size=cfg.batch_size, shuffle=shuffle,
                          collate_fn=collate)

    history: list[dict] = []
    best_val = float("inf")
    best_path = out_dir / (f"pointer_net_{tag}.pt" if tag else "pointer_net.pt")

    for epoch in range(cfg.epochs):
        model.train()
        tf_p = scheduled_sampling_p(epoch, cfg)
        losses = []
        t0 = time.perf_counter()
        for batch in loader(train_enc, True):
            opt.zero_grad()
            loss = model.loss(batch, teacher_forcing_p=tf_p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        model.eval()
        val_losses, val_top1, val_uniform = [], [], []
        with torch.no_grad():
            for batch in loader(val_enc, False) if val_enc else []:
                val_losses.append(float(model.loss(batch, teacher_forcing_p=1.0).item()))
                val_top1.append(_top1_accuracy(model, batch))
                val_uniform.append(_uniform_loss(batch))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        top1 = float(np.mean(val_top1)) if val_top1 else float("nan")
        unif = float(np.mean(val_uniform)) if val_uniform else float("nan")
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_top1": top1, "uniform_loss": unif,
            "teacher_forcing_p": tf_p, "seconds": time.perf_counter() - t0,
        })
        log.info(
            "epoch %02d | train %.4f | val %.4f (uniform %.4f) | top1 %.3f "
            "| tf_p %.2f | %.1fs",
            epoch, train_loss, val_loss, unif, top1, tf_p,
            history[-1]["seconds"],
        )
        score = val_loss if np.isfinite(val_loss) else train_loss
        if score < best_val:
            best_val = score
            model.save(best_path)

    if not best_path.exists():          # no epoch improved (epochs == 0)
        model.save(best_path)
    (out_dir / f"pointer_history_{tag or 'all'}.json").write_text(
        json.dumps({"history": history, "best_val": best_val,
                    "cfg": cfg.__dict__}, indent=2)
    )
    return {"best_val_loss": best_val, "epochs": len(history), "path": str(best_path)}


# ---------------------------------------------------------------------- GBM
def train_gbm(train_instances, out_dir: Path, tag: str, seed: int = 20260101) -> dict:
    usable = [i for i in train_instances if i.actual_sequence is not None]
    log.info("GBM transition model: %d routes", len(usable))
    model = GBMTransitionModel.fit(usable, seed=seed)
    path = out_dir / (f"gbm_transition_{tag}.pkl" if tag else "gbm_transition.pkl")
    model.save(path)
    (out_dir / f"gbm_importances_{tag or 'all'}.json").write_text(
        json.dumps(model.importances, indent=2)
    )
    return {"path": str(path), "trained": model.booster is not None,
            "importances": model.importances}


# ---------------------------------------------------------------------- driver
def run_fold(metro: str, args, cfg: Config) -> dict:
    tag = _fold_tag(metro)
    train, val = model_split(args.processed, metro, high_quality_only=not args.all_quality)
    train_i = train.instances(args.limit)
    val_i = val.instances(args.limit_val)
    out = Path(cfg.paths.artifacts)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"held_out_metro": metro, "n_train": len(train_i), "n_val": len(val_i)}
    if args.model in ("pointer", "both"):
        summary["pointer"] = train_pointer(train_i, val_i, cfg.behavior, out, tag)
    if args.model in ("gbm", "both"):
        summary["gbm"] = train_gbm(train_i, out, tag, cfg.behavior.seed)
    (out / f"train_behavior_summary_{tag or 'all'}.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--held-out-metro", default="Austin", choices=list(METROS))
    ap.add_argument("--all-folds", action="store_true",
                    help="train one model per leave-one-metro-out fold")
    ap.add_argument("--model", default="both", choices=["pointer", "gbm", "both"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-stops", type=int, default=250)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--all-quality", action="store_true",
                    help="do not restrict sigma* to High-quality routes")
    ap.add_argument("--seed", type=int, default=20260101)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.paths = Paths().ensure()
    cfg.behavior = BehaviorConfig(
        model=args.model if args.model != "both" else "pointer",
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        n_layers=args.layers, lr=args.lr, batch_size=args.batch_size,
        epochs=args.epochs, max_stops=args.max_stops, seed=args.seed,
    )

    metros = list(METROS) if args.all_folds else [args.held_out_metro]
    for m in metros:
        log.info("=== fold: held out %s ===", m)
        print(json.dumps(run_fold(m, args, cfg), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
