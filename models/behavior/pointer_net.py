"""Stage-1a primary model: a pointer network trained by behavioural cloning.

Architecture
------------
* **Encoder** -- per-stop embedding of ``[zone rank (embedding), within-zone
  lat/long offset, package count, package volume, time-window width, window
  start, depot travel times]`` followed by ``n_layers`` transformer blocks with
  a padding mask (permutation-equivariant, so nothing leaks from the input
  ordering of the stops).
* **Decoder** -- a GRU cell carrying the visit state, with additive (Bahdanau)
  attention over the remaining stops.  Training is cross-entropy against
  sigma*, teacher forcing for the first ``teacher_forcing_epochs`` epochs and
  then a linear scheduled-sampling curriculum.

Why the attention weights are read out directly
-----------------------------------------------
The decoder's attention distribution at the step where stop ``i`` is current
*is* ``p_ij``.  :meth:`PointerNet.transition_matrix` returns the full dense
matrix so it can be dropped into the MILP as ``c^beh_ij = -log p_ij``.  Two
distributions are produced per step:

* the **masked** one (over unvisited stops only) -- used for imitation metrics
  and greedy decoding, exactly what an imitation-learning baseline would use;
* the **unmasked** one (over all stops but the current one) -- used for the cost
  matrix, because the optimizer needs a cost on every arc, including arcs the
  greedy rollout never considered.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import BehaviorConfig
from core.types import DEPOT, Instance
from models.behavior.dataset import (
    MAX_ZONES,
    N_STOP_FEATURES,
    EncodedRoute,
    encode_instance,
)

# Per-candidate features that depend on the current stop and the visit state:
#   normalised travel time, log of it, same-zone flag, candidate's zone already
#   visited, candidate's zone already visited but left, rank of the travel time
#   among the remaining candidates.
N_ARC_FEATURES = 6


class _Attention(nn.Module):
    """Additive attention producing pointer logits over the encoder memory.

    The ``clip * tanh(.)`` logit clipping is the standard pointer-network device
    (Bello et al. 2017) and it keeps ``-log p_ij`` bounded, which the MILP wants.
    It does have a failure mode worth knowing: with too large a learning rate the
    pre-activations saturate, every candidate clips to the same value and the
    policy collapses to uniform with zero gradient.  Keep ``lr`` at or below
    ~5e-3 for this architecture; the default 1e-3 is well inside the safe range.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, 1, bias=False)
        self.clip = 10.0

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # query (B, D); memory (B, N, D) -> logits (B, N)
        h = torch.tanh(self.w_k(memory) + self.w_q(query).unsqueeze(1))
        return self.clip * torch.tanh(self.v(h).squeeze(-1))


class PointerNet(nn.Module):
    def __init__(self, cfg: BehaviorConfig | None = None):
        super().__init__()
        self.cfg = cfg or BehaviorConfig()
        d = self.cfg.embed_dim
        h = self.cfg.hidden_dim

        self.zone_emb = nn.Embedding(MAX_ZONES + 1, d)
        self.feat_proj = nn.Linear(N_STOP_FEATURES, d)
        self.in_norm = nn.LayerNorm(d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=max(1, d // 16),
            dim_feedforward=4 * d,
            dropout=self.cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=self.cfg.n_layers, enable_nested_tensor=False
        )
        self.mem_proj = nn.Linear(d, h)
        self.cell = nn.GRUCell(h, h)
        self.attn = _Attention(h)
        self.start = nn.Parameter(torch.zeros(h))
        # Arc-feature head: a small MLP on per-candidate features that depend on
        # the *current* stop.  Without it the decoder has no way to know how far
        # away a candidate is -- the encoder is per-stop and permutation
        # equivariant, so proximity to the current stop is simply absent from
        # its representation.  Trained from scratch this produced a policy stuck
        # at exactly the uniform-distribution loss on real routes; the GBM's
        # three strongest features are all of this kind.
        self.arc_mlp = nn.Sequential(
            nn.Linear(N_ARC_FEATURES, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.arc_scale = nn.Parameter(torch.tensor(1.0))

    # ---------------------------------------------------------------- encoding
    def encode(self, features: torch.Tensor, zone_rank: torch.Tensor,
               pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(self.feat_proj(features) + self.zone_emb(zone_rank))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.mem_proj(x)                                  # (B, N, H)

    # ------------------------------------------------------------ arc features
    def _arc_bias(
        self,
        travel: torch.Tensor,        # (B, N, N) median-normalised
        zone_of: torch.Tensor,       # (B, N)
        zone_seen: torch.Tensor,     # (B, N) zone already visited
        current: torch.Tensor,       # (B,)
        visited: torch.Tensor,       # (B, N)
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Per-candidate logit bias from features relative to the current stop.

        These are the signals a per-stop encoder cannot carry: how far each
        candidate is from where the vehicle *is*, whether it shares the current
        zone, and whether its zone has already been worked.  The rank feature
        matters because the choice is a comparison among the remaining stops,
        not an absolute judgement.
        """
        tt = travel[batch_idx, current]                       # (B, N)
        cur_zone = zone_of[batch_idx, current].unsqueeze(1)    # (B, 1)
        same_zone = (zone_of == cur_zone).float()

        # Rank of each candidate's travel time among the *unvisited* ones.
        big = tt.max().detach() + 1.0
        masked_tt = torch.where(visited, torch.full_like(tt, float(big)), tt)
        order = masked_tt.argsort(dim=1)
        rank = torch.zeros_like(tt)
        rank.scatter_(
            1, order,
            torch.arange(tt.shape[1], device=tt.device, dtype=tt.dtype)
            .unsqueeze(0).expand_as(tt),
        )
        n_left = (~visited).sum(dim=1, keepdim=True).clamp(min=1)
        rank = rank / n_left.to(tt.dtype)

        feats = torch.stack(
            [
                tt,
                torch.log1p(tt),
                same_zone,
                zone_seen.float(),
                zone_seen.float() * (1.0 - same_zone),
                rank,
            ],
            dim=-1,
        )                                                      # (B, N, F)
        return self.arc_scale * self.arc_mlp(feats).squeeze(-1)

    # ---------------------------------------------------------------- decoding
    def forward(
        self,
        features: torch.Tensor,
        zone_rank: torch.Tensor,
        pad_mask: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_p: float = 1.0,
        return_unmasked: bool = False,
        travel: torch.Tensor | None = None,
        zone_of: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Roll the decoder out for ``N-1`` steps.

        ``logits[b, t, j]`` scores visiting ``j`` at step ``t+1`` given the stop
        occupied at step ``t``.  Step 0 always starts at the depot.
        """
        memory = self.encode(features, zone_rank, pad_mask)
        b, n, _ = memory.shape
        device = memory.device

        visited = pad_mask.clone()                     # padding counts as visited
        visited[:, DEPOT] = True
        current = torch.full((b,), DEPOT, dtype=torch.long, device=device)
        state = self.start.unsqueeze(0).expand(b, -1).contiguous()

        if travel is None:
            travel = torch.zeros(b, n, n, device=device)
        if zone_of is None:
            zone_of = torch.zeros(b, n, dtype=torch.long, device=device)
        zone_seen = torch.zeros(b, n, dtype=torch.bool, device=device)

        logits_all, unmasked_all, picks, tgt_ok = [], [], [], []
        batch_idx = torch.arange(b, device=device)

        for t in range(n - 1):
            state = self.cell(memory[batch_idx, current], state)
            raw = self.attn(state, memory)                        # (B, N)
            raw = raw + self._arc_bias(
                travel, zone_of, zone_seen, current, visited, batch_idx
            )
            if return_unmasked:
                self_mask = torch.zeros_like(raw, dtype=torch.bool)
                self_mask[batch_idx, current] = True
                unmasked_all.append(
                    raw.masked_fill(self_mask | pad_mask, -1e9)
                )
            masked = raw.masked_fill(visited, -1e9)
            logits_all.append(masked)

            greedy = masked.argmax(dim=-1)
            if target is not None:
                tgt_step = target[:, t + 1].clamp(min=0)
                # Under scheduled sampling the rollout can diverge from sigma*,
                # after which the teacher's next stop may already be visited.
                # Its logit is then -1e9 and the cross-entropy against it blows
                # up (observed: a train loss of 1.98e8 in the first
                # scheduled-sampling epoch).  The teacher signal is simply
                # undefined at those steps, so they are masked out of the loss.
                tgt_ok.append(~visited[batch_idx, tgt_step])
                use_tgt = torch.rand(b, device=device) < teacher_forcing_p
                nxt = torch.where(use_tgt, tgt_step, greedy)
                # Never step onto a padded/visited slot (short routes in a batch).
                bad = visited[batch_idx, nxt]
                nxt = torch.where(bad, greedy, nxt)
            else:
                nxt = greedy
            picks.append(nxt)
            # Clone before marking: ``visited`` is captured by this step's
            # masked_fill, and mutating it in place would corrupt the backward
            # pass (autograd version counter).
            visited = visited.clone()
            visited[batch_idx, nxt] = True
            # Mark every stop sharing the chosen stop's zone as "zone seen".
            zone_seen = zone_seen | (
                zone_of == zone_of[batch_idx, nxt].unsqueeze(1)
            )
            current = nxt

        out = {
            "logits": torch.stack(logits_all, dim=1),             # (B, N-1, N)
            "picks": torch.stack(picks, dim=1),                   # (B, N-1)
        }
        if tgt_ok:
            out["target_reachable"] = torch.stack(tgt_ok, dim=1)  # (B, N-1)
        if return_unmasked:
            out["unmasked_logits"] = torch.stack(unmasked_all, dim=1)
        return out

    # ------------------------------------------------------------------- loss
    def loss(self, batch: dict, teacher_forcing_p: float = 1.0) -> torch.Tensor:
        out = self.forward(
            batch["features"], batch["zone_rank"], batch["pad_mask"],
            target=batch["target"], teacher_forcing_p=teacher_forcing_p,
            travel=batch.get("travel"), zone_of=batch.get("zone_of"),
        )
        logits = out["logits"]                                     # (B, N-1, N)
        tgt = batch["target"][:, 1:]                               # (B, N-1)
        lengths = batch["lengths"]
        valid = torch.arange(tgt.shape[1], device=tgt.device)[None, :] < (
            lengths[:, None] - 1
        )
        reachable = out.get("target_reachable")
        if reachable is not None:
            valid = valid & reachable
        tgt = tgt.masked_fill(~valid, -100)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100
        )

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def greedy_order(self, enc: EncodedRoute) -> list[int]:
        self.eval()
        batch = _single(enc, self.device)
        out = self.forward(
            batch["features"], batch["zone_rank"], batch["pad_mask"],
            travel=batch.get("travel"), zone_of=batch.get("zone_of"),
        )
        return [DEPOT] + out["picks"][0].tolist()

    @torch.no_grad()
    def transition_matrix(
        self, inst: Instance, prob_floor: float | None = None
    ) -> np.ndarray:
        """Dense ``p_ij`` read off the decoder's attention.

        Row ``i`` is the unmasked attention distribution at the rollout step
        where ``i`` was the current stop.  Rows for stops the rollout never
        occupies (only the final stop can be such) fall back to the row's
        marginal over the encoder memory, i.e. a uniform distribution -- the
        least-committal choice.
        """
        self.eval()
        floor = self.cfg.prob_floor if prob_floor is None else prob_floor
        enc = encode_instance(inst)
        batch = _single(enc, self.device)
        out = self.forward(
            batch["features"], batch["zone_rank"], batch["pad_mask"],
            return_unmasked=True,
            travel=batch.get("travel"), zone_of=batch.get("zone_of"),
        )
        unmasked = out["unmasked_logits"][0]                       # (N-1, N)
        picks = out["picks"][0].tolist()
        path = [DEPOT] + picks

        n = inst.n
        p = np.full((n, n), 1.0 / max(n - 1, 1), dtype=float)
        probs = torch.softmax(unmasked, dim=-1).cpu().numpy()
        for t, cur in enumerate(path[:-1]):
            p[cur] = probs[t]
        np.fill_diagonal(p, 0.0)
        p[:, DEPOT] = 0.0                     # the decoder never returns to depot
        row = p.sum(axis=1, keepdims=True)
        p = np.divide(p, np.where(row > 0, row, 1.0))
        return np.maximum(p, floor)

    def behavior_cost_matrix(self, inst: Instance) -> np.ndarray:
        return -np.log(self.transition_matrix(inst))

    # --------------------------------------------------------------------- io
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.__dict__, "state_dict": self.state_dict()}, path)
        return path

    @staticmethod
    def load(path: str | Path, map_location: str = "cpu") -> "PointerNet":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        cfg = BehaviorConfig(**blob["cfg"])
        model = PointerNet(cfg)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


def _single(enc: EncodedRoute, device: torch.device) -> dict:
    feats = torch.from_numpy(enc.features).unsqueeze(0).to(device)
    zr = torch.from_numpy(enc.zone_rank).unsqueeze(0).to(device)
    pad = torch.zeros(1, enc.n, dtype=torch.bool, device=device)
    tgt = (
        torch.from_numpy(enc.target).unsqueeze(0).to(device)
        if enc.target is not None
        else None
    )
    travel = (
        torch.from_numpy(enc.travel_norm).unsqueeze(0).to(device)
        if enc.travel_norm is not None
        else None
    )
    return {
        "features": feats,
        "zone_rank": zr,
        "zone_of": torch.from_numpy(enc.zone_of).unsqueeze(0).to(device),
        "pad_mask": pad,
        "target": tgt,
        "travel": travel,
        "lengths": torch.tensor([enc.n], device=device),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scheduled_sampling_p(epoch: int, cfg: BehaviorConfig) -> float:
    """1.0 while teacher forcing, then a linear decay to the configured floor."""
    if epoch < cfg.teacher_forcing_epochs:
        return 1.0
    span = max(cfg.epochs - cfg.teacher_forcing_epochs, 1)
    frac = (epoch - cfg.teacher_forcing_epochs) / span
    return float(1.0 - frac * (1.0 - cfg.scheduled_sampling_final_p))


def positional_sanity(n: int) -> float:  # small helper used by tests
    return math.log(max(n - 1, 1))
