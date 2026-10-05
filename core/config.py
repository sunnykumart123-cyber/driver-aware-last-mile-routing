"""Configuration objects. Every experiment is fully described by a ``Config``."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

METROS = ("Los Angeles", "Seattle", "Chicago", "Boston", "Austin")
METRO_SLUGS = {
    "Los Angeles": "los_angeles",
    "Seattle": "seattle",
    "Chicago": "chicago",
    "Boston": "boston",
    "Austin": "austin",
}


@dataclass
class Paths:
    root: Path = REPO_ROOT
    raw: Path = REPO_ROOT / "data" / "raw"
    processed: Path = REPO_ROOT / "data" / "processed"
    artifacts: Path = REPO_ROOT / "results" / "artifacts"
    results: Path = REPO_ROOT / "results"

    @property
    def experiment_log(self) -> Path:
        return Path(self.results) / "experiment_log.parquet"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.artifacts, self.results):
            Path(p).mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class UncertaintyConfig:
    """Stage-1b quantile model plus the ambiguity set it feeds."""

    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    n_scenarios: int = 64
    eps: float = 0.10                 # violation tolerance
    delta: float = 0.0                # Wasserstein radius (0 == SAA, no DRO)
    ground_norm: str = "linf"         # linf | l1 | l2  -> dual l1 | linf | l2
    geometry: str = "wasserstein"     # wasserstein | box | ellipsoidal
    seed: int = 20260101
    # RQ3 degradation hooks
    degrade_features: bool = False
    degrade_quantiles: bool = False
    degrade_train_fraction: float = 1.0

    def effective_quantiles(self) -> tuple[float, ...]:
        if self.degrade_quantiles:
            return (0.25, 0.5, 0.75)
        return tuple(self.quantiles)


@dataclass
class BehaviorConfig:
    """Stage-1a behaviour model."""

    model: str = "pointer"            # pointer | gbm | uniform
    embed_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 20
    teacher_forcing_epochs: int = 8   # then scheduled sampling
    scheduled_sampling_final_p: float = 0.5
    grad_clip: float = 2.0
    max_stops: int = 250
    seed: int = 20260101
    prob_floor: float = 1e-6          # keeps -log p finite


@dataclass
class SolverConfig:
    name: str = "alns"                # alns | exact | benders | greedy
    time_limit_s: float = 60.0
    alns_iterations: int = 4000
    alns_segment: int = 100
    sa_start_temp_frac: float = 0.05
    sa_cooling: float = 0.9995
    destroy_min_frac: float = 0.05
    destroy_max_frac: float = 0.35
    # Absolute cap on one destroy step.  The reliability term couples the whole
    # route, so a large removal makes the repair scan quadratically expensive;
    # Ropke & Pisinger cap at 100 for plain VRP, a tighter cap pays here.
    destroy_max_stops: int = 25
    # Insertions applied per scored scan in the repair operators (1 == textbook
    # full rescan after every insertion).  See solvers/alns/repair_ops.py.
    repair_refresh_every: int = 8
    regret_k: int = 3
    # Candidate positions scored exactly per insertion; 0 == all legal ones.
    # Only binds when zone precedence is off, where every position is legal.
    candidate_cap: int = 24
    # OFF by default on the strength of the data: only a minority of real
    # driver sequences visit each sort zone in one unbroken block, so imposing
    # contiguity excludes sigma* itself on most routes and puts a floor on the
    # attainable ERP.  See scripts/data_diagnostics.py and docs/DATA_NOTES.md.
    # Set True to run it as the §3 ablation.
    zone_precedence: bool = False
    mip_gap: float = 1e-4
    exact_backend: str = "auto"       # auto | gurobi | pulp | bruteforce
    seed: int = 20260101


@dataclass
class Config:
    lam: float = 0.5                  # lambda in the combined objective
    theta: float = 1.0                # soft-lateness penalty (base MILP only)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    paths: Paths = field(default_factory=Paths)
    held_out_metro: str | None = None
    tag: str = "default"

    # -------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        d["uncertainty"]["quantiles"] = list(self.uncertainty.quantiles)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        sub = {
            "behavior": BehaviorConfig,
            "uncertainty": UncertaintyConfig,
            "solver": SolverConfig,
        }
        kwargs: dict = {}
        for f in fields(cls):
            if f.name not in raw:
                continue
            if f.name in sub:
                cl = sub[f.name]
                known = {g.name for g in fields(cl)}
                kwargs[f.name] = cl(**{k: v for k, v in raw[f.name].items() if k in known})
            elif f.name == "paths":
                kwargs["paths"] = Paths(**{k: Path(v) for k, v in raw["paths"].items()})
            else:
                kwargs[f.name] = raw[f.name]
        cfg = cls(**kwargs)
        cfg.uncertainty.quantiles = tuple(cfg.uncertainty.quantiles)
        return cfg
