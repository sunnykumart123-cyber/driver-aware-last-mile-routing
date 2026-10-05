# Formulation as implemented

This is the paper-side derivation that `solvers/` implements, written so that
every constant in the code can be traced back to a line here. Notation follows
§3 of the build spec.

## 1. Sets and data

| Symbol | Meaning | Code |
| --- | --- | --- |
| $N=\{0,\dots,n-1\}$ | stops, $0$ = station | `Instance.stops` |
| $A$ | arcs $(i,j)$, $i\neq j$ | implicit |
| $z(i)$ | sort zone of stop $i$ | `Instance.zones` |
| $\sigma^\*$ | observed driver sequence | `Instance.actual_sequence` |
| $t_{ij},\,s_i$ | travel / service time, random | `AmbiguitySet.travel/.service` |
| $[a_i,b_i]$ | delivery window | `Stop.tw_start/.tw_end` |
| $p_{ij}$ | Stage-1a transition probability | `*.transition_matrix` |
| $\lambda$ | behaviour-vs-reliability weight | `Config.lam` |
| $\varepsilon$ | violation tolerance | `UncertaintyConfig.eps` |
| $\delta$ | Wasserstein radius | `UncertaintyConfig.delta` |

## 2. Base MILP (deterministic, soft windows)

Implemented in `solvers/exact_milp.py::solve_base_milp`, built and unit-tested
before the reliability layer exists.

$$\min \sum_{(i,j)\in A} c_{ij}x_{ij} + \theta\sum_{i\in N}\ell_i$$

subject to

$$\sum_{j\neq i} x_{ij}=1,\qquad \sum_{j\neq i} x_{ji}=1 \qquad \forall i\in N$$
$$T_j \ge T_i + s_i + t_{ij} - M(1-x_{ij}) \qquad \forall (i,j)\in A,\ j\neq 0$$
$$\ell_i \ge T_i - b_i,\qquad \ell_i\ge 0 \qquad \forall i\in N$$
$$u_i - u_j + n\,x_{ij}\le n-1 \qquad \forall (i,j)\in A,\ i,j\neq 0$$
$$T_0 = 0,\quad u_0 = 0,\quad u_i \in \{1,\dots,n-1\}$$

**Why $j\neq 0$ in the propagation block.** With a closed tour, applying the
propagation constraint to the arc $(\text{last},0)$ together with $T_0=0$ makes
the model infeasible. Arrival back at the station is not constrained by a window,
so the head-at-depot copies are simply dropped. `solve_base_milp` and both DRO
backends all do this.

## 3. Zone precedence

The LMRRC sort zones are worked as blocks: a route enters a zone, finishes it, and
moves on. That is a *contiguity* requirement on positions, not a precedence
relation between fixed pairs, so it is encoded with zone-order binaries. For
zones $Z$ with sizes $m_z$:

$$y_{zz'} + y_{z'z} = 1,\qquad y_{zz'}+y_{z'z''}-y_{zz''}\le 1$$
$$o_z = \sum_{z'\neq z} m_{z'}\,y_{z'z},\qquad o_z+1 \le u_i \le o_z+m_z\quad \forall i: z(i)=z$$

$y_{z'z}=1$ means $z'$ precedes $z$; the transitivity triples make $y$ a total
order, so $o_z$ counts the stops in all preceding zones and every zone occupies
exactly the position block $[o_z+1,\,o_z+m_z]$. `core.types.zone_contiguous`
checks the same property on a sequence, and `RouteState.feasible_positions`
maintains it as a search invariant.

## 4. Behaviour cost

$$c^{\mathrm{beh}}_{ij} = -\log p_{ij}$$

with $p_{ij}$ floored at `BehaviorConfig.prob_floor` and the cost clipped at 30
(`models/behavior/cost.py`) so that one near-zero probability cannot swamp the
objective. The closing arc into the depot carries no behaviour cost: the
Stage-1a decoder emits no depot-return probability, so the sum runs over the
$n-1$ delivery arcs only.

## 5. Reliability layer

### 5.1 From chance constraint to CVaR

The spec's requirement is

$$\inf_{\mathbb P\in\mathfrak B_\delta(\hat P)}\ \Pr_{\mathbb P}[T_i\le b_i]\ \ge\ 1-\varepsilon .$$

The chance constraint is non-convex in general, so it is enforced through its
standard CVaR safeguard,

$$\mathrm{CVaR}_{1-\varepsilon}[T_i-b_i]\ \le\ \eta_i,\qquad \eta_i\ge 0 .$$

Because $\mathrm{VaR}_{1-\varepsilon}\le \mathrm{CVaR}_{1-\varepsilon}$, any solution with
$\eta_i=0$ satisfies the chance constraint. The converse does not hold, so the
model is *conservative*, and $\eta_i>0$ is a continuous measure of how far a stop
is from the guarantee — which is what makes it usable as an objective term rather
than only as a constraint.

### 5.2 Arrival time is linear in the primitives

For a fixed sequence, the arrival at the stop in position $p$ is

$$T_p(\xi) = \langle w_p, \xi\rangle,\qquad w_p\in\{0,1\}^d,\quad \|w_p\|_0 = 2p-1,$$

summing the $p$ travel times and $p-1$ service times on the prefix. `ReliabilityEvaluator.arrivals`
computes all of these with one `cumsum` over the scenario axis.

### 5.3 Worst-case CVaR over a Wasserstein ball

For a Lipschitz loss and a type-1 Wasserstein ball of radius $\delta$ around
$\hat P$ (Mohajerin Esfahani & Kuhn 2018, Thm. 6.3; the reformulation route used
by Zhang et al. 2021),

$$\sup_{\mathbb P\in\mathfrak B_\delta(\hat P)} \mathrm{CVaR}_{1-\varepsilon}\big[\langle w,\xi\rangle-b\big]
= \mathrm{CVaR}^{\hat P}_{1-\varepsilon}\big[\langle w,\xi\rangle-b\big] + \frac{\delta}{\varepsilon}\,\|w\|_{*} .$$

Sketch: $\mathrm{CVaR}_{1-\varepsilon}[\,\cdot\,] = \min_\tau \tau + \tfrac1\varepsilon \mathbb E[(\cdot-\tau)_+]$;
the inner map $\xi\mapsto \tfrac1\varepsilon(\langle w,\xi\rangle-b-\tau)_+$ is Lipschitz with
modulus $\tfrac1\varepsilon\|w\|_*$ in the ground norm; the worst-case expectation of an
$L$-Lipschitz function over a Wasserstein-$\delta$ ball exceeds its nominal value by at
most $L\delta$, and the bound is attained by transporting mass along the gradient
direction. Hence the inflation $\tfrac{\delta}{\varepsilon}\|w\|_*$, exactly.

Dual norms, for $w$ a 0/1 vector with $2p-1$ ones (`AmbiguitySet.dual_norm_weight`):

| ground norm on $\xi$ | dual | $\|w_p\|_*$ |
| --- | --- | --- |
| $\ell_\infty$ | $\ell_1$ | $2p-1$ |
| $\ell_1$ | $\ell_\infty$ | $1$ |
| $\ell_2$ | $\ell_2$ | $\sqrt{2p-1}$ |

The $\ell_\infty$ ground norm is the default: it says "every leg of the route may
deviate by up to the transport budget", which grows the penalty with route depth
and is what makes long fragile prefixes expensive.

### 5.4 Sample CVaR in closed form

$\hat P$ is supported on $S$ scenarios, so

$$\mathrm{CVaR}^{\hat P}_{1-\varepsilon}[L] = \min_\tau\ \tau + \frac{1}{\varepsilon S}\sum_{s}(L_s-\tau)_+
= \mathrm{VaR} + \frac{1}{\varepsilon S}\sum_s (L_s-\mathrm{VaR})_+,$$

with $\mathrm{VaR} = L_{(\lceil (1-\varepsilon)S\rceil)}$ the order statistic. This is
`solvers.reliability.sample_cvar`: one `partition` and one masked sum, no LP.
That closed form is what lets the ALNS evaluate a candidate insertion without
solving anything, which §5b of the spec requires.

### 5.5 Scenario-based MILP block

For the exact solvers the same quantity is written with explicit variables, one
arrival copy per scenario:

$$\nu_i^s \ge T_i^s - b_i - \tau_i,\qquad \nu_i^s\ge 0$$
$$\eta_i \ge \tau_i + \frac{1}{\varepsilon S}\sum_{s}\nu_i^s + \frac{\delta}{\varepsilon}\big(2u_i-1\big),\qquad \eta_i\ge 0$$

The inflation is linear in the MTZ position variable $u_i$ under the $\ell_\infty$
ground norm, and constant under $\ell_1$. Under $\ell_2$ it is $\sqrt{2u_i-1}$,
which is not MILP-representable; `solve_exact` raises rather than silently
approximating it, while the heuristics support all three.

## 6. Combined objective

$$\min_{x,T,\eta,u}\quad \frac{\lambda}{n-1}\sum_{(i,j)\in A} c^{\mathrm{beh}}_{ij}x_{ij}
\;+\; \frac{1-\lambda}{n-1}\sum_{i\in N}\eta_i$$

subject to assignment, MTZ, zone precedence, per-scenario arrival propagation and
the reliability block of §5.5.

$\lambda=1$ is the pure imitation baseline, $\lambda=0$ the uncertainty-aware
robust baseline with no behaviour information, and the sweep in between is the
RQ2 experiment.

**Scaling remark.** Both terms are divided by $n-1$. This is a positive
rescaling of each term separately, so it does not change the optimum at
$\lambda\in\{0,1\}$ and does not change the feasible set anywhere; it makes a
single $\lambda$ grid comparable across routes of 60 and 240 stops, which matters
because the experiment pools routes of very different length. Reported behaviour
and risk costs are therefore per-arc and per-stop averages.

## 7. Logic-based Benders cut

Master: sequencing with the behaviour cost and a single $\zeta\ge 0$ standing in
for $\frac{1-\lambda}{n-1}\sum_i\eta_i$. Subproblem: evaluate the DRO layer on
the master's sequence.

Let $p$ be the earliest position with $\eta_p>0$ and $A_p$ the arcs of the
depot-rooted prefix ending there. The arrival distribution at $p$ is a function
of exactly those arcs, so any feasible solution containing all of $A_p$ has
prefix identical to this one and surplus at least $\eta_p$ at that stop:

$$\zeta \ \ge\ \frac{1-\lambda}{n-1}\,\eta_p\Big(1-\sum_{(i,j)\in A_p}(1-x_{ij})\Big).$$

If the route contains every arc of $A_p$ the bracket is 1 and the bound binds;
otherwise it is $\le 0$ and the cut is vacuous. Validity therefore needs no
Lipschitz or monotonicity assumption. The cut is much stronger than a no-good
cut on the whole tour because $|A_p|$ is typically a short prefix.

The master objective is a valid lower bound at every iteration and the best
evaluated sequence a valid upper bound, so stopping early still certifies a gap.

## 8. What is *not* claimed

* The CVaR safeguard is sufficient, not necessary, for the chance constraint: a
  route with $\eta_i>0$ may still satisfy $\Pr[T_i\le b_i]\ge 1-\varepsilon$.
* $\hat P$ treats the per-arc and per-stop primitives as independent draws from
  their conditional laws. Dependence across legs of the same route (a driver
  having a bad day) is *not* modelled; the Wasserstein radius is the only
  protection against it, and §6 of `docs/DATA_NOTES.md` explains why the data
  cannot identify that correlation.
* The reliability index reported in results is a Monte-Carlo estimate on an
  independent scenario sample, not a proof about the true law.
