# route-choice-due-lcp

Route-choice dynamic user equilibrium (DUE) traffic assignment on point-queue
networks, solved through its linear complementarity formulation. The Python
package is imported as `rcdue`.

The package is a reference implementation of the time-decomposed algorithm of

> M. Wakui, T. Sakai and T. Akamatsu (2023). Efficient algorithm for solving
> dynamic user equilibrium traffic assignment. *Journal of JSCE*, Vol. 79,
> No. 4, 22-00301 (in Japanese, English abstract).

for one-to-many demand (one origin, several destinations), together with

* an **independent checker** that evaluates any solution against the discrete
  equilibrium conditions *and* the physical-consistency conditions that the
  paper's Step 5 / ZD algorithm are meant to guarantee, and
* a **UXsim bridge** for like-for-like comparisons between the LCP solution and
  simulation-based assignment (UXsim DUO and UXsim's day-to-day DUE solver).

A `many_to_one` subpackage (one destination, several origins) is planned next
to `one_to_many`; the network container and the checker are written so that it
can share them.

## Installation

```bash
pip install -e .                 # core: numpy, scipy (HiGHS LP backend); import as rcdue
pip install -e ".[uxsim,plot]"   # + UXsim comparison and figures
pip install -e ".[gurobi]"       # + Gurobi backend (licence required)
pip install -e ".[dev]"          # + pytest, ruff, matplotlib, jupyter
```

The LP sub-problems are solved with HiGHS (via `scipy.optimize.linprog`) by
default. If `gurobipy` is importable and licensed, `backend="auto"` picks
Gurobi, which keeps persistent models across departure steps and is faster on
large networks. Results are identical up to solver tolerance.

## Quick start

```python
from rcdue.networks import wakui_5node
from rcdue.one_to_many import solve_sequential, check_sequential_result

net = wakui_5node()                       # paper's 5-node example
res = solve_sequential(net, eps=1e-8)     # PD / ZD algorithm, backend="auto"
print(check_sequential_result(net, res).summary())
```

`res.w`, `res.y`, `res.pi` hold queue delay, link inflow rate and
origin-to-node travel time, each indexed by link (or node) and **departure
step**. Step `k` (0-based) refers to users leaving the origin at clock time
`(k + 1) * dt`, the end of demand bin `[k dt, (k + 1) dt)`.

Build your own network with `NetworkData` and `Link`:

```python
import numpy as np
from rcdue import Link, NetworkData

q = np.zeros(40); q[:8] = 20.0            # departure rate per step toward node 3
net = NetworkData(
    nodes=[0, 1, 2, 3], origin=0, destinations=[3],
    links=[Link(0, 1, tau_hat=2.0, mu_hat=10.0),
           Link(1, 2, tau_hat=2.0, mu_hat=5.0),
           Link(2, 3, tau_hat=2.0, mu_hat=100.0)],
    dt=1.0, T=40, demand={3: q},
)
```

## What is implemented

`rcdue.one_to_many.sequential` follows Section 4 of the paper step by step:

| Paper | Code |
|---|---|
| Step 0, `[DUE-FW-init]` (L1-closest feasible point) | `StepLPs.solve_init` |
| Step 1, `[DUE-FW-LP]` with constraints (19), (20) | `StepLPs.solve_fw` |
| Step 2, exact line search, eqs (24)-(25) | `fw_solver.line_search` |
| Steps 3-4, convex combination and `z(x) < eps` | `_frank_wolfe` |
| Step 5, reconciliation (Appendix III) | `_dijkstra_rewrite(base_shift=0)` |
| ZD algorithm (queue decay `ds - d_v`) | `_dijkstra_rewrite(base_shift=ds)` |

`rcdue.one_to_many.verify.check_solution(net, w, y, pi, pi_nodes)` re-implements
the residuals and the shortest-path search independently of the solver and
reports, per condition, the worst violation over all steps:

* equilibrium: non-negativity, `R_w >= 0`, `R_y >= 0`, conservation,
  `w . R_w = 0`, `y . R_y = 0`, FIFO bound, free-flow bound;
* physical consistency: `pi` equals the shortest travel time under `c0 + w`
  at every node, and no flow / no queue growth on zero-demand steps.

`excess_travel_time` gives a scalar DUE gap: the mean travel time in excess
of the shortest-path time per user (0 at an exact DUE).

A bulk solver (`lcp_builder` + `fw_solver`) that handles all departure steps in
one LCP is kept as a reference for small networks. It is not part of the paper
and does not perform Step 5, so it may return equilibria whose off-path `pi`
are not physically consistent.

## Examples (notebooks)

* `examples/01_sequential_5node.ipynb`: solve the paper's 5-node example, verify it,
  and draw the paper's figures (cumulative curves at the bottlenecks, route-choice
  equilibrium per OD pair).
* `examples/02_uxsim_fair_comparison.ipynb`: like-for-like comparison with UXsim
  DUO and UXsim's day-to-day DUE (requires `pip install -e ".[uxsim,plot]"`).

The notebooks are committed with their outputs; figure helpers live in
`rcdue.plotting`.

## Comparing with UXsim

The bridge (`rcdue.uxsim_bridge`) keeps the comparison fair in four ways:

1. **Same input.** Demand is quantized to multiples of UXsim's platoon size
   and the quantized network is given to both solvers.
2. **Same physics.** UXsim is configured as a point queue (huge jam density,
   unthrottled entry, one lane, explicit `capacity_out`, dummy sink links so
   the last real link still enforces its capacity).
3. **Same variables.** UXsim output is converted to `(w, y, pi)` by departure
   step: `y` from vehicle logs, `(w, pi)` from a time-dependent shortest-path
   search over counterfactual link travel times rebuilt from the logs. The
   same checker and the same DUE gap are then applied to every method.
4. **Same aggregates.** Total travel time, link volumes, route travel times
   and wall-clock time are computed with one set of formulas.

The notebook shows link volumes, total travel time, equilibrium residuals,
the day-to-day convergence of UXsim's DUE solver, and paper-style
route-equilibrium plots (route travel times with bands marking the departure
steps on which each route carries flow) for every method side by side.

## Tests

```bash
pytest -q
```

The suite checks agreement with the bulk reference on the 5-node network,
exactness on serial two-bottleneck chains (where a naive zero-demand rule is
shown to fail), equilibrium and physical residuals on random grids, agreement
of the HiGHS and Gurobi backends, and, when UXsim is installed, that the
counterfactual queue delay extracted from UXsim reproduces the LCP delay on a
single bottleneck.

## Citation

See `CITATION.cff`. Please cite the paper above when you use the algorithm.

## License

MIT
