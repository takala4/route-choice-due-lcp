"""Bridge between DUE-LCP inputs / outputs and UXsim, for like-for-like comparison.

Fairness measures implemented here:

1. Same input.  UXsim spawns vehicles in platoons of ``deltan``; a demand
   profile therefore has to be quantized to multiples of ``deltan`` per
   departure step.  `quantize_demand` does this once, and the quantized
   `NetworkData` is fed to BOTH the LCP solver and UXsim.
2. Same physics.  `build_world` applies the point-queue + FIFO recipe
   approved by the UXsim maintainer (GitHub Discussion #316): huge
   jam density (no spillback), explicit ``capacity_out = mu``, one lane,
   unthrottled ``capacity_in``, ``length = tau_hat`` with unit speed, and a
   dummy sink link behind every destination so that the last real link
   still enforces its bottleneck capacity.
3. Same output variables.  `extract_solution` converts a simulated World
   into the LCP variables indexed by origin-departure step:
   ``y`` from vehicle logs, ``(w, pi)`` from a time-dependent label-setting
   search over counterfactual link travel times rebuilt from the vehicle
   logs (`LinkLogs`).  The result can be passed
   to `verify.check_solution`, so the distance of a UXsim assignment from
   the DUE conditions is measured with exactly the same residuals as for
   the LCP solution.
4. Same aggregate definitions.  Total travel time, link volumes and route
   travel times are computed by the same formulas for both sides
   (`lcp_summary`, `UXsimSolution`).

UXsim is imported lazily so that the core package does not depend on it.
"""

from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass, field

import numpy as np

from .network import Link, NetworkData

LinkKey = tuple[int, int]
RouteKey = tuple[LinkKey, ...]


@dataclass
class UXsimConfig:
    """UXsim settings for the point-queue approximation.

    Attributes:
        deltan: Platoon size (vehicles per simulated entity).
        reaction_time: UXsim reaction time; the simulation time step is
            ``reaction_time * deltan``.  Small values sharpen the queue
            discretization and raise the fundamental-diagram capacity.
        jam_density: Huge value removes spillback (point queue).
        capacity_in: Huge value removes entry throttling.
        sink_length: Free-flow time of the dummy sink link behind each
            destination (subtracted from all travel times).
        tmax_factor: Simulation horizon as a multiple of the demand horizon.
        due_max_iter: Day-to-day iterations of UXsim's SolverDUE.
        due_n_routes: Routes enumerated per OD by SolverDUE.
        due_swap_prob: Route-swap probability of SolverDUE.
    """

    deltan: int = 5
    reaction_time: float = 0.02
    jam_density: float = 1000.0
    capacity_in: float = 1.0e4
    sink_length: float = 0.01
    tmax_factor: float = 4.0
    due_max_iter: int = 50
    due_n_routes: int = 8
    due_swap_prob: float = 0.10


@dataclass
class UXsimSolution:
    """A UXsim assignment expressed in LCP variables."""

    w: np.ndarray  # (L, K) queue delay by departure step, counterfactual chain
    y: np.ndarray  # (L, K) inflow rate of users departing in bin k, from vehicle logs
    pi: np.ndarray  # (N - 1, K) origin-to-node travel time, counterfactual chain
    pi_nodes: list[int]
    ttt: float  # total travel time (vehicles x time), sink link excluded
    n_vehicles: int  # actual vehicles that completed their trip
    n_unfinished: int
    link_volumes: dict[LinkKey, float]  # vehicles per link
    route_counts: dict[RouteKey, np.ndarray]  # vehicles per route per step
    route_tt: dict[RouteKey, np.ndarray]  # mean experienced TT per route per step
    route_tt_all: dict[RouteKey, np.ndarray]  # counterfactual TT, every simple route
    entry_wait_mean: float  # mean wait before entering the first link
    elapsed: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. same input
# ---------------------------------------------------------------------------
def _round_to_multiple(arr: np.ndarray, m: int) -> np.ndarray:
    """Round each element to a multiple of m preserving the (rounded) total."""
    units = arr / m
    target = int(round(units.sum()))
    floored = np.floor(units).astype(int)
    deficit = target - int(floored.sum())
    if deficit > 0:
        order = np.argsort(-(units - floored))
        floored[order[:deficit]] += 1
    return (floored * m).astype(float)


def quantize_demand(net: NetworkData, deltan: int) -> NetworkData:
    """Return a copy of `net` whose per-step demand volumes are multiples of deltan.

    UXsim's ``adddemand(volume=...)`` spawns ``volume / deltan`` entities, so
    non-multiples are silently truncated.  Quantizing up front and using the
    quantized profile on both sides keeps the inputs identical.
    """
    demand = {}
    for d, rate in net.demand.items():
        volume = np.asarray(rate, dtype=float) * net.dt
        demand[d] = _round_to_multiple(volume, deltan) / net.dt
    return NetworkData(
        nodes=list(net.nodes),
        origin=net.origin,
        destinations=list(net.destinations),
        links=list(net.links),
        dt=net.dt,
        T=net.T,
        demand=demand,
    )


# ---------------------------------------------------------------------------
# 2. same physics
# ---------------------------------------------------------------------------
def _node_name(n: int) -> str:
    return str(n)


def _sink_name(d: int) -> str:
    return f"{d}_sink"


def _link_name(link: Link) -> str:
    return f"{link.tail}->{link.head}"


def build_world(
    net: NetworkData,
    cfg: UXsimConfig,
    *,
    name: str = "rcdue",
    seed: int = 0,
    route_choice_principle: str = "homogeneous_DUO",
):
    """Build a UXsim World reproducing `net` under the point-queue recipe."""
    from uxsim import World

    W = World(
        name=name,
        deltan=cfg.deltan,
        reaction_time=cfg.reaction_time,
        tmax=net.horizon * cfg.tmax_factor,
        random_seed=seed,
        print_mode=0,
        save_mode=0,
        show_mode=0,
        show_progress=0,
        route_choice_principle=route_choice_principle,
        duo_update_time=net.dt,
        duo_update_weight=1.0,
    )
    for n in net.nodes:
        W.addNode(_node_name(n), x=0, y=0)
    for d in net.destinations:
        W.addNode(_sink_name(d), x=0, y=0)
    for link in net.links:
        W.addLink(
            _link_name(link),
            _node_name(link.tail),
            _node_name(link.head),
            length=link.tau_hat,
            free_flow_speed=1.0,
            jam_density=cfg.jam_density,
            capacity_out=link.mu_hat,
            capacity_in=cfg.capacity_in,
            number_of_lanes=1,
        )
    for d in net.destinations:
        W.addLink(
            f"{d}->{_sink_name(d)}",
            _node_name(d),
            _sink_name(d),
            length=cfg.sink_length,
            free_flow_speed=1.0,
            jam_density=cfg.jam_density,
            capacity_in=cfg.capacity_in,
            number_of_lanes=1,
        )
    for d in net.destinations:
        volumes = np.asarray(net.demand[d]) * net.dt
        for k, vol in enumerate(volumes):
            v = int(round(vol))
            if v > 0:
                W.adddemand(
                    _node_name(net.origin),
                    _sink_name(d),
                    k * net.dt,
                    (k + 1) * net.dt,
                    volume=v,
                )
    return W


def run_duo(net: NetworkData, cfg: UXsimConfig, *, seed: int = 0):
    """Run a single UXsim simulation under homogeneous DUO route choice."""
    t0 = time.perf_counter()
    W = build_world(net, cfg, name="DUO", seed=seed)
    W.exec_simulation()
    return W, time.perf_counter() - t0


def run_due(
    net: NetworkData, cfg: UXsimConfig, *, seed: int = 0, verbose: bool = False
):
    """Run UXsim's day-to-day SolverDUE.  Returns (World, solver, elapsed)."""
    from uxsim.DTAsolvers.DTAsolvers import SolverDUE

    random.seed(seed)  # SolverDUE draws route swaps from the ``random`` module
    t0 = time.perf_counter()
    solver = SolverDUE(lambda: build_world(net, cfg, name="DUE", seed=seed))
    W = solver.solve(
        max_iter=cfg.due_max_iter,
        n_routes_per_od=cfg.due_n_routes,
        swap_prob=cfg.due_swap_prob,
        print_progress=verbose,
    )
    return W, solver, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 3. same output variables
# ---------------------------------------------------------------------------
def _link_index(net: NetworkData) -> dict[str, int]:
    return {_link_name(link): ij for ij, link in enumerate(net.links)}


class LinkLogs:
    """Per-link entry / exit times of every completed vehicle, from vehicle logs.

    Gives a counterfactual link travel time for ANY entry time t under the
    point-queue + FIFO rule: a vehicle entering at t leaves at

        max( t + tau_hat,  (latest exit among vehicles that entered before t)
                           + deltan / mu_hat ),

    i.e. it queues behind everybody already on the link and is served one
    platoon-headway after the last of them.  Unlike ``Link.actual_travel_time``
    this decays correctly after the last real vehicle has entered, which is
    what the LCP variables need on demand-free steps.  While vehicles are
    present the two agree to within one platoon headway.
    """

    def __init__(self, W, net: NetworkData):
        self.net = net
        self.deltan = float(W.DELTAN)
        name_to_ij = _link_index(net)
        entries: list[list[float]] = [[] for _ in net.links]
        exits: list[list[float]] = [[] for _ in net.links]
        for v in W.VEHICLES.values():
            if v.state != "end":
                continue
            route, times = v.traveled_route()
            for i, link in enumerate(route):
                ij = name_to_ij.get(link.name)
                if ij is None or i + 1 >= len(times):
                    continue
                entries[ij].append(float(times[i]))
                exits[ij].append(float(times[i + 1]))
        self.entry: list[np.ndarray] = []
        self.exit_cummax: list[np.ndarray] = []
        for ij in range(net.n_links):
            ent = np.asarray(entries[ij])
            ex = np.asarray(exits[ij])
            order = np.argsort(ent, kind="stable")
            self.entry.append(ent[order])
            self.exit_cummax.append(
                np.maximum.accumulate(ex[order]) if len(ex) else np.empty(0)
            )

    def travel_time(self, ij: int, t: float) -> float:
        """Counterfactual travel time on link ij for entry at clock time t."""
        link = self.net.links[ij]
        i = int(np.searchsorted(self.entry[ij], t, side="left"))
        if i == 0:
            return link.tau_hat
        last_exit = float(self.exit_cummax[ij][i - 1])
        return max(t + link.tau_hat, last_exit + self.deltan / link.mu_hat) - t


def counterfactual_state(
    logs: LinkLogs, net: NetworkData, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """(w, pi) for LCP step k (0-based), i.e. departure at clock time (k + 1) dt.

    Time convention.  The LCP variables w^kappa, pi^kappa belong to users
    departing the origin at s_kappa = kappa ds, and the queue-dynamics
    row uses the inflow y^kappa of the same step (implicit discretization),
    so w^kappa is the delay seen at the END of demand bin kappa
    [(kappa - 1) ds, kappa ds).  With 0-based k this is clock time
    (k + 1) dt.  Verified on a single-bottleneck chain: LCP w = 1, 2, ...
    equals the point-queue delay at s = kappa ds, and the UXsim counterfactual
    at bin end reproduces it to within one platoon headway.

    Time-dependent label-setting from the origin: node u settled at clock
    time t_u; out-link (u, v) is entered at t_u and takes
    ``logs.travel_time(uv, t_u)``.  Because the links are FIFO the earliest
    arrival labels are exact.  ``w_uv`` is recorded when u is settled, so
    every link gets a queue delay for the entry time a shortest-path user
    would experience, whether or not any vehicle actually used it.
    """
    L = net.n_links
    t_dep = (k + 1) * net.dt
    inf = float("inf")
    arrival = {n: inf for n in net.nodes}
    arrival[net.origin] = t_dep
    outlinks: dict[int, list[int]] = {n: [] for n in net.nodes}
    for ij, link in enumerate(net.links):
        outlinks[link.tail].append(ij)
    w = np.zeros(L)
    settled: set[int] = set()
    heap: list[tuple[float, int]] = [(t_dep, net.origin)]
    while heap:
        t_u, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        for ij in outlinks[u]:
            link = net.links[ij]
            tt = logs.travel_time(ij, t_u)
            w[ij] = max(tt - link.tau_hat, 0.0)
            t_v = t_u + tt
            if t_v < arrival[link.head]:
                arrival[link.head] = t_v
                heapq.heappush(heap, (t_v, link.head))
    pi_nodes = [n for n in net.nodes if n != net.origin]
    pi = np.array([arrival[n] - t_dep for n in pi_nodes])
    return w, pi


def enumerate_routes(net: NetworkData) -> dict[int, list[RouteKey]]:
    """All simple origin-to-destination routes, keyed by destination."""
    outlinks: dict[int, list[Link]] = {n: [] for n in net.nodes}
    for link in net.links:
        outlinks[link.tail].append(link)
    routes: dict[int, list[RouteKey]] = {d: [] for d in net.destinations}

    def dfs(node: int, path: list[LinkKey], visited: set[int]) -> None:
        if node in routes and path:
            routes[node].append(tuple(path))
        for link in outlinks[node]:
            if link.head in visited:
                continue
            dfs(link.head, [*path, link.key], visited | {link.head})

    dfs(net.origin, [], {net.origin})
    return routes


def route_counterfactual_tt(
    logs: LinkLogs, net: NetworkData, route: RouteKey
) -> np.ndarray:
    """Travel time of `route` for a departure at clock time (k + 1) dt, k = 0..K-1.

    Same time convention as `counterfactual_state` (LCP step k = departure
    at the end of demand bin k).  Chains `LinkLogs.travel_time` link by
    link, so the value is defined whether or not any vehicle used the route.
    """
    ijs = [net.link_index(*key) for key in route]
    tt = np.zeros(net.T)
    for k in range(net.T):
        t_dep = (k + 1) * net.dt
        t = t_dep
        for ij in ijs:
            t += logs.travel_time(ij, t)
        tt[k] = t - t_dep
    return tt


def extract_solution(W, net: NetworkData, *, elapsed: float = 0.0) -> UXsimSolution:
    """Convert a simulated World into LCP variables and aggregate statistics."""
    L = net.n_links
    K = net.T
    dt = net.dt
    deltan = float(W.DELTAN)
    name_to_ij = _link_index(net)
    pi_nodes = [n for n in net.nodes if n != net.origin]

    y = np.zeros((L, K))
    link_volumes: dict[LinkKey, float] = {link.key: 0.0 for link in net.links}
    route_counts: dict[RouteKey, np.ndarray] = {}
    route_tt_bins: dict[RouteKey, list[list[float]]] = {}
    ttt = 0.0
    n_done = 0
    n_unfinished = 0
    entry_waits: list[float] = []

    for v in W.VEHICLES.values():
        if v.state != "end":
            n_unfinished += 1
            continue
        route, times = v.traveled_route()
        sig: list[LinkKey] = []
        for link in route:
            ij = name_to_ij.get(link.name)
            if ij is not None:
                sig.append(net.links[ij].key)
        dep = float(v.departure_time) * float(W.DELTAT)
        k = int(np.floor(dep / dt + 1e-9))
        # arrival at the destination node = entry time of the sink link
        t_dest = float(times[-2]) if len(times) >= 2 else float(times[-1])
        tt = t_dest - dep
        entry_waits.append(float(times[0]) - dep)
        ttt += tt * deltan
        n_done += 1
        if not 0 <= k < K:
            continue
        for key in sig:
            ij = net.link_index(*key)
            y[ij, k] += deltan / dt
            link_volumes[key] += deltan
        rk: RouteKey = tuple(sig)
        if rk not in route_counts:
            route_counts[rk] = np.zeros(K)
            route_tt_bins[rk] = [[] for _ in range(K)]
        route_counts[rk][k] += deltan
        route_tt_bins[rk][k].append(tt)

    logs = LinkLogs(W, net)
    w = np.zeros((L, K))
    pi = np.zeros((len(pi_nodes), K))
    for k in range(K):
        w[:, k], pi[:, k] = counterfactual_state(logs, net, k)

    route_tt = {
        rk: np.array([np.mean(b) if b else np.nan for b in bins])
        for rk, bins in route_tt_bins.items()
    }
    route_tt_all = {
        rk: route_counterfactual_tt(logs, net, rk)
        for routes in enumerate_routes(net).values()
        for rk in routes
    }
    return UXsimSolution(
        w=w,
        y=y,
        pi=pi,
        pi_nodes=pi_nodes,
        ttt=ttt,
        n_vehicles=int(n_done * deltan),
        n_unfinished=int(n_unfinished * deltan),
        link_volumes=link_volumes,
        route_counts=route_counts,
        route_tt=route_tt,
        route_tt_all=route_tt_all,
        entry_wait_mean=float(np.mean(entry_waits)) if entry_waits else 0.0,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# 4. same aggregate definitions for the LCP side
# ---------------------------------------------------------------------------
def lcp_summary(net: NetworkData, res) -> dict:
    """Total travel time and link volumes of a `SequentialResult`."""
    ttt = 0.0
    for d in net.destinations:
        ttt += float(np.sum(net.demand[d] * net.dt * res.pi[res.pi_idx_of[d]]))
    volumes = {
        link.key: float(np.sum(res.y[ij]) * net.dt) for ij, link in enumerate(net.links)
    }
    return {"ttt": ttt, "link_volumes": volumes}


def lcp_route_used(
    net: NetworkData, res, route: RouteKey, *, tol: float = 1e-6
) -> np.ndarray:
    """Boolean per step: route travel time equals pi_d and every link carries flow.

    The LCP is link based, so route flows are not unique; this is the
    used-route criterion implied by the DUE conditions.
    """
    d = route[-1][1]
    tt = lcp_route_tt(net, res, route)
    on_shortest = np.abs(tt - res.pi[res.pi_idx_of[d]]) <= tol
    flow = np.all([res.y[net.link_index(*key)] > tol for key in route], axis=0)
    return on_shortest & flow


def lcp_route_tt(net: NetworkData, res, route: RouteKey) -> np.ndarray:
    """Route travel time by departure step: sum of (tau_hat + w_ij[kappa])."""
    tt = np.zeros(net.T)
    for key in route:
        ij = net.link_index(*key)
        tt += net.links[ij].tau_hat + res.w[ij]
    return tt
