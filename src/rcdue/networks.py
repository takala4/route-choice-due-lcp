"""Small test networks used in the examples and the test suite."""

from __future__ import annotations

import numpy as np

from .network import Link, NetworkData


def triangular_demand(T: int, dt: float, pulse_end: float, total: float) -> np.ndarray:
    """Symmetric triangular rate profile on [0, pulse_end] with integral ``total``."""
    pulse_steps = int(round(pulse_end / dt))
    half = pulse_end / 2.0
    peak = 2.0 * total / pulse_end
    q = np.zeros(T)
    for k in range(pulse_steps):
        s_mid = (k + 0.5) * dt
        q[k] = peak * max(0.0, 1.0 - abs(s_mid - half) / half)
    return q


def wakui_5node(
    dt: float = 1.0,
    horizon: float = 60.0,
    demand_pulse_end: float = 30.0,
    total_to_4: float = 320.0,
    total_to_5: float = 320.0,
) -> NetworkData:
    """5-node, 6-link test network of Wakui, Sakai and Akamatsu (2023), Fig. 1.

    Origin 1, destinations 4 and 5.  Default settings reproduce the paper's
    preliminary experiment: horizon 60, ds = 1, 320 users to each
    destination as a triangular pulse over [0, 30].
    """
    T = int(round(horizon / dt))
    return NetworkData(
        nodes=[1, 2, 3, 4, 5],
        origin=1,
        destinations=[4, 5],
        links=[
            Link(1, 2, tau_hat=3.0, mu_hat=8.0),
            Link(1, 3, tau_hat=15.0, mu_hat=12.0),
            Link(2, 3, tau_hat=2.0, mu_hat=4.0),
            Link(2, 4, tau_hat=5.0, mu_hat=6.0),
            Link(3, 4, tau_hat=10.0, mu_hat=4.0),
            Link(3, 5, tau_hat=1.0, mu_hat=6.0),
        ],
        dt=dt,
        T=T,
        demand={
            4: triangular_demand(T, dt, demand_pulse_end, total_to_4),
            5: triangular_demand(T, dt, demand_pulse_end, total_to_5),
        },
    )


def serial_chain(
    capacities: tuple[float, ...] = (10.0, 5.0, 100.0),
    tau: float = 2.0,
    dt: float = 1.0,
    T: int = 40,
    bursts: tuple[tuple[int, int, float], ...] = ((0, 8, 20.0), (14, 20, 20.0)),
) -> NetworkData:
    """Serial chain 0 -> 1 -> ... -> n with one bottleneck per link.

    Two bottlenecks in series with a demand gap followed by resumed demand
    is the smallest case in which the upstream queue delays the clearing of
    the downstream queue during zero-demand steps (the ZD algorithm's
    ``ds - d_v`` rule matters).
    """
    n = len(capacities)
    demand = np.zeros(T)
    for a, b, rate in bursts:
        demand[a:b] = rate
    return NetworkData(
        nodes=list(range(n + 1)),
        origin=0,
        destinations=[n],
        links=[Link(i, i + 1, tau, mu) for i, mu in enumerate(capacities)],
        dt=dt,
        T=T,
        demand={n: demand},
    )


def grid(n: int, T: int = 60, seed: int = 1) -> NetworkData:
    """n x n grid with random free-flow times / capacities and three destinations.

    Origin is node 0 (top-left); destinations are the other three corners
    minus one, i.e. nodes n-1, n(n-1) and n*n-1.  Links point right and
    down only, so every node is reachable and the graph is acyclic.
    """
    rng = np.random.default_rng(seed)
    nodes = list(range(n * n))
    links: list[Link] = []
    for i in range(n):
        for j in range(n):
            u = i * n + j
            if j + 1 < n:
                links.append(
                    Link(u, u + 1, float(rng.integers(1, 4)), float(rng.uniform(3, 12)))
                )
            if i + 1 < n:
                links.append(
                    Link(u, u + n, float(rng.integers(1, 4)), float(rng.uniform(3, 12)))
                )
    dests = [n * n - 1, n - 1, n * (n - 1)]
    demand: dict[int, np.ndarray] = {}
    for d in dests:
        q = np.zeros(T)
        a, b = sorted(rng.integers(0, T // 2, 2))
        q[a : b + 5] = rng.uniform(5, 15)
        q[T // 2 + 5 : T // 2 + 12] = rng.uniform(3, 8)
        demand[d] = q
    return NetworkData(
        nodes=nodes,
        origin=0,
        destinations=dests,
        links=links,
        dt=1.0,
        T=T,
        demand=demand,
    )
