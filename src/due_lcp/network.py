"""Network and demand data on a discrete departure-time grid.

The point-queue network is described by links with a free-flow travel time
``tau_hat`` and a bottleneck (outflow) capacity ``mu_hat``.  Demand is a
piecewise-constant departure rate per destination on the grid
s_k = k * dt, k = 0, ..., T.

`NetworkData` is written for the one-to-many case (single origin).  The
many-to-one case will reuse the same link representation with the roles of
origin and destinations swapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Link:
    """Directed link with free-flow time and bottleneck capacity."""

    tail: int
    head: int
    tau_hat: float
    mu_hat: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.tail, self.head)


@dataclass
class NetworkData:
    """One-to-many network on a discrete time grid.

    Attributes:
        nodes: Node ids.
        origin: The single origin node.
        destinations: Destination nodes (subset of ``nodes``, excluding the origin).
        links: Directed links; (tail, head) pairs must be unique.
        dt: Length of one departure-time step.
        T: Number of steps; the horizon is ``T * dt``.
        demand: Departure *rate* (vehicles per unit time) toward each destination
            during the half-open interval [s_k, s_{k+1}), as arrays of length T.
    """

    nodes: list[int]
    origin: int
    destinations: list[int]
    links: list[Link]
    dt: float
    T: int
    demand: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.origin not in self.nodes:
            raise ValueError(f"origin {self.origin} is not a node")
        if self.dt <= 0 or self.T <= 0:
            raise ValueError("dt and T must be positive")
        for d in self.destinations:
            if d not in self.nodes or d == self.origin:
                raise ValueError(f"destination {d} must be a non-origin node")
            if d not in self.demand:
                raise ValueError(f"demand[{d}] is missing")
            arr = np.asarray(self.demand[d], dtype=float)
            if arr.shape != (self.T,):
                raise ValueError(
                    f"demand[{d}] must have length T={self.T}, got {arr.shape}"
                )
            if np.any(arr < 0):
                raise ValueError(f"demand[{d}] must be non-negative")
            self.demand[d] = arr
        for link in self.links:
            if link.tail not in self.nodes or link.head not in self.nodes:
                raise ValueError(f"link {link.key} references an unknown node")
            if link.tau_hat <= 0 or link.mu_hat <= 0:
                raise ValueError(
                    f"link {link.key}: tau_hat and mu_hat must be positive"
                )
        keys = [link.key for link in self.links]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate link")
        self._index = {key: i for i, key in enumerate(keys)}

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def horizon(self) -> float:
        return self.T * self.dt

    @property
    def pi_nodes(self) -> list[int]:
        """Non-origin nodes, in the order used for the pi variables."""
        return [n for n in self.nodes if n != self.origin]

    def link_index(self, tail: int, head: int) -> int:
        return self._index[(tail, head)]

    def total_demand(self) -> float:
        return sum(float(arr.sum()) * self.dt for arr in self.demand.values())
