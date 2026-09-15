"""Build sparse LCP system (M, q, Aineq, bineq) for [DUE-LCP].

Reference: Wakui, Sakai, Akamatsu (2023), JSCE Vol.79(4), 22-00301.

Variables x = (w, y, pi) stacked, size (2L + N - 1) * K.
  w[ij, k] : queue delay (time units) on link (i,j) for users departing
             origin at discrete step k                              size  L * K
  y[ij, k] : link inflow rate (vehicles/time) on (i,j) for users
             departing origin at step k                             size  L * K
  pi[n, k] : shortest TT from origin to non-origin node n for users
             departing origin at step k                             size (N-1)*K

Boundary / initial:
  w[*, 0] = 0,  y[*, 0] = 0,  pi[n, 0] = pi0[n]  (free-flow shortest path)
  pi[origin, *] = 0  (excluded from x)

LCP residuals (R = M x + q, with 0 <= x perp R >= 0):

  R_w[ij, k]  = alpha_ij * (w[ij,k] - w[ij,k-1])
                - y[ij, k]
                + alpha_ij * (pi[i,k] - pi[i,k-1])     (i = link.tail; 0 if origin)
                + mu_ij
              alpha_ij = mu_ij / dt                                     (queue dyn)

  R_y[ij, k]  = c_hat_ij + w[ij, k] + pi[i, k] - pi[j, k]               (DUE)
              (pi[origin, *] = 0)

  R_pi[n, k]  = sum_{(j,n) in L} y[jn, k] - sum_{(n,j) in L} y[nj, k]
                - q^k_n                                                 (conservation)

Inequality (Aineq x <= bineq):
  pi[n, k] >= pi[n, k-1] - dt
  pi[n, k] >= pi0[n]
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..network import NetworkData


@dataclass
class LCPSystem:
    """Sparse LCP: 0 <= x perp (M x + q) >= 0, Aineq x <= bineq."""

    M: sp.csr_matrix
    q: np.ndarray
    Aineq: sp.csr_matrix
    bineq: np.ndarray
    var_index: dict[str, slice]
    var_meta: dict
    n: int

    def residual(self, x: np.ndarray) -> np.ndarray:
        return self.M @ x + self.q

    def phi(self, x: np.ndarray) -> float:
        """LCP gap function phi(x) = x^T (M x + q). Zero at LCP solution."""
        return float(x @ (self.M @ x + self.q))

    def grad_phi(self, x: np.ndarray) -> np.ndarray:
        """Gradient of phi: (M + M^T) x + q."""
        return (self.M + self.M.T) @ x + self.q


def free_flow_shortest_path(net: NetworkData) -> dict[int, float]:
    """Dijkstra: free-flow shortest TT from origin to each node."""
    inf = float("inf")
    dist = {n: inf for n in net.nodes}
    dist[net.origin] = 0.0
    adj: dict[int, list[tuple[int, float]]] = {n: [] for n in net.nodes}
    for link in net.links:
        adj[link.tail].append((link.head, link.tau_hat))
    pq: list[tuple[float, int]] = [(0.0, net.origin)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w_uv in adj[u]:
            nd = d + w_uv
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def build_lcp(net: NetworkData) -> LCPSystem:
    """Assemble the sparse LCP per Wakui-Sakai-Akamatsu 2023, eqs (w/y/pi)_comple."""
    L = net.n_links
    K = net.T  # paper's K = number of non-initial discrete time points
    dt = net.dt
    nodes = net.nodes
    origin = net.origin

    pi_nodes = [n for n in nodes if n != origin]
    n_pi_nodes = len(pi_nodes)
    pi_idx_of = {n: idx for idx, n in enumerate(pi_nodes)}

    n_w = L * K
    n_y = L * K
    n_pi = n_pi_nodes * K
    n = n_w + n_y + n_pi

    var_index = {
        "w": slice(0, n_w),
        "y": slice(n_w, n_w + n_y),
        "pi": slice(n_w + n_y, n),
    }

    def i_w(ij: int, k: int) -> int:
        return ij * K + (k - 1)

    def i_y(ij: int, k: int) -> int:
        return n_w + ij * K + (k - 1)

    def i_pi(node: int, k: int) -> int:
        return n_w + n_y + pi_idx_of[node] * K + (k - 1)

    pi0_dist = free_flow_shortest_path(net)
    pi0 = np.array([pi0_dist[n] for n in pi_nodes])

    # Total demand RATE per non-origin node n at step k: q_n^k
    q_node_step = np.zeros((n_pi_nodes, K))
    for d in net.destinations:
        q_node_step[pi_idx_of[d]] += net.demand[d]  # rate (vehicles/time)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    qvec = np.zeros(n)

    # ---- Block W: queue dynamics
    # R_w[ij,k] = alpha (w^k - w^{k-1}) - y^k + alpha (pi_i^k - pi_i^{k-1}) + mu
    # pi_i^0 = pi0[i]  (constant).  pi_origin = 0 always.
    for ij, link in enumerate(net.links):
        alpha = link.mu_hat / dt
        i_tail_is_origin = link.tail == origin
        for k in range(1, K + 1):
            row = i_w(ij, k)
            # +alpha * w^k
            rows.append(row)
            cols.append(i_w(ij, k))
            vals.append(alpha)
            # -alpha * w^{k-1}  (skip if k=1 since w^0 = 0)
            if k >= 2:
                rows.append(row)
                cols.append(i_w(ij, k - 1))
                vals.append(-alpha)
            # -y^k
            rows.append(row)
            cols.append(i_y(ij, k))
            vals.append(-1.0)
            # +alpha * pi_i^k  (only if tail != origin)
            if not i_tail_is_origin:
                rows.append(row)
                cols.append(i_pi(link.tail, k))
                vals.append(alpha)
            # -alpha * pi_i^{k-1}
            if not i_tail_is_origin:
                if k >= 2:
                    rows.append(row)
                    cols.append(i_pi(link.tail, k - 1))
                    vals.append(-alpha)
                else:
                    # k = 1: -alpha * pi_i^0 -> goes to qvec
                    qvec[row] += -alpha * pi0[pi_idx_of[link.tail]]
            # constant +mu
            qvec[row] += link.mu_hat

    # ---- Block Y: shortest-path complementarity (DUE)
    # R_y[ij,k] = c_hat_ij + w^k + pi_i^k - pi_j^k
    for ij, link in enumerate(net.links):
        for k in range(1, K + 1):
            row = i_y(ij, k)
            # +1 * w^k
            rows.append(row)
            cols.append(i_w(ij, k))
            vals.append(1.0)
            # +pi_i^k (if tail != origin)
            if link.tail != origin:
                rows.append(row)
                cols.append(i_pi(link.tail, k))
                vals.append(1.0)
            # -pi_j^k (if head != origin)
            if link.head != origin:
                rows.append(row)
                cols.append(i_pi(link.head, k))
                vals.append(-1.0)
            # constant +c_hat
            qvec[row] = link.tau_hat

    # ---- Block Pi: conservation
    # R_pi[n,k] = sum_{(j,n) in L} y[jn, k] - sum_{(n,j) in L} y[nj, k] - q^k_n
    inlinks: dict[int, list[int]] = {n: [] for n in nodes}
    outlinks: dict[int, list[int]] = {n: [] for n in nodes}
    for ij, link in enumerate(net.links):
        outlinks[link.tail].append(ij)
        inlinks[link.head].append(ij)
    for n_node in pi_nodes:
        n_local = pi_idx_of[n_node]
        for k in range(1, K + 1):
            row = i_pi(n_node, k)
            for ij in inlinks[n_node]:
                rows.append(row)
                cols.append(i_y(ij, k))
                vals.append(1.0)
            for ij in outlinks[n_node]:
                rows.append(row)
                cols.append(i_y(ij, k))
                vals.append(-1.0)
            qvec[row] = -q_node_step[n_local, k - 1]

    M = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    # ---- Inequality on pi (boundary): in form  Aineq x <= bineq
    # (B1) pi^k - pi^{k-1} >= -dt   <=>   -pi^k + pi^{k-1} <= dt
    # (B2) pi^k >= pi0                <=>   -pi^k <= -pi0
    rows_I: list[int] = []
    cols_I: list[int] = []
    vals_I: list[float] = []
    bineq_list: list[float] = []
    row_counter = 0
    for n_node in pi_nodes:
        for k in range(1, K + 1):
            # (B1)
            rows_I.append(row_counter)
            cols_I.append(i_pi(n_node, k))
            vals_I.append(-1.0)
            if k >= 2:
                rows_I.append(row_counter)
                cols_I.append(i_pi(n_node, k - 1))
                vals_I.append(1.0)
                bineq_list.append(dt)
            else:
                # pi^0 is constant
                bineq_list.append(dt - pi0[pi_idx_of[n_node]])
            row_counter += 1
            # (B2)
            rows_I.append(row_counter)
            cols_I.append(i_pi(n_node, k))
            vals_I.append(-1.0)
            bineq_list.append(-pi0[pi_idx_of[n_node]])
            row_counter += 1
    Aineq = sp.csr_matrix(
        (vals_I, (rows_I, cols_I)),
        shape=(row_counter, n),
    )
    bineq = np.array(bineq_list)

    var_meta = {
        "L": L,
        "K": K,
        "dt": dt,
        "n_pi_nodes": n_pi_nodes,
        "pi_idx_of": pi_idx_of,
        "pi_nodes": pi_nodes,
        "pi0": pi0,
        "origin": origin,
    }
    return LCPSystem(
        M=M,
        q=qvec,
        Aineq=Aineq,
        bineq=bineq,
        var_index=var_index,
        var_meta=var_meta,
        n=n,
    )
