"""Correctness tests for the sequential (paper) solver."""

from __future__ import annotations

import numpy as np
import pytest

from rcdue.lp_backend import gurobi_available
from rcdue.networks import grid, serial_chain, wakui_5node
from rcdue.one_to_many import (
    check_bulk_result,
    check_sequential_result,
    fw_solver,
    solve_sequential,
)
from rcdue.one_to_many.lcp_builder import build_lcp

BACKENDS = ["highs"] + (["gurobi"] if gurobi_available() else [])


def _bulk(net, eps=1e-10, max_iter=3000):
    lcp = build_lcp(net)
    res = fw_solver.solve(lcp, eps=eps, max_iter=max_iter, backend="highs")
    L, K = net.n_links, net.T
    w = res.x[lcp.var_index["w"]].reshape(L, K)
    y = res.x[lcp.var_index["y"]].reshape(L, K)
    pi = res.x[lcp.var_index["pi"]].reshape(-1, K)
    return lcp, res, w, y, pi


@pytest.mark.parametrize("backend", BACKENDS)
def test_5node_matches_bulk_and_passes_checks(backend):
    net = wakui_5node()
    seq = solve_sequential(net, eps=1e-10, backend=backend)
    rep = check_sequential_result(net, seq, tol=1e-8)
    assert rep.passed, rep.summary()
    lcp, bulk, w, y, pi = _bulk(net)
    assert check_bulk_result(net, lcp, bulk.x, tol=1e-8).equilibrium_ok
    assert np.max(np.abs(seq.pi - pi)) < 1e-8
    assert np.max(np.abs(seq.y - y)) < 1e-8
    assert np.max(np.abs(seq.w - w)) < 1e-8


@pytest.mark.parametrize("backend", BACKENDS)
def test_serial_chain_exact(backend):
    """Two bottlenecks in series with a demand gap: exercises the ZD rule ds - d_v."""
    net = serial_chain()
    seq = solve_sequential(net, eps=1e-10, backend=backend)
    rep = check_sequential_result(net, seq, tol=1e-9)
    assert rep.passed, rep.summary()
    _, _, w, y, pi = _bulk(net)
    assert np.max(np.abs(seq.w - w)) < 1e-9
    assert np.max(np.abs(seq.pi - pi)) < 1e-9


def test_single_bottleneck_analytic():
    """LCP step kappa equals the point-queue delay at departure time kappa * ds."""
    net = serial_chain(capacities=(10.0, 100.0), T=30, bursts=((0, 8, 20.0),))
    seq = solve_sequential(net, eps=1e-10, backend="highs")
    expected = [min(k, 16 - k) for k in range(1, 16)]  # grows 1/min, then clears
    assert np.allclose(seq.w[0, :15], expected, atol=1e-8)


@pytest.mark.parametrize("n", [4, 6])
def test_grid_residuals(n):
    net = grid(n)
    seq = solve_sequential(net, eps=1e-8, backend="highs")
    rep = check_sequential_result(net, seq, tol=1e-6)
    assert rep.passed, rep.summary()


def test_zd_method_pd_agrees_with_dijkstra():
    net = serial_chain()
    a = solve_sequential(net, eps=1e-10, backend="highs", zd_method="dijkstra")
    b = solve_sequential(net, eps=1e-10, backend="highs", zd_method="pd")
    assert np.max(np.abs(a.pi - b.pi)) < 1e-8
    assert np.max(np.abs(a.y - b.y)) < 1e-8


@pytest.mark.skipif(not gurobi_available(), reason="gurobi not available")
def test_backends_agree():
    net = wakui_5node()
    a = solve_sequential(net, eps=1e-10, backend="highs")
    b = solve_sequential(net, eps=1e-10, backend="gurobi")
    assert np.max(np.abs(a.pi - b.pi)) < 1e-8
    assert np.max(np.abs(a.y - b.y)) < 1e-8
