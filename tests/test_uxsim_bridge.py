"""UXsim bridge: quantized demand, world construction, extraction consistency."""

from __future__ import annotations

import numpy as np
import pytest

uxsim = pytest.importorskip("uxsim")

from due_lcp.networks import serial_chain, wakui_5node  # noqa: E402
from due_lcp.one_to_many import check_solution, solve_sequential  # noqa: E402
from due_lcp.uxsim_bridge import (  # noqa: E402
    LinkLogs,
    UXsimConfig,
    enumerate_routes,
    extract_solution,
    quantize_demand,
    run_duo,
)


def test_quantize_preserves_total():
    net = wakui_5node()
    q = quantize_demand(net, 5)
    assert abs(q.total_demand() - net.total_demand()) < 1e-9
    for d in q.destinations:
        assert np.allclose((q.demand[d] * q.dt) % 5, 0.0)


def test_duo_extraction_shapes_and_conservation():
    net = quantize_demand(wakui_5node(), 5)
    W, _ = run_duo(net, UXsimConfig())
    sol = extract_solution(W, net)
    assert sol.w.shape == (net.n_links, net.T)
    assert sol.y.shape == (net.n_links, net.T)
    assert sol.n_unfinished == 0
    assert (
        abs(sum(sol.link_volumes[(1, 2)] + sol.link_volumes[(1, 3)] for _ in [0]) - 640)
        < 1e-9
    )
    rep = check_solution(net, sol.w, sol.y, sol.pi, sol.pi_nodes, tol=1e-6)
    assert rep.equilibrium.conservation < 1e-9
    assert len(enumerate_routes(net)[4]) == 3


def test_single_bottleneck_counterfactual_matches_lcp():
    """UXsim counterfactual delay at bin end reproduces the LCP queue delay."""
    net = quantize_demand(
        serial_chain(capacities=(10.0, 100.0), tau=3.0, T=30, bursts=((0, 8, 20.0),)), 5
    )
    lcp = solve_sequential(net, eps=1e-10, backend="highs")
    W, _ = run_duo(net, UXsimConfig())
    logs = LinkLogs(W, net)
    cf = np.array([logs.travel_time(0, (k + 1) * net.dt) - 3.0 for k in range(12)])
    assert np.max(np.abs(cf - lcp.w[0, :12])) < 0.5  # within one platoon headway
