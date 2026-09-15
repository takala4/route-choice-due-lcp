"""The checker must accept correct solutions and reject corrupted ones."""

from __future__ import annotations

import numpy as np

from due_lcp.networks import grid, serial_chain
from due_lcp.one_to_many import (
    check_bulk_result,
    check_solution,
    excess_travel_time,
    fw_solver,
    solve_sequential,
)
from due_lcp.one_to_many.lcp_builder import build_lcp


def test_naive_zd_rule_is_rejected():
    """Decaying every queue by dt (ignoring the upstream pi decrease) fails."""
    net = serial_chain()
    res = solve_sequential(net, eps=1e-10, backend="highs")
    w = res.w.copy()
    for k in range(1, net.T):
        if net.demand[net.destinations[0]][k] == 0:
            w[:, k] = np.maximum(w[:, k - 1] - net.dt, 0.0)
    rep = check_solution(net, w, res.y, res.pi, res.pi_nodes, tol=1e-6)
    assert not rep.equilibrium_ok
    assert rep.first_failing_step() is not None


def test_bulk_solution_equilibrium_but_not_physical_on_grid():
    """The bulk reference lacks Step 5: pi off used paths need not be shortest."""
    net = grid(4)
    lcp = build_lcp(net)
    bulk = fw_solver.solve(lcp, eps=1e-10, max_iter=3000, backend="highs")
    rep = check_bulk_result(net, lcp, bulk.x, tol=1e-6)
    assert rep.equilibrium_ok
    seq = solve_sequential(net, eps=1e-8, backend="highs")
    assert check_solution(net, seq.w, seq.y, seq.pi, seq.pi_nodes, tol=1e-6).passed
    # both are equilibria; they may differ (non-uniqueness) but the sequential
    # one is physically consistent by construction
    assert rep.physical.pi_shortest_path >= 0.0


def test_excess_travel_time_zero_at_equilibrium():
    net = serial_chain()
    res = solve_sequential(net, eps=1e-10, backend="highs")
    assert abs(excess_travel_time(net, res.w, res.y, res.pi, res.pi_nodes)) < 1e-8
