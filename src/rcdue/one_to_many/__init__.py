"""One-to-many dynamic user equilibrium (single origin, several destinations).

Reference: Wakui, Sakai, Akamatsu (2023), "Efficient algorithm for solving
dynamic user equilibrium traffic assignment", Journal of JSCE, Vol. 79,
No. 4, 22-00301 (in Japanese).

Modules:
    sequential: the paper's time-decomposed PD / ZD algorithm (main solver)
    verify:     independent checker of the equilibrium and physical conditions
    lcp_builder, fw_solver: bulk all-time LCP solved by Frank-Wolfe (reference)
"""

from .sequential import SequentialResult, solve_sequential
from .verify import (
    CheckReport,
    check_bulk_result,
    check_sequential_result,
    check_solution,
    excess_travel_time,
)

__all__ = [
    "CheckReport",
    "SequentialResult",
    "check_bulk_result",
    "check_sequential_result",
    "check_solution",
    "excess_travel_time",
    "solve_sequential",
]
