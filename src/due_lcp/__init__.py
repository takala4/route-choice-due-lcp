"""due-lcp: dynamic user equilibrium assignment via linear complementarity.

Reference implementation of the time-decomposed DUE algorithm of Wakui,
Sakai and Akamatsu (2023) for one-to-many demand on point-queue networks,
together with an independent equilibrium checker and a bridge to UXsim for
like-for-like comparisons with simulation-based assignment.

Layout:
    due_lcp.network       NetworkData / Link
    due_lcp.one_to_many   solver, checker, bulk reference
    due_lcp.uxsim_bridge  UXsim world builder and result extractor (optional)
    due_lcp.networks      small test networks (paper 5-node, chains, grids)
    due_lcp.lp_backend    HiGHS (default) / Gurobi (optional) LP backends

A ``many_to_one`` subpackage is planned alongside ``one_to_many``.
"""

from .network import Link, NetworkData

__version__ = "0.1.0"
__all__ = ["Link", "NetworkData", "__version__"]
