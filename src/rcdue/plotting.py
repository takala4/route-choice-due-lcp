"""Matplotlib figures for solutions and comparisons (optional dependency).

All functions return the created `Figure` so they display inline in
notebooks; call ``fig.savefig(...)`` to write files.

Figures:
    plot_route_equilibrium      paper Figs 3-4: route travel times by departure
                                step with bands marking steps that carry inflow
    plot_cumulative_curves      paper Fig 2: cumulative inflow / outflow at
                                each bottleneck against clock time
    plot_queue_delays           w_ij by departure step
    plot_link_volumes, plot_total_travel_time, plot_residuals,
    plot_due_convergence        comparison figures for the UXsim pipeline
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .network import NetworkData
from .uxsim_bridge import (
    RouteKey,
    enumerate_routes,
    lcp_route_tt,
    lcp_route_used,
)

COLORS = {
    "red": "#FF4B00",
    "blue": "#005AFF",
    "green": "#03AF7A",
    "cyan": "#4DC4FF",
    "orange": "#F6AA00",
    "purple": "#990099",
    "gray": "#84919E",
    "black": "#000000",
}
CYCLE = ["red", "blue", "green", "orange", "purple", "cyan", "gray", "black"]
METHOD_COLORS = {"DUE-LCP": "red", "UXsim DUO": "blue", "UXsim DUE": "green"}


def apply_style() -> None:
    """Colour-universal palette, inward ticks, no top / right spines."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": False,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=[COLORS[c] for c in CYCLE]),
        }
    )


def route_label(route: RouteKey) -> str:
    nodes = [route[0][0]] + [head for _, head in route]
    return "-".join(str(n) for n in nodes)


# ---------------------------------------------------------------------------
# single-solution figures
# ---------------------------------------------------------------------------
def _route_panel(ax_tt, ax_band, kappa, routes, tt_of, used_of, title: str) -> float:
    ymax = 0.0
    for r, route in enumerate(routes):
        color = COLORS[CYCLE[r % len(CYCLE)]]
        tt = tt_of(route)
        used = used_of(route)
        ymax = max(ymax, float(np.nanmax(tt)))
        ax_tt.plot(kappa, tt, color=color, label=f"route {r + 1}: {route_label(route)}")
        spans = [(k + 0.5, 1.0) for k in range(len(kappa)) if used[k]]
        ax_band.broken_barh(spans, (r + 0.15, 0.7), color=color)
    ax_tt.set_title(title)
    ax_tt.set_xlim(1, len(kappa))
    ax_tt.tick_params(labelbottom=False)
    ax_band.set_ylim(0, len(routes))
    ax_band.set_yticks([r + 0.5 for r in range(len(routes))])
    ax_band.set_yticklabels([str(r + 1) for r in range(len(routes))])
    ax_band.set_xlabel(r"Departure step $\kappa$")
    ax_band.spines["left"].set_visible(False)
    ax_band.tick_params(axis="y", length=0)
    return ymax


def plot_route_equilibrium(
    net: NetworkData,
    res,
    destination: int,
    *,
    uxsim_runs: dict[str, list] | None = None,
) -> plt.Figure:
    """Route travel times by departure step with inflow bands (paper Figs 3-4).

    Args:
        net: Network.
        res: `SequentialResult` of the LCP solver.
        destination: Destination node whose routes are drawn.
        uxsim_runs: Optional ``{method: [UXsimSolution, ...]}``; one extra
            panel per method is drawn from the first solution in each list.
    """
    routes = enumerate_routes(net)[destination]
    panels: list[tuple[str, object]] = [("DUE-LCP", res)]
    if uxsim_runs:
        panels += [(m, sols[0]) for m, sols in uxsim_runs.items()]
    kappa = np.arange(1, net.T + 1)
    fig = plt.figure(figsize=(2.4 * len(panels) + 0.6, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(2, len(panels), height_ratios=[4, 0.35 * len(routes)])
    ax_tt = [fig.add_subplot(gs[0, j]) for j in range(len(panels))]
    ax_band = [fig.add_subplot(gs[1, j], sharex=ax_tt[j]) for j in range(len(panels))]
    ymax = 0.0
    for j, (name, sol) in enumerate(panels):
        if name == "DUE-LCP":
            ymax = max(
                ymax,
                _route_panel(
                    ax_tt[j],
                    ax_band[j],
                    kappa,
                    routes,
                    lambda rt: lcp_route_tt(net, res, rt),
                    lambda rt: lcp_route_used(net, res, rt),
                    name,
                ),
            )
        else:
            ymax = max(
                ymax,
                _route_panel(
                    ax_tt[j],
                    ax_band[j],
                    kappa,
                    routes,
                    lambda rt, s=sol: s.route_tt_all[rt],
                    lambda rt, s=sol: s.route_counts.get(rt, np.zeros(net.T)) > 0,
                    name,
                ),
            )
    for ax in ax_tt:
        ax.set_ylim(0, ymax * 1.08)
    ax_tt[0].set_ylabel("Route travel time")
    ax_band[0].set_ylabel("inflow", fontsize=10)
    ax_tt[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"OD {net.origin} → {destination}")
    return fig


def cumulative_curves(net: NetworkData, res, link_key: tuple[int, int]):
    """Cumulative inflow and outflow at the bottleneck of one link.

    Users departing at step k (clock time (k + 1) dt) enter link (i, j) at
    (k + 1) dt + pi_i[k] and leave its bottleneck at entry + tau + w_ij[k].
    Returns (t_in, N_in, t_out, N_out) as step functions.
    """
    ij = net.link_index(*link_key)
    link = net.links[ij]
    K = net.T
    t_dep = (np.arange(K) + 1) * net.dt
    pi_tail = (
        np.zeros(K) if link.tail == net.origin else res.pi[res.pi_idx_of[link.tail]]
    )
    t_in = t_dep + pi_tail
    t_out = t_in + link.tau_hat + res.w[ij]
    n = res.y[ij] * net.dt
    order_in = np.argsort(t_in, kind="stable")
    order_out = np.argsort(t_out, kind="stable")
    return (
        t_in[order_in],
        np.cumsum(n[order_in]),
        t_out[order_out],
        np.cumsum(n[order_out]),
    )


def plot_cumulative_curves(
    net: NetworkData, res, links: Sequence[tuple[int, int]] | None = None
) -> plt.Figure:
    """Cumulative inflow / outflow at every bottleneck vs clock time (paper Fig 2)."""
    keys = list(links) if links is not None else [ln.key for ln in net.links]
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(2.4 * ncol + 0.4, 2.3 * nrow + 0.3),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, key in zip(axes.flat, keys, strict=False):
        t_in, n_in, t_out, n_out = cumulative_curves(net, res, key)
        ax.step(t_in, n_in, where="post", color=COLORS["red"], label="inflow")
        ax.step(t_out, n_out, where="post", color=COLORS["blue"], label="outflow")
        ax.set_title(f"link {key}")
        ax.set_xlabel("Clock time")
        ax.set_ylabel("Cumulative vehicles")
        ax.set_ylim(bottom=0)
    for ax in list(axes.flat)[len(keys) :]:
        ax.axis("off")
    axes.flat[0].legend()
    return fig


def plot_queue_delays(net: NetworkData, res) -> plt.Figure:
    """Queue delay w_ij by departure step for every link."""
    kappa = np.arange(1, net.T + 1)
    fig, ax = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)
    for ij, link in enumerate(net.links):
        ax.plot(kappa, res.w[ij], label=f"link {link.key}")
    ax.set_xlabel(r"Departure step $\kappa$")
    ax.set_ylabel("Queue delay $w$")
    ax.set_xlim(1, net.T)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# comparison figures (UXsim pipeline)
# ---------------------------------------------------------------------------
def plot_link_volumes(net: NetworkData, lcp_volumes: dict, runs: dict[str, list]):
    keys = [ln.key for ln in net.links]
    x = np.arange(len(keys))
    bw = 0.8 / (1 + len(runs))
    fig, ax = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)
    ax.bar(
        x - bw, [lcp_volumes[k] for k in keys], bw, label="DUE-LCP", color=COLORS["red"]
    )
    for i, (method, sols) in enumerate(runs.items()):
        vals = np.array([[s.link_volumes[k] for k in keys] for s in sols])
        ax.bar(
            x + i * bw,
            vals.mean(axis=0),
            bw,
            label=method,
            color=COLORS[METHOD_COLORS.get(method, CYCLE[i + 1])],
            yerr=vals.std(axis=0) if len(vals) > 1 else None,
            capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"({k[0]},{k[1]})" for k in keys])
    ax.set_xlabel("Link")
    ax.set_ylabel("Vehicles")
    ax.set_ylim(bottom=0)
    ax.legend()
    return fig


def plot_total_travel_time(ttt: dict[str, Sequence[float]]) -> plt.Figure:
    """Bar chart of total travel time per method (mean and std over seeds)."""
    methods = list(ttt)
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    ax.bar(
        methods,
        [np.mean(ttt[m]) for m in methods],
        color=[COLORS[METHOD_COLORS.get(m, "gray")] for m in methods],
        yerr=[np.std(ttt[m]) for m in methods],
        capsize=3,
    )
    ax.set_ylabel("Total travel time")
    ax.set_ylim(bottom=0)
    return fig


def plot_residuals(reports: dict[str, list]) -> plt.Figure:
    """Equilibrium residuals per method (log scale), from `CheckReport` lists."""
    fields = [
        ("r_w_negative", r"$R_w < 0$"),
        ("r_y_negative", r"$R_y < 0$"),
        ("conservation", "conservation"),
        ("w_complementarity", r"$w \cdot R_w$"),
        ("y_complementarity", r"$y \cdot R_y$"),
        ("fifo", "FIFO"),
    ]
    methods = list(reports)
    x = np.arange(len(fields))
    bw = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.0, 2.8), constrained_layout=True)
    for i, m in enumerate(methods):
        vals = np.array(
            [[getattr(r.equilibrium, f) for f, _ in fields] for r in reports[m]]
        )
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * bw,
            np.maximum(vals.mean(axis=0), 1e-16),
            bw,
            label=m,
            color=COLORS[METHOD_COLORS.get(m, CYCLE[i])],
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in fields])
    ax.set_ylabel("Max violation")
    ax.legend()
    return fig


def plot_due_convergence(histories: list[dict], ttt_lcp: float) -> plt.Figure:
    """UXsim SolverDUE day-to-day histories: total travel time and its own time gap."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for h in histories:
        it = np.arange(len(h["ttts"]))
        axes[0].plot(it, h["ttts"], color=COLORS["green"], alpha=0.8)
        axes[1].plot(it, h["t_gaps"], color=COLORS["green"], alpha=0.8)
    axes[0].axhline(ttt_lcp, color=COLORS["red"], linestyle="--", label="DUE-LCP")
    axes[0].set_xlabel("Day-to-day iteration")
    axes[0].set_ylabel("Total travel time")
    axes[0].legend()
    axes[1].set_xlabel("Day-to-day iteration")
    axes[1].set_ylabel("Time gap per vehicle")
    axes[1].set_ylim(bottom=0)
    return fig
