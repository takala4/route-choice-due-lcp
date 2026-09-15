"""Like-for-like comparison of DUE-LCP against UXsim DUO and UXsim DUE.

Requires the optional dependencies:  pip install ".[uxsim,plot]"

Pipeline
  1. Build the 5-node network of Wakui-Sakai-Akamatsu (2023) and quantize
     its demand to multiples of ``deltan``.  The quantized network is the
     single input for every method.
  2. Solve DUE with the paper's sequential algorithm (`solve_sequential`).
  3. Simulate UXsim DUO (one run) and UXsim DUE (SolverDUE, day-to-day) for
     each requested seed, under the point-queue recipe.
  4. Express every result in the LCP variables (w, y, pi) and evaluate the
     same equilibrium / physical-consistency residuals with
     `verify.check_solution`.
  5. Compare total travel time, link volumes, route travel times and
     wall-clock time; write ``results.json`` and figures.

Usage
  python examples/uxsim_fair_comparison.py --out examples/output \
      --due-iters 50 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from rcdue.networks import wakui_5node
from rcdue.one_to_many import (
    check_sequential_result,
    check_solution,
    excess_travel_time,
    solve_sequential,
)
from rcdue.uxsim_bridge import (
    UXsimConfig,
    UXsimSolution,
    enumerate_routes,
    extract_solution,
    lcp_route_tt,
    lcp_route_used,
    lcp_summary,
    quantize_demand,
    run_due,
    run_duo,
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
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "stixsans",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "legend.frameon": False,
    }
)
METHOD_COLORS = {"DUE-LCP": "red", "UXsim DUO": "blue", "UXsim DUE": "green"}


def residual_dict(rep, excess: float) -> dict:
    return {
        "equilibrium": asdict(rep.equilibrium),
        "physical": asdict(rep.physical),
        "equilibrium_max": rep.equilibrium.max_violation(),
        "physical_max": rep.physical.max_violation(),
        "excess_tt_per_vehicle": excess,
    }


def route_label(route) -> str:
    nodes = [route[0][0]] + [head for _, head in route]
    return "-".join(str(n) for n in nodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="DUE-LCP vs UXsim fair comparison")
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--deltan", type=int, default=5)
    parser.add_argument("--reaction-time", type=float, default=0.02)
    parser.add_argument("--due-iters", type=int, default=50)
    parser.add_argument("--due-routes", type=int, default=8)
    parser.add_argument("--due-swap-prob", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--backend", default="auto", help="LP backend for DUE-LCP")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else Path(__file__).parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = UXsimConfig(
        deltan=args.deltan,
        reaction_time=args.reaction_time,
        due_max_iter=args.due_iters,
        due_n_routes=args.due_routes,
        due_swap_prob=args.due_swap_prob,
    )

    # ---- 1. common input
    net_raw = wakui_5node()
    net = quantize_demand(net_raw, cfg.deltan)
    print(
        f"[input] total demand raw={net_raw.total_demand():.1f}"
        f" quantized={net.total_demand():.1f} (deltan={cfg.deltan})"
    )

    # ---- 2. DUE-LCP (paper algorithm)
    t0 = time.perf_counter()
    res = solve_sequential(net, eps=args.eps, backend=args.backend)
    t_lcp = time.perf_counter() - t0
    rep_lcp = check_sequential_result(net, res, tol=1e-6)
    lcp = lcp_summary(net, res)
    print(
        f"[DUE-LCP] t={t_lcp:.3f}s TTT={lcp['ttt']:.1f}"
        f" eq={rep_lcp.equilibrium.max_violation():.1e}"
        f" phys={rep_lcp.physical.max_violation():.1e}"
    )

    # ---- 3. UXsim DUO / DUE
    runs: dict[str, list[UXsimSolution]] = {"UXsim DUO": [], "UXsim DUE": []}
    due_histories: list[dict] = []
    for seed in args.seeds:
        W, t = run_duo(net, cfg, seed=seed)
        sol = extract_solution(W, net, elapsed=t)
        runs["UXsim DUO"].append(sol)
        print(
            f"[UXsim DUO seed={seed}] t={t:.2f}s TTT={sol.ttt:.1f}"
            f" unfinished={sol.n_unfinished} entry_wait={sol.entry_wait_mean:.3f}"
        )
        W, solver, t = run_due(net, cfg, seed=seed, verbose=args.verbose)
        sol = extract_solution(W, net, elapsed=t)
        runs["UXsim DUE"].append(sol)
        due_histories.append(
            {
                "seed": seed,
                "ttts": [float(x) for x in solver.ttts],
                "t_gaps": [float(x) for x in solver.t_gaps],
            }
        )
        print(
            f"[UXsim DUE seed={seed}] t={t:.2f}s ({cfg.due_max_iter} iters)"
            f" TTT={sol.ttt:.1f} last time-gap/veh={solver.t_gaps[-1]:.3f}"
        )

    # ---- 4. same residuals for every method
    residuals: dict[str, list[dict]] = {
        "DUE-LCP": [
            residual_dict(
                rep_lcp, excess_travel_time(net, res.w, res.y, res.pi, res.pi_nodes)
            )
        ]
    }
    for method, sols in runs.items():
        residuals[method] = []
        for sol in sols:
            rep = check_solution(net, sol.w, sol.y, sol.pi, sol.pi_nodes, tol=1e-6)
            excess = excess_travel_time(net, sol.w, sol.y, sol.pi, sol.pi_nodes)
            residuals[method].append(residual_dict(rep, excess))

    # ---- 5. summary
    summary = {
        "config": asdict(cfg),
        "seeds": args.seeds,
        "eps": args.eps,
        "demand_total": net.total_demand(),
        "methods": {
            "DUE-LCP": {
                "ttt": [lcp["ttt"]],
                "elapsed": [t_lcp],
                "link_volumes": [
                    {f"{k[0]}-{k[1]}": v for k, v in lcp["link_volumes"].items()}
                ],
            }
        },
        "residuals": residuals,
        "due_histories": due_histories,
    }
    for method, sols in runs.items():
        summary["methods"][method] = {
            "ttt": [s.ttt for s in sols],
            "elapsed": [s.elapsed for s in sols],
            "unfinished": [s.n_unfinished for s in sols],
            "entry_wait_mean": [s.entry_wait_mean for s in sols],
            "link_volumes": [
                {f"{k[0]}-{k[1]}": v for k, v in s.link_volumes.items()} for s in sols
            ],
        }
    (out_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== summary (mean over seeds) ===")
    print(
        f"{'method':10s} {'TTT':>10s} {'time[s]':>9s} {'excess/veh':>11s}"
        f" {'eq gap':>9s} {'phys gap':>9s}"
    )
    for method in ("DUE-LCP", "UXsim DUO", "UXsim DUE"):
        m = summary["methods"][method]
        eq = np.mean([r["equilibrium_max"] for r in residuals[method]])
        ph = np.mean([r["physical_max"] for r in residuals[method]])
        ex = np.mean([r["excess_tt_per_vehicle"] for r in residuals[method]])
        print(
            f"{method:10s} {np.mean(m['ttt']):10.1f} {np.mean(m['elapsed']):9.2f}"
            f" {ex:11.3f} {eq:9.2e} {ph:9.2e}"
        )
    print("\nlink volumes (vehicles):")
    keys = [link.key for link in net.links]
    print(f"{'link':8s}" + "".join(f"{m:>11s}" for m in METHOD_COLORS))
    for key in keys:
        row = f"({key[0]},{key[1]})".ljust(8)
        row += f"{lcp['link_volumes'][key]:11.1f}"
        for method in ("UXsim DUO", "UXsim DUE"):
            row += f"{np.mean([s.link_volumes[key] for s in runs[method]]):11.1f}"
        print(row)

    # ---- figures
    fig_link_volumes(net, lcp, runs, out_dir)
    fig_ttt(summary, out_dir)
    fig_residuals(residuals, out_dir)
    fig_route_equilibrium(net, res, runs, out_dir)
    fig_due_convergence(due_histories, lcp["ttt"], out_dir)
    print(f"\nwritten to {out_dir}")


def fig_link_volumes(net, lcp, runs, out_dir: Path) -> None:
    keys = [link.key for link in net.links]
    x = np.arange(len(keys))
    bw = 0.27
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.bar(
        x - bw,
        [lcp["link_volumes"][k] for k in keys],
        bw,
        label="DUE-LCP",
        color=COLORS["red"],
    )
    for off, method in ((0.0, "UXsim DUO"), (bw, "UXsim DUE")):
        vals = np.array([[s.link_volumes[k] for k in keys] for s in runs[method]])
        ax.bar(
            x + off,
            vals.mean(axis=0),
            bw,
            label=method,
            color=COLORS[METHOD_COLORS[method]],
            yerr=vals.std(axis=0) if len(vals) > 1 else None,
            capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"({k[0]},{k[1]})" for k in keys])
    ax.set_xlabel("Link")
    ax.set_ylabel("Vehicles")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "link_volumes.pdf")
    plt.close(fig)


def fig_ttt(summary, out_dir: Path) -> None:
    methods = list(METHOD_COLORS)
    means = [np.mean(summary["methods"][m]["ttt"]) for m in methods]
    stds = [np.std(summary["methods"][m]["ttt"]) for m in methods]
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.bar(
        methods,
        means,
        color=[COLORS[METHOD_COLORS[m]] for m in methods],
        yerr=stds,
        capsize=3,
    )
    ax.set_ylabel("Total travel time")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "total_travel_time.pdf")
    plt.close(fig)


def fig_residuals(residuals, out_dir: Path) -> None:
    fields = [
        "r_w_negative",
        "r_y_negative",
        "conservation",
        "w_complementarity",
        "y_complementarity",
        "fifo",
    ]
    labels = [
        r"$R_w < 0$",
        r"$R_y < 0$",
        "conservation",
        r"$w \cdot R_w$",
        r"$y \cdot R_y$",
        "FIFO",
    ]
    methods = list(METHOD_COLORS)
    x = np.arange(len(fields))
    bw = 0.27
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for i, m in enumerate(methods):
        vals = np.array([[r["equilibrium"][f] for f in fields] for r in residuals[m]])
        ax.bar(
            x + (i - 1) * bw,
            np.maximum(vals.mean(axis=0), 1e-16),
            bw,
            label=m,
            color=COLORS[METHOD_COLORS[m]],
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Max violation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "equilibrium_residuals.pdf")
    plt.close(fig)


def fig_route_equilibrium(net, res, runs, out_dir: Path) -> None:
    """Paper-style (Figs 3, 4): per OD, route travel times by departure step
    with a band below marking the steps on which the route carries inflow.
    One panel per method; UXsim panels use the first seed."""
    route_colors = ["red", "blue", "green", "orange", "purple", "cyan", "gray"]
    kappa = np.arange(1, net.T + 1)
    methods = list(METHOD_COLORS)
    for d, routes in enumerate_routes(net).items():
        fig = plt.figure(figsize=(7.0, 3.4), constrained_layout=True)
        gs = fig.add_gridspec(
            2, len(methods), height_ratios=[4, 0.35 * len(routes)], hspace=0.08
        )
        ax_tt = [fig.add_subplot(gs[0, j]) for j in range(len(methods))]
        ax_band = [
            fig.add_subplot(gs[1, j], sharex=ax_tt[j]) for j in range(len(methods))
        ]
        ymax = 0.0
        for j, method in enumerate(methods):
            for r, route in enumerate(routes):
                color = COLORS[route_colors[r % len(route_colors)]]
                if method == "DUE-LCP":
                    tt = lcp_route_tt(net, res, route)
                    used = lcp_route_used(net, res, route)
                else:
                    sol = runs[method][0]
                    tt = sol.route_tt_all[route]
                    used = sol.route_counts.get(route, np.zeros(net.T)) > 0
                ymax = max(ymax, float(np.nanmax(tt)))
                ax_tt[j].plot(
                    kappa, tt, color=color, label=f"route {r + 1}: {route_label(route)}"
                )
                spans = [(k + 1 - 0.5, 1.0) for k in range(net.T) if used[k]]
                ax_band[j].broken_barh(spans, (r + 0.15, 0.7), color=color)
            ax_tt[j].set_title(method)
            ax_tt[j].set_xlim(1, net.T)
            ax_tt[j].tick_params(labelbottom=False)
            ax_band[j].set_ylim(0, len(routes))
            ax_band[j].set_yticks([r + 0.5 for r in range(len(routes))])
            ax_band[j].set_yticklabels([str(r + 1) for r in range(len(routes))])
            ax_band[j].set_xlabel(r"Departure step $\kappa$")
            ax_band[j].spines["left"].set_visible(False)
            ax_band[j].tick_params(axis="y", length=0)
        for ax in ax_tt:
            ax.set_ylim(0, ymax * 1.08)
        ax_tt[0].set_ylabel("Route travel time")
        ax_band[0].set_ylabel("inflow", fontsize=10)
        ax_tt[0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"OD {net.origin} → {d}", fontsize=12)
        fig.savefig(out_dir / f"route_equilibrium_od{net.origin}-{d}.pdf")
        plt.close(fig)


def fig_due_convergence(histories, ttt_lcp: float, out_dir: Path) -> None:
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.625))
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
    fig.tight_layout()
    fig.savefig(out_dir / "due_convergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
