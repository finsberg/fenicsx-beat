"""Figures 2, 4 and 5 of Neic et al. 2017 from the results of reaction_eikonal_wedge.py.

python scripts/reaction_eikonal_wedge_figures.py \
    --results results-reaction-eikonal --out docs/_static
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL = {"rd": "R-D", "re+": "R-E$^+$", "re-": "R-E$^-$"}


def load(results, model, dx):
    path = results / f"{model}_dx{dx:g}.npz"
    return dict(np.load(path)) if path.is_file() else None


def histogram(ax, values, bins, title, xlabel):
    values = values[np.isfinite(values)]
    weights = np.full(values.size, 1.0 / max(values.size, 1))
    ax.hist(
        np.clip(values, bins[0], bins[-1]),
        bins=bins,
        weights=weights,
        color="0.5",
        edgecolor="k",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative frequency")


def fig2(results, ref_dx, coarse_dx, out):
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.1])
    ref = load(results, "rd", ref_dx)
    im = None
    for row, dx in enumerate((ref_dx, coarse_dx)):
        for col, model in enumerate(("rd", "re+", "re-")):
            ax = fig.add_subplot(grid[row, col])
            d = load(results, model, dx)
            if d is None or not np.isfinite(d["t_a"]).any():
                ax.set_axis_off()
                continue
            im = ax.contourf(
                d["grid_x"],
                d["grid_y"],
                d["t_a"],
                levels=np.linspace(0, 65, 27),
                cmap="turbo",
                extend="max",
            )
            ax.contour(
                d["grid_x"],
                d["grid_y"],
                d["t_a"],
                levels=np.arange(0, 70, 5),
                colors="k",
                linewidths=0.4,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(LABEL[model])
            if col == 0:
                ax.set_ylabel(f"{dx * 1000:g} µm")
    if im is not None:
        fig.colorbar(im, ax=fig.axes, location="bottom", shrink=0.4, label="$t_a$ (ms)")
    sub = grid[:, 3].subgridspec(3, 1, hspace=0.8)
    for i, model in enumerate(("rd", "re+", "re-")):
        d = load(results, model, coarse_dx)
        ax = fig.add_subplot(sub[i])
        if d is None or ref is None:
            ax.set_axis_off()
            continue
        mask = ref["t_a"] > 1.0  # relative deviation is undefined at the sources
        drel = np.abs(ref["t_a"] - d["t_a"])[mask] / ref["t_a"][mask]
        histogram(
            ax,
            drel,
            np.linspace(0, 0.1, 11),
            f"{LABEL[model]} {coarse_dx * 1000:g} µm",
            "relative deviation",
        )
    fig.savefig(out / "reaction_eikonal_fig2.png", dpi=150, bbox_inches="tight")


def fig4(results, dxs, ref_dx, coarse_dx, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    colours = dict(zip(dxs, ("tab:blue", "tab:green", "tab:red")))
    for dx in dxs:
        for model, style in (("rd", "--"), ("re+", "-")):
            d = load(results, model, dx)
            if d is not None:
                a.plot(
                    d["trace_t"],
                    d["traces"][:, 0],
                    style,
                    color=colours[dx],
                    label=f"{LABEL[model]} {dx * 1000:g} µm",
                )
    for model, dx, style, colour in (
        ("rd", ref_dx, "-", "tab:blue"),
        ("re+", ref_dx, "-", "tab:green"),
        ("re-", ref_dx, "-", "tab:red"),
        ("re+", coarse_dx, "--", "tab:green"),
        ("re-", coarse_dx, "--", "tab:red"),
    ):
        d = load(results, model, dx)
        if d is not None:
            b.plot(
                d["trace_t"],
                d["traces"][:, 0],
                style,
                color=colour,
                label=f"{LABEL[model]} {dx * 1000:g} µm",
            )
    for ax, xlim in ((a, (20, 45)), (b, (22, 32))):
        ax.set_xlim(*xlim)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("$v$ (mV) at $P_E$")
        ax.legend(fontsize=8)
    fig.savefig(out / "reaction_eikonal_fig4.png", dpi=150, bbox_inches="tight")


def fig5(results, ref_dx, coarse_dx, out):
    ref = load(results, "rd", ref_dx)
    if ref is None or np.isnan(ref["v_r50_line"]).all():
        print("No v at t_r50 (run to at least 305 ms); skipping Fig. 5")
        return
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 3)
    a, da = fig.add_subplot(grid[0, :2]), None
    da = a.twinx()
    for model, dx in (("rd", ref_dx), ("re+", coarse_dx), ("re-", coarse_dx)):
        d = load(results, model, dx)
        if d is None or np.isnan(d["v_r50_line"]).all():
            continue
        x, v = d["line_x"], d["v_r50_line"]
        a.plot(x, v, label=f"{LABEL[model]} {dx * 1000:g} µm")
        dv = np.gradient(v, x)
        da.plot(x, dv / np.nanmax(np.abs(dv)), ":", alpha=0.7)
    a.set_xlabel("x (transmural, mm)")
    a.set_ylabel("$v(x, t_{r50})$ (mV)")
    da.set_ylabel("normalised $\\partial_x v$ (dotted)")
    a.legend()
    for i, model in enumerate(("re+", "re-")):
        d = load(results, model, coarse_dx)
        ax = fig.add_subplot(grid[1, i])
        if d is None:
            ax.set_axis_off()
            continue
        histogram(
            ax,
            np.abs(ref["v_r50_grid"] - d["v_r50_grid"]).ravel(),
            np.linspace(0, 6, 13),
            f"{LABEL[model]} {coarse_dx * 1000:g} µm",
            "absolute deviation (mV)",
        )
    c = fig.add_subplot(grid[:, 2])
    for model, style in (("re+", "-"), ("re-", "--")):
        d = load(results, model, coarse_dx)
        if d is None:
            continue
        names = list(d["probe_names"])
        for probe, colour in (("endo", "tab:blue"), ("M", "tab:green")):
            c.plot(
                d["trace_t"],
                d["traces"][:, names.index(probe)],
                style,
                color=colour,
                label=f"{LABEL[model]} {probe}",
            )
    c.set_xlabel("time (ms)")
    c.set_ylabel("$v$ (mV)")
    c.legend(fontsize=8)
    fig.savefig(out / "reaction_eikonal_fig5.png", dpi=150, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results-reaction-eikonal"))
    p.add_argument("--out", type=Path, default=Path("docs/_static"))
    p.add_argument("--dxs", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    args = p.parse_args()
    ref_dx, coarse_dx = min(args.dxs), max(args.dxs)
    args.out.mkdir(parents=True, exist_ok=True)
    fig2(args.results, ref_dx, coarse_dx, args.out)
    fig4(args.results, args.dxs, ref_dx, coarse_dx, args.out)
    fig5(args.results, ref_dx, coarse_dx, args.out)


if __name__ == "__main__":
    main()
