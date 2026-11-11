"""Figures 2, 4, 5 and 7 of Neic et al. 2017 from the results of reaction_eikonal_wedge.py.

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
    # Values above the range are clipped into the last bin
    ax.set_xlabel(f"{xlabel}\n(last bin: $\\geq {bins[-2]:g}$, overflow)")
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
        fig.colorbar(
            im,
            ax=fig.axes,
            location="bottom",
            shrink=0.4,
            label="$t_a$ (ms)",
            ticks=range(0, 70, 10),
        )
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
    grid = fig.add_gridspec(2, 3, wspace=0.55, hspace=0.5)
    a, da = fig.add_subplot(grid[0, :2]), None
    da = a.twinx()
    for model, dx in (("rd", ref_dx), ("re+", coarse_dx), ("re-", coarse_dx)):
        d = load(results, model, dx)
        if d is None or np.isnan(d["v_r50_line"]).all():
            continue
        x, v = d["line_x"], d["v_r50_line"]
        a.plot(x, v, label=f"{LABEL[model]} {dx * 1000:g} µm")
        dv = np.gradient(v, x)
        da.plot(x, dv, ":", alpha=0.7)
    a.set_xlabel("x (transmural, mm)")
    a.set_ylabel("$v(x, t_{r50})$ (mV)")
    da.set_ylabel("$\\partial_x v$ (mV/mm, dotted)")
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
    for model, dx, style in (
        ("rd", ref_dx, ":"),
        ("re+", coarse_dx, "-"),
        ("re-", coarse_dx, "--"),
    ):
        d = load(results, model, dx)
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


def load_phie(results, model, dx):
    path = results / f"phie_{model}_dx{dx:g}.npz"
    return dict(np.load(path)) if path.is_file() else None


def check_electrode(results, models, dxs):
    """P_E is the mid-plane grid point [50, 100]: compare phie there with the electrogram"""
    for model in models:
        for dx in dxs:
            d, run = load_phie(results, model, dx), load(results, model, dx)
            if d is None or run is None or not run["electrogram"].size:
                continue
            for t in (40, 320):
                i = int(np.argmin(np.abs(run["electrogram_t"] - t)))
                print(
                    f"P_E check {model} {dx * 1000:g} um t={t} ms: phie[50, 100] = "
                    f"{d[f'phie_t{t}'][50, 100]:.3f} mV, electrogram({run['electrogram_t'][i]:.1f}"
                    f" ms) = {run['electrogram'][i]:.3f} mV",
                )


def fig7(results, ref_dx, coarse_dx, out):
    panels = [
        ("rd", ref_dx, "R-D"),
        ("re+", ref_dx, "R-E$^+$"),
        ("re+", coarse_dx, f"R-E$^+$ {coarse_dx * 1000:g} µm"),
        ("re-", ref_dx, "R-E$^-$"),
    ]
    phie = {(m, dx): load_phie(results, m, dx) for m, dx, _ in panels}
    runs = {(m, dx): load(results, m, dx) for m, dx, _ in panels}
    ref = phie[("rd", ref_dx)]
    fig = plt.figure(figsize=(17, 9))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.12)
    left = outer[0].subgridspec(4, 4, height_ratios=[1, 0.55, 1, 0.55], hspace=1.0, wspace=0.1)
    for block, t in enumerate((40, 320)):
        key = f"phie_t{t}"
        values = ref[key] if ref is not None and key in ref else None
        lo, hi = np.nanpercentile(values, [1, 99]) if values is not None else (-10.0, 10.0)
        levels = np.linspace(lo, hi, 26)
        cs = None
        for col, (m, dx, label) in enumerate(panels):
            ax = fig.add_subplot(left[2 * block, col])
            ax.set_xticks([])
            ax.set_yticks([])
            d = phie[(m, dx)]
            if d is None or key not in d:
                ax.set_title(f"{label}: missing", fontsize=9)
                continue
            cs = ax.contourf(
                d["grid_x"],
                d["grid_y"],
                d[key],
                levels=levels,
                cmap="jet",
                extend="both",
            )
            ax.set_aspect("equal")
            ax.set_title(f"{label}, t = {t} ms", fontsize=10)
        if cs is not None:
            fig.colorbar(cs, ax=fig.axes[-4:], label=r"$\phi_e$ [mV]", shrink=0.9)
        for col, m in enumerate(("re+", "re-")):
            ax = fig.add_subplot(left[2 * block + 1, 2 * col : 2 * col + 2])
            d = phie[(m, ref_dx)]
            if values is None or d is None or key not in d:
                ax.set_axis_off()
                continue
            deviation = np.abs(d[key] - values).ravel()
            histogram(
                ax,
                deviation,
                np.linspace(0, 3, 13),
                f"{LABEL[m]}, t = {t} ms",
                "absolute deviation [mV]",
            )
            if col == 1:
                ax.set_ylabel("")
            q90, q95 = np.nanquantile(deviation, [0.9, 0.95])
            print(f"fig7 t={t} ms {m}: 90% < {q90:.3f} mV, 95% < {q95:.3f} mV")
    right = outer[1].subgridspec(4, 1, hspace=0.45)
    rd = runs[("rd", ref_dx)]
    first = None
    for row, (m, dx, label) in enumerate(panels):
        ax = fig.add_subplot(right[row], sharex=first, sharey=first)
        first = first or ax
        if rd is not None and m != "rd" and rd["electrogram"].size:
            ax.plot(rd["electrogram_t"], rd["electrogram"], color="0.7", lw=2.5, label="R-D")
        d = runs[(m, dx)]
        if d is not None and d["electrogram"].size:
            ax.plot(d["electrogram_t"], d["electrogram"], lw=1, label=label)
            print(
                f"fig7 electrogram {label}: min {d['electrogram'].min():.2f}, "
                f"max {d['electrogram'].max():.2f} mV",
            )
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlim(0, 400)
        ax.set_ylabel(r"$\phi_e(P_E)$ [mV]")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("t [ms]")
    fig.savefig(out / "reaction_eikonal_fig7.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    check_electrode(results, ("rd", "re+", "re-"), sorted({dx for _, dx, _ in panels}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results-reaction-eikonal-bath"))
    p.add_argument("--out", type=Path, default=Path("docs/_static"))
    p.add_argument("--dxs", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    args = p.parse_args()
    ref_dx, coarse_dx = min(args.dxs), max(args.dxs)
    args.out.mkdir(parents=True, exist_ok=True)
    fig2(args.results, ref_dx, coarse_dx, args.out)
    fig4(args.results, args.dxs, ref_dx, coarse_dx, args.out)
    fig5(args.results, ref_dx, coarse_dx, args.out)
    fig7(args.results, ref_dx, coarse_dx, args.out)


if __name__ == "__main__":
    main()
