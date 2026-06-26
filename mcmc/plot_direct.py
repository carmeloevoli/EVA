"""Plotting routines for the direct-spectrum MCMC fit.

Reads the ``.npz`` produced by ``mcmc_direct.py`` and makes the posterior
reconstruction of every species spectrum (median + 68%/95% bands) over the
data, plus the break and slope summary plots.  The corner plot lives separately
in ``plot_corner.py`` (it needs the default matplotlib style, not EVA.mplstyle).

Run with:  python plot_direct.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import kiss_reader
from mcmc_direct import (OUTPUT, SPECIES, Z, E0_P, E_MIN_P, MAX_ENERGY,
                         SCALE_EXPERIMENTS, SCALE_START, species_model,
                         RB_INDEX, SLOPE_INDEX)

plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "EVA.mplstyle"))

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PLOT_EMAX = 3.0e6   # GeV, upper energy for the model bands (extrapolated beyond the fit)
PLOT_SLOPE = 2.6    # the spectra are shown multiplied by E**PLOT_SLOPE
LHAASO_PLOT_EMAX = 1.2e7
EXTRAPOLATION_PLOT_EMIN = 1.0e4
EXTRAPOLATION_START = 3.0e5
LHAASO_EXPERIMENT = "LHAASO_QGSJET-II-04"
SPECIES_COLORS = {"H": "tab:blue", "He": "tab:red", "C": "tab:green",
                  "O": "tab:orange", "Fe": "tab:purple"}
EXP_MARKERS = {"DAMPE": "o", "CALET": "s", "CREAM": "^", "ISS-CREAM": "D"}


def savefig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


def plot_breaks(samples):
    """Overlay the marginalized posteriors of the three rigidity breaks R_b
    (proton, helium, nuclei) to show whether they differ significantly."""
    groups = [("H", "tab:blue", "p"),
              ("He", "tab:red", "He"),
              ("heavy", "tab:green", "nuclei")]

    chains = {g: samples[:, RB_INDEX[g]] for g, _, _ in groups}
    lo = min(c.min() for c in chains.values())
    hi = max(c.max() for c in chains.values())
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(11, 8))
    for g, color, label in groups:
        c = chains[g]
        q16, q50, q84 = np.percentile(c, [16, 50, 84])
        ax.hist(c, bins=bins, density=True, histtype="stepfilled",
                color=color, alpha=0.30)
        ax.hist(c, bins=bins, density=True, histtype="step",
                color=color, lw=2.5,
                label=rf"{label}: ${q50:.2f}^{{+{q84 - q50:.2f}}}_{{-{q50 - q16:.2f}}}$")
        ax.axvline(q50, color=color, lw=1.5, ls="--")

    ax.set_xlabel(r"$\log_{10}(R_b\,/\,{\rm GV})$")
    ax.set_ylabel("posterior density")
    ax.set_title("Rigidity breaks")
    ax.legend(fontsize=20)
    savefig(fig, "EVA_mcmc_direct_breaks.pdf")


GROUP_META = [("H", "tab:blue", "p"),
              ("He", "tab:red", "He"),
              ("heavy", "tab:green", "nuclei")]


def _kde_levels(density, fractions):
    """Density thresholds enclosing the given probability fractions (HPD)."""
    flat = np.sort(density.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return [flat[np.searchsorted(csum, f)] for f in fractions]


def _overlay_kde_2d(ax, series, cred=(0.68, 0.95)):
    """Overlay filled KDE credible contours for several (x, y) sample sets.

    ``series`` is a list of (color, x, y); a common robust axis range is used.
    """
    from scipy.stats import gaussian_kde

    clean_series = []
    for color, x, y in series:
        finite = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x[finite], dtype=float)
        y = np.asarray(y[finite], dtype=float)
        if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
            raise ValueError("KDE input needs at least three varying finite samples")
        clean_series.append((color, x, y))
    series = clean_series

    rng = np.random.default_rng(0)
    xs = np.concatenate([x for _, x, _ in series])
    ys = np.concatenate([y for _, _, y in series])
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    px, py = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
    xg = np.linspace(x0 - px, x1 + px, 200)
    yg = np.linspace(y0 - py, y1 + py, 200)
    XX, YY = np.meshgrid(xg, yg)
    grid = np.vstack([XX.ravel(), YY.ravel()])

    for color, x, y in series:
        sub = rng.choice(len(x), size=min(6000, len(x)), replace=False)
        values = np.vstack([x[sub], y[sub]])
        # NumPy's Accelerate-backed weighted covariance can emit spurious
        # floating-point warnings here even though the covariance is finite.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            kde = gaussian_kde(values)
        dens = kde(grid).reshape(XX.shape)
        if not np.all(np.isfinite(dens)):
            raise FloatingPointError("KDE produced non-finite density values")
        levels = sorted(_kde_levels(dens, cred))       # ascending for contour()
        ax.contourf(XX, YY, dens, levels=[levels[0], dens.max()],
                    colors=[color], alpha=0.18)
        ax.contour(XX, YY, dens, levels=levels, colors=color, linewidths=2.2)
        ax.plot(np.median(x), np.median(y), marker="o", color=color,
                ms=10, markeredgecolor="k", zorder=5)
    ax.set_xlim(xg[0], xg[-1])
    ax.set_ylim(yg[0], yg[-1])


def plot_slopes_2d(samples):
    """Overlay the 2D (alpha1, alpha2) posteriors of the three slope groups."""
    fig, ax = plt.subplots(figsize=(9.5, 9))
    series = [(c, samples[:, SLOPE_INDEX[g]], samples[:, SLOPE_INDEX[g] + 1])
              for g, c, _ in GROUP_META]
    _overlay_kde_2d(ax, series)
    ax.set_xlabel(r"$\alpha_1$ (low energy)")
    ax.set_ylabel(r"$\alpha_2$ (high energy)")
    ax.set_title("Spectral slopes (68\\%, 95\\% C.L.)")
    handles = [Line2D([], [], color=c, lw=3, label=l) for _, c, l in GROUP_META]
    ax.legend(handles=handles, fontsize=20)
    savefig(fig, "EVA_mcmc_direct_slopes.pdf")


def plot_alpha1_break_2d(samples):
    """Overlay the 2D (alpha1, log10 R_b) posteriors of the three groups."""
    fig, ax = plt.subplots(figsize=(9.5, 9))
    series = [(c, samples[:, SLOPE_INDEX[g]], samples[:, RB_INDEX[g]])
              for g, c, _ in GROUP_META]
    _overlay_kde_2d(ax, series)
    ax.set_xlabel(r"$\alpha_1$ (low energy)")
    ax.set_ylabel(r"$\log_{10}(R_b\,/\,{\rm GV})$")
    ax.set_title("Low-energy slope vs rigidity break (68\\%, 95\\% C.L.)")
    handles = [Line2D([], [], color=c, lw=3, label=l) for _, c, l in GROUP_META]
    ax.legend(handles=handles, fontsize=20)
    savefig(fig, "EVA_mcmc_direct_alpha1_break.pdf")


def plot_reconstruction(d):
    samples = d["samples"]
    E, y, exp, sp = d["data_E"], d["data_y"], d["data_exp"], d["data_sp"]
    stat_lo, stat_up = d["data_stat_lo"], d["data_stat_up"]
    sys_lo, sys_up = d["data_sys_lo"], d["data_sys_up"]

    # Median energy-scale factor per experiment, applied to displayed data.
    median_scale = {name: np.median(samples[:, SCALE_START + i])
                    for i, name in enumerate(SCALE_EXPERIMENTS)}

    idx = np.random.default_rng(0).integers(len(samples), size=2000)
    used_exp = list(dict.fromkeys(exp))  # unique, in order

    fig, ax = plt.subplots(figsize=(13.0, 9.5))

    # Colour encodes species, marker encodes experiment.
    for s in SPECIES:
        color = SPECIES_COLORS.get(s, "k")
        Egrid = np.logspace(np.log10(Z[s] * E_MIN_P), np.log10(PLOT_EMAX), 300)
        scale = Egrid ** PLOT_SLOPE

        curves = np.array([species_model(samples[i], s, Egrid) for i in idx])
        lo95, lo68, med, up68, up95 = np.percentile(
            curves, [2.5, 16, 50, 84, 97.5], axis=0)

        ax.fill_between(Egrid, lo95 * scale, up95 * scale, color=color, alpha=0.08)
        ax.fill_between(Egrid, lo68 * scale, up68 * scale, color=color, alpha=0.18)
        ax.plot(Egrid, med * scale, color=color, lw=3.0)

        for name in used_exp:
            m = (sp == s) & (exp == name)
            if not m.any():
                continue
            f = median_scale.get(name, 1.0)
            Em = f * E[m]
            yv = (y[m] / f) * Em ** PLOT_SLOPE
            sys_bottom = yv - (sys_lo[m] / f) * Em ** PLOT_SLOPE
            sys_top = yv + (sys_up[m] / f) * Em ** PLOT_SLOPE
            # Systematic uncertainty: dotted interval with short end ticks.
            ax.vlines(
                Em, sys_bottom, sys_top,
                color=color, linewidth=1.2, linestyles=":",
                alpha=0.9, zorder=2,
            )
            ax.plot(
                Em, sys_bottom, ls="", marker="_", markersize=7,
                markeredgewidth=1.2, color=color, zorder=2,
            )
            ax.plot(
                Em, sys_top, ls="", marker="_", markersize=7,
                markeredgewidth=1.2, color=color, zorder=2,
            )
            # Statistical uncertainty: solid capped bar and open data marker.
            ax.errorbar(Em, yv,
                        yerr=[(stat_lo[m] / f) * Em ** PLOT_SLOPE, (stat_up[m] / f) * Em ** PLOT_SLOPE],
                        fmt=EXP_MARKERS.get(name, "o"), markersize=8,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.8, color=color,
                        elinewidth=1.8, capsize=3.5, capthick=1.8, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(E_MIN_P, PLOT_EMAX)
    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_ylabel(r"$E^{2.6}\,I(E)$ [GeV$^{1.6}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]")

    species_handles = [Line2D([], [], color=SPECIES_COLORS.get(s, "k"), lw=3, label=s)
                       for s in SPECIES]
    exp_handles = []
    for e in used_exp:
        factor = median_scale.get(e, 1.0)
        scale_label = (
            rf"$f={factor:.2f}$"
            if e in SCALE_EXPERIMENTS
            else r"$f=1$ (reference)"
        )
        exp_handles.append(
            Line2D(
                [], [], ls="", marker=EXP_MARKERS.get(e, "o"),
                markersize=9, markerfacecolor="white",
                markeredgecolor="0.3", markeredgewidth=1.8,
                label=rf"{e} ({scale_label})",
            )
        )
    ax.legend(handles=species_handles + exp_handles,
              loc="lower left", fontsize=18, ncol=2)
    savefig(fig, "EVA_mcmc_direct_spectrum.pdf")


def plot_lhaaso_extrapolation(d):
    """Compare direct-fit H, He, and H+He extrapolations with LHAASO."""
    samples = d["samples"]
    rng = np.random.default_rng(4)
    idx = rng.choice(
        len(samples), size=min(2000, len(samples)), replace=False
    )

    lhaaso_data = {}
    dampe_data = {}
    for species in ("H", "He"):
        lhaaso_data[species] = kiss_reader.load_experiment(
            LHAASO_EXPERIMENT, species, 0.0, LHAASO_PLOT_EMAX
        )
        dampe_data[species] = kiss_reader.load_experiment(
            "DAMPE", species, EXTRAPOLATION_PLOT_EMIN, LHAASO_PLOT_EMAX
        )
    light_path = os.path.join(
        kiss_reader.KISS_TABLES_DIR,
        "LHAASO_QGSJET-II-04_light_totalEnergy.txt",
    )
    light_data = np.loadtxt(
        light_path, usecols=range(6), unpack=True
    )
    light_mask = light_data[0] <= LHAASO_PLOT_EMAX
    lhaaso_data["light"] = tuple(
        values[light_mask] for values in light_data
    )

    lower = EXTRAPOLATION_PLOT_EMIN
    energy = np.logspace(
        np.log10(lower), np.log10(LHAASO_PLOT_EMAX), 450
    )
    scale = energy ** PLOT_SLOPE

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 17), sharex=True,
        layout="constrained",
    )

    panel_info = (
        ("H", "H", SPECIES_COLORS["H"]),
        ("He", "He", SPECIES_COLORS["He"]),
        ("light", "H+He", "tab:purple"),
    )
    for ax, (observable, panel_label, color) in zip(axes, panel_info):
        if observable == "light":
            curves = np.asarray([
                species_model(samples[i], "H", energy)
                + species_model(samples[i], "He", energy)
                for i in idx
            ])
            datasets = (
                (
                    "LHAASO QGSJET-II-04",
                    lhaaso_data["light"],
                    color,
                    "o",
                ),
            )
        else:
            curves = np.asarray([
                species_model(samples[i], observable, energy) for i in idx
            ])
            datasets = (
                ("DAMPE", dampe_data[observable], "0.45", "s"),
                (
                    "LHAASO QGSJET-II-04",
                    lhaaso_data[observable],
                    color,
                    "o",
                ),
            )
        lo95, lo68, median, up68, up95 = np.percentile(
            curves, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0
        )
        ax.fill_between(
            energy, lo95 * scale, up95 * scale,
            color=color, alpha=0.08, lw=0.0,
        )
        ax.fill_between(
            energy, lo68 * scale, up68 * scale,
            color=color, alpha=0.20, lw=0.0,
        )
        ax.plot(
            energy, median * scale, color=color, lw=3.2,
            label="direct-fit posterior",
        )

        for label, dataset, data_color, marker in datasets:
            e, y, stat_lo, stat_up, sys_lo, sys_up = dataset
            data_scale = e ** PLOT_SLOPE
            y_plot = y * data_scale
            sys_bottom = (y - sys_lo) * data_scale
            sys_top = (y + sys_up) * data_scale
            ax.vlines(
                e, sys_bottom, sys_top,
                color=data_color, linewidth=1.2, linestyles=":",
                alpha=0.9, zorder=3,
            )
            ax.plot(
                e, sys_bottom, ls="", marker="_", markersize=7,
                markeredgewidth=1.2, color=data_color, zorder=3,
            )
            ax.plot(
                e, sys_top, ls="", marker="_", markersize=7,
                markeredgewidth=1.2, color=data_color, zorder=3,
            )
            ax.errorbar(
                e, y_plot,
                yerr=[stat_lo * data_scale, stat_up * data_scale],
                fmt=marker, markersize=7,
                markerfacecolor="white", markeredgecolor=data_color,
                markeredgewidth=1.5, color=data_color,
                elinewidth=1.4, capsize=3.0, capthick=1.4,
                zorder=4, label=label,
            )

        ax.axvspan(
            EXTRAPOLATION_START, LHAASO_PLOT_EMAX,
            color="0.5", alpha=0.06, zorder=0,
        )
        ax.axvline(
            EXTRAPOLATION_START, color="0.35", lw=1.5, ls="--", zorder=1,
        )
        ax.text(
            0.97, 0.92, panel_label,
            transform=ax.transAxes, ha="right", va="top",
            color=color, fontsize=24,
        )
        ax.text(
            EXTRAPOLATION_START * 1.08, 0.08,
            "extrapolation",
            transform=ax.get_xaxis_transform(),
            color="0.35", fontsize=15,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, LHAASO_PLOT_EMAX)

    axes[0].legend(loc="lower left", fontsize=17)
    axes[-1].set_xlabel(r"$E$ [GeV]")
    fig.supylabel(
        rf"$E^{{{PLOT_SLOPE:g}}} I_s(E)$ "
        rf"[GeV$^{{{PLOT_SLOPE - 1.0:g}}}$ "
        r"m$^{-2}$ s$^{-1}$ sr$^{-1}$]"
    )
    savefig(fig, "EVA_mcmc_direct_lhaaso_extrapolation.pdf")


def main():
    d = np.load(OUTPUT, allow_pickle=False)
    plot_reconstruction(d)
    plot_lhaaso_extrapolation(d)
    plot_breaks(d["samples"])
    plot_slopes_2d(d["samples"])
    plot_alpha1_break_2d(d["samples"])


if __name__ == "__main__":
    main()
