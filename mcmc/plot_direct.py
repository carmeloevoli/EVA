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
from scipy.stats import norm, gaussian_kde

from mcmc_direct import (OUTPUT, SPECIES, Z, E0_P, E_MIN_P, SCALE_EXPERIMENTS,
                         SCALE_START, species_model, RB_INDEX, SLOPE_INDEX)

plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "EVA.mplstyle"))

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PLOT_EMAX = 3.0e6   # GeV, upper energy for the model bands (extrapolated beyond the fit)
PLOT_SLOPE = 2.6    # the spectra are shown multiplied by E**PLOT_SLOPE
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


BREAK_PAIRS = [("nuclei - p", "heavy", "H", "tab:green"),
               ("He - p", "He", "H", "tab:red"),
               ("He - nuclei", "He", "heavy", "tab:purple")]


def _hdi(x, cred=0.68):
    """Narrowest (highest-density) interval containing a fraction ``cred``."""
    xs = np.sort(x)
    n = len(xs)
    k = max(1, int(np.floor(cred * n)))
    widths = xs[k:] - xs[:n - k]
    i = int(np.argmin(widths))
    return xs[i], xs[i + k]


def _direction_sigma(delta, n):
    """Tail-probability -> Gaussian sigma for a difference vs zero.

    Uses the probability of direction: the fraction of the posterior sharing
    the sign of the median, converted to an equivalent one-sided Gaussian level.
    Returns (n_sigma, p_cross, is_lower_bound)."""
    p_cross = np.mean(delta < 0) if np.median(delta) >= 0 else np.mean(delta > 0)
    if p_cross == 0:                 # unresolved by the finite sample
        return norm.ppf(1 - 1.0 / n), 1.0 / n, True
    return norm.ppf(1 - p_cross), p_cross, False


def report_break_differences(samples):
    """Print the posterior of the pairwise rigidity-break differences with the
    68% HDI and the tail-probability -> sigma significance (correlations kept)."""
    rb = {g: samples[:, RB_INDEX[g]] for g in ("H", "He", "heavy")}
    n = len(samples)
    print("\nPairwise rigidity-break differences "
          "Delta = log10(R_b,X) - log10(R_b,Y) [dex]:")
    for label, X, Y, _ in BREAK_PAIRS:
        delta = rb[X] - rb[Y]
        med = np.median(delta)
        h_lo, h_hi = _hdi(delta, 0.68)
        nsig, p_cross, lower = _direction_sigma(delta, n)
        sig_str = f">{nsig:.1f}" if lower else f"{nsig:.1f}"
        p_str = f"<{100 / n:.2g}%" if lower else f"{p_cross * 100:.2f}%"
        print(f"  {label:13s}: median={med:+.3f} dex (R_b ratio {10 ** med:.2f}), "
              f"68% HDI=[{h_lo:+.3f}, {h_hi:+.3f}], "
              f"P(cross 0)={p_str}, significance={sig_str} sigma")


def plot_break_differences(samples):
    """Posterior of the pairwise rigidity-break differences, with zero marked."""
    rb = {g: samples[:, RB_INDEX[g]] for g in ("H", "He", "heavy")}
    deltas = {lab: rb[X] - rb[Y] for lab, X, Y, _ in BREAK_PAIRS}
    lo = min(d.min() for d in deltas.values())
    hi = max(d.max() for d in deltas.values())
    bins = np.linspace(lo, hi, 60)

    n = len(samples)
    fig, ax = plt.subplots(figsize=(11, 8))
    for label, X, Y, color in BREAK_PAIRS:
        delta = deltas[label]
        med = np.median(delta)
        nsig, _, lower = _direction_sigma(delta, n)
        sig_str = rf">{nsig:.1f}\sigma" if lower else rf"{nsig:.1f}\sigma"
        ax.hist(delta, bins=bins, density=True, histtype="stepfilled",
                color=color, alpha=0.25)
        ax.hist(delta, bins=bins, density=True, histtype="step", color=color, lw=2.5,
                label=rf"{label}: ${med:+.2f}$ (${sig_str}$)")
        ax.axvline(med, color=color, lw=1.5, ls="--")
    ax.axvline(0.0, color="k", lw=2.0)

    ax.set_xlabel(r"$\Delta \log_{10} R_b$ [dex]")
    ax.set_ylabel("posterior density")
    ax.set_title("Pairwise break differences")
    ax.legend(fontsize=20)
    savefig(fig, "EVA_mcmc_direct_break_diffs.pdf")


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
        dens = gaussian_kde(np.vstack([x[sub], y[sub]]))(grid).reshape(XX.shape)
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
            # systematic error: wide, semi-transparent bar without caps (drawn behind)
            ax.errorbar(Em, yv,
                        yerr=[(sys_lo[m] / f) * Em ** PLOT_SLOPE, (sys_up[m] / f) * Em ** PLOT_SLOPE],
                        fmt="none", ecolor=color, elinewidth=5.0, alpha=0.30,
                        capsize=0, zorder=2)
            # statistical error: crisp bar with caps + open marker on top
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
    ax.set_title("Direct spectra -- single-break SBPL")

    species_handles = [Line2D([], [], color=SPECIES_COLORS.get(s, "k"), lw=3, label=s)
                       for s in SPECIES]
    exp_handles = [Line2D([], [], ls="", marker=EXP_MARKERS.get(e, "o"), markersize=9,
                          markerfacecolor="white", markeredgecolor="0.3",
                          markeredgewidth=1.8, label=e) for e in used_exp]
    ax.legend(handles=species_handles + exp_handles,
              loc="lower left", fontsize=18, ncol=2)
    savefig(fig, "EVA_mcmc_direct_spectrum.pdf")


def main():
    d = np.load(OUTPUT, allow_pickle=False)
    plot_reconstruction(d)
    plot_breaks(d["samples"])
    report_break_differences(d["samples"])
    plot_break_differences(d["samples"])
    plot_slopes_2d(d["samples"])
    plot_alpha1_break_2d(d["samples"])


if __name__ == "__main__":
    main()
