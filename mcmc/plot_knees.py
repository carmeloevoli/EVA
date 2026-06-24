"""Plots for the all-particle knee MCMC fit (reads the .npz from mcmc_knees.py).

  * spectrum reconstruction (median + 68%/95% bands) over the data
        -> figures/EVA_mcmc_knees_spectrum.pdf
  * corner plot of the posterior (default matplotlib style)
        -> figures/EVA_mcmc_knees_corner.pdf

Run with:  python plot_knees.py
"""

import os

import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mcmc_knees import OUTPUT, model, MIN_ENERGY, SCALE_EXPERIMENTS, SCALE_START, E1_INDEX

STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EVA.mplstyle")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PLOT_EMAX = 2.0e9     # GeV, upper energy for the model band
PLOT_SLOPE = 3.0      # spectra shown multiplied by E**PLOT_SLOPE

EXP_COLORS = {"LHAASO": "tab:blue", "TALE": "olive", "IceTop-IceCube": "tab:green",
              "KASCADE": "tab:red", "TUNKA-133": "tab:cyan"}
EXP_MARKERS = {"LHAASO": "o", "TALE": "s", "IceTop-IceCube": "^", "KASCADE": "D",
               "TUNKA-133": "v"}


def savefig(fig, name, dpi=300):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


def plot_reconstruction(d):
    samples = d["samples"]
    E, y, elo, eup, exp = (d["data_E"], d["data_y"], d["data_elo"],
                           d["data_eup"], d["data_exp"])

    # Median energy-scale factor per experiment, applied to the displayed data.
    median_scale = {name: np.median(samples[:, SCALE_START + i])
                    for i, name in enumerate(SCALE_EXPERIMENTS)}

    Egrid = np.logspace(np.log10(MIN_ENERGY), np.log10(PLOT_EMAX), 400)
    idx = np.random.default_rng(0).integers(len(samples), size=2000)
    curves = np.array([model(samples[i], Egrid) for i in idx])
    lo95, lo68, med, up68, up95 = np.percentile(curves, [2.5, 16, 50, 84, 97.5], axis=0)
    scale = Egrid ** PLOT_SLOPE

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 9.5))
        ax.fill_between(Egrid, lo95 * scale, up95 * scale, color="tab:purple", alpha=0.15)
        ax.fill_between(Egrid, lo68 * scale, up68 * scale, color="tab:purple", alpha=0.30)
        ax.plot(Egrid, med * scale, color="tab:purple", lw=4.5, zorder=10)

        for name in dict.fromkeys(exp):
            m = exp == name
            color = EXP_COLORS.get(name, "k")
            f = median_scale.get(name, 1.0)
            Em = f * E[m]
            label = name if f == 1.0 else rf"{name} ($f={f:.3f}$)"
            ax.errorbar(Em, (y[m] / f) * Em ** PLOT_SLOPE,
                        yerr=[(elo[m] / f) * Em ** PLOT_SLOPE,
                              (eup[m] / f) * Em ** PLOT_SLOPE],
                        fmt=EXP_MARKERS.get(name, "o"), markersize=8,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.8, color=color, elinewidth=1.8,
                        capsize=3.5, capthick=1.8, alpha=0.7, zorder=3, label=label)

        ax.set_xscale("log")
        ax.set_xlim(MIN_ENERGY, PLOT_EMAX)
        ax.set_ylim(1.5e6, 6.5e6)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(6, 6))
        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_ylabel(r"$E^{3}\,I(E)$ [GeV$^{2}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.legend(loc="upper left", fontsize=18)
        savefig(fig, "EVA_mcmc_knees_spectrum.pdf")


def plot_breaks(d):
    """Marginalized posteriors of the three break energies (knees + hardening)."""
    samples = d["samples"]
    groups = [(E1_INDEX, "tab:blue", "1st knee"),
              (E1_INDEX + 1, "tab:green", "hardening"),
              (E1_INDEX + 2, "tab:red", "2nd knee")]
    chains = {i: samples[:, i] for i, _, _ in groups}
    lo = min(c.min() for c in chains.values())
    hi = max(c.max() for c in chains.values())
    bins = np.linspace(lo, hi, 70)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 8))
        for i, color, label in groups:
            c = chains[i]
            q16, q50, q84 = np.percentile(c, [16, 50, 84])
            ax.hist(c, bins=bins, density=True, histtype="stepfilled",
                    color=color, alpha=0.30)
            ax.hist(c, bins=bins, density=True, histtype="step", color=color, lw=2.5,
                    label=rf"{label}: ${q50:.2f}^{{+{q84 - q50:.2f}}}_{{-{q50 - q16:.2f}}}$")
            ax.axvline(q50, color=color, lw=1.5, ls="--")
        ax.set_xlabel(r"$\log_{10}(E\,/\,{\rm GeV})$")
        ax.set_ylabel("posterior density")
        ax.legend(fontsize=20)
        savefig(fig, "EVA_mcmc_knees_breaks.pdf")


def plot_corner(d):
    labels = d["labels"]
    with plt.style.context("default"):
        fig = corner.corner(
            d["samples"], labels=list(labels), show_titles=True, title_fmt=".2f",
            quantiles=[0.16, 0.5, 0.84], max_n_ticks=3,
            label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10},
        )
        fig.set_size_inches(1.7 * len(labels), 1.7 * len(labels))
        savefig(fig, "EVA_mcmc_knees_corner.pdf", dpi=200)


def main():
    d = np.load(OUTPUT, allow_pickle=False)
    plot_reconstruction(d)
    plot_breaks(d)
    plot_corner(d)


if __name__ == "__main__":
    main()
