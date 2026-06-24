"""Spectrum and corner plots for the two-population MCMC fit.

Reads ``output/EVA_mcmc_twopop.npz`` from :mod:`mcmc_twopop` and creates:

    figures/EVA_mcmc_twopop_spectrum.pdf
    figures/EVA_mcmc_twopop_corner.pdf

The spectrum figure has one panel per species.  The total LE+HE posterior is
shown with median, 68%, and 95% credible regions.  LE and HE median components
are overlaid separately.  Statistical data errors are solid capped bars;
systematic errors are dotted intervals.

Run with:

    python plot_twopop.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import kiss_reader
from mcmc_twopop import (
    A,
    EXPERIMENTS,
    MIN_RIGIDITY,
    NDIM,
    OUTPUT,
    POP_SPECIES,
    LABELS,
    SCALE_EXPERIMENTS,
    SCALE_START,
    SPECIES,
    Z,
    load_data,
    species_components,
)


STYLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "EVA.mplstyle"
)
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures"
)

PLOT_SLOPE = 2.6
PLOT_EMAX = 3.0e7
PLOT_YLIM = (6.0e1, 1.0e4)
N_POSTERIOR_CURVES = 2000
N_CORNER_SAMPLES = 50000
DOPLOTPOPS = True

EXP_MARKERS = {
    "DAMPE": "o",
    "CALET": "s",
    "LHAASO": "D",
    "KASCADE": "^",
}
SPECIES_COLORS = {
    "H": "tab:blue",
    "He": "tab:orange",
    "C": "tab:green",
    "O": "tab:red",
    "Fe": "tab:purple",
}


def savefig(fig, filename, dpi=300):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


def validate_output(d):
    required = (
        "samples", "labels", "species", "populations", "model_kind",
        "data_E", "data_y", "data_stat_lo", "data_stat_up",
        "data_sys_lo", "data_sys_up", "data_exp", "data_sp",
        "scale_start", "scale_experiments",
    )
    missing = [key for key in required if key not in d.files]
    if missing:
        raise RuntimeError(
            f"{OUTPUT} is missing {', '.join(missing)}. Re-run "
            "mcmc_twopop.py before plotting."
        )

    samples = d["samples"]
    if samples.ndim != 2 or samples.shape[1] != NDIM:
        raise RuntimeError(
            f"{OUTPUT} contains samples with shape {samples.shape}, while "
            f"the current model expects (*, {NDIM})."
        )
    if list(d["labels"]) != LABELS:
        raise RuntimeError(
            "stored parameter labels do not match the current SBPL model. "
            "Re-run mcmc_twopop.py."
        )
    if str(d["model_kind"]) != "two_population_sbpl":
        raise RuntimeError(
            f"{OUTPUT} was generated with model '{d['model_kind']}', not "
            "the current two-population SBPL model."
        )
    if list(d["species"]) != SPECIES:
        raise RuntimeError(
            f"stored species {list(d['species'])} do not match {SPECIES}"
        )
    if int(d["scale_start"]) != SCALE_START:
        raise RuntimeError("stored energy-scale parameter layout has changed")
    if list(d["scale_experiments"]) != SCALE_EXPERIMENTS:
        raise RuntimeError("stored energy-scale experiments have changed")


def reload_and_validate_data(d):
    """Reload source tables and check that they match the fitted data."""
    data = load_data()
    mapping = {
        "E": "data_E",
        "y": "data_y",
        "stat_lo": "data_stat_lo",
        "stat_up": "data_stat_up",
        "sys_lo": "data_sys_lo",
        "sys_up": "data_sys_up",
        "exp": "data_exp",
        "sp": "data_sp",
    }

    for current_key, stored_key in mapping.items():
        current = data[current_key]
        stored = d[stored_key]
        if current.dtype.kind in "USO":
            matches = np.array_equal(current, stored)
        else:
            matches = (
                current.shape == stored.shape
                and np.allclose(current, stored, rtol=1.0e-12, atol=0.0)
            )
        if not matches:
            raise RuntimeError(
                f"current source-table field '{current_key}' does not match "
                f"{stored_key} in {OUTPUT}. Re-run the fit."
            )

    print("verified current source tables against the fitted data")
    return data


def load_kascade_fe():
    """Load KASCADE Fe as a plot-only comparison dataset."""
    filename = "KASCADE_Kuznetsov2024_Fe_totalEnergy.txt"
    path = os.path.join(kiss_reader.KISS_TABLES_DIR, filename)
    energy, flux, stat_lo, stat_up, sys_lo, sys_up = np.loadtxt(
        path, usecols=range(6), unpack=True
    )
    mask = energy <= PLOT_EMAX
    return {
        "E": energy[mask],
        "y": flux[mask],
        "stat_lo": stat_lo[mask],
        "stat_up": stat_up[mask],
        "sys_lo": sys_lo[mask],
        "sys_up": sys_up[mask],
    }


def posterior_indices(nsamples, size=N_POSTERIOR_CURVES, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(nsamples, size=min(size, nsamples), replace=False)


def median_scales(samples):
    return {
        exp: np.median(samples[:, SCALE_START + i])
        for i, exp in enumerate(SCALE_EXPERIMENTS)
    }


def energy_grid(data, species, npoints=350):
    mask = data["sp"] == species
    data_min = np.min(data["E"][mask])
    mass = A[species] * 0.938272
    rigidity_min_energy = np.sqrt(
        (Z[species] * MIN_RIGIDITY) ** 2 + mass ** 2
    )
    lower = min(data_min, rigidity_min_energy)
    return np.logspace(np.log10(lower), np.log10(PLOT_EMAX), npoints)


def draw_posterior(ax, samples, sample_idx, species, energy):
    le_curves = []
    he_curves = []
    for i in sample_idx:
        le, he = species_components(samples[i], species, energy)
        le_curves.append(le)
        he_curves.append(he)

    le_curves = np.asarray(le_curves)
    he_curves = np.asarray(he_curves)
    total_curves = le_curves + he_curves
    lo95, lo68, median, up68, up95 = np.percentile(
        total_curves, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0
    )

    color = SPECIES_COLORS[species]
    scale = energy ** PLOT_SLOPE
    ax.fill_between(
        energy, lo95 * scale, up95 * scale,
        color=color, alpha=0.05, lw=0.0,
    )
    ax.fill_between(
        energy, lo68 * scale, up68 * scale,
        color=color, alpha=0.12, lw=0.0,
    )
    ax.plot(
        energy, median * scale, color=color, lw=3.0,
    )

    if DOPLOTPOPS:
        le_median = np.median(le_curves, axis=0)
        ax.plot(
            energy, le_median * scale, color=color,
            lw=1.8, ls="--",
        )

        if species in POP_SPECIES["HE"]:
            he_median = np.median(he_curves, axis=0)
            ax.plot(
                energy, he_median * scale, color=color,
                lw=1.8, ls=":",
            )


def draw_data(ax, data, samples, species):
    scales = median_scales(samples)
    for exp in EXPERIMENTS:
        mask = (data["sp"] == species) & (data["exp"] == exp)
        if not np.any(mask):
            continue

        factor = scales.get(exp, 1.0)
        energy = factor * data["E"][mask]
        plot_scale = energy ** PLOT_SLOPE / factor
        flux = data["y"][mask] * plot_scale
        stat_lo = data["stat_lo"][mask] * plot_scale
        stat_up = data["stat_up"][mask] * plot_scale
        sys_lo = data["sys_lo"][mask] * plot_scale
        sys_up = data["sys_up"][mask] * plot_scale
        color = SPECIES_COLORS[species]

        sys_bottom = flux - sys_lo
        sys_top = flux + sys_up
        ax.vlines(
            energy, sys_bottom, sys_top,
            color=color, linewidth=1.1, linestyles=":",
            alpha=0.9, zorder=2,
        )
        ax.plot(
            energy, sys_bottom, ls="", marker="_", markersize=6,
            markeredgewidth=1.1, color=color, zorder=2,
        )
        ax.plot(
            energy, sys_top, ls="", marker="_", markersize=6,
            markeredgewidth=1.1, color=color, zorder=2,
        )

        ax.errorbar(
            energy, flux, yerr=[stat_lo, stat_up],
            fmt=EXP_MARKERS[exp], markersize=5.5,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.3, color=color,
            elinewidth=1.2, capsize=2.5, capthick=1.2,
            zorder=3,
        )


def draw_kascade_fe(ax):
    data = load_kascade_fe()
    energy = data["E"]
    plot_scale = energy ** PLOT_SLOPE
    flux = data["y"] * plot_scale
    stat_lo = data["stat_lo"] * plot_scale
    stat_up = data["stat_up"] * plot_scale
    sys_lo = data["sys_lo"] * plot_scale
    sys_up = data["sys_up"] * plot_scale
    color = SPECIES_COLORS["Fe"]

    sys_bottom = flux - sys_lo
    sys_top = flux + sys_up
    ax.vlines(
        energy, sys_bottom, sys_top,
        color=color, linewidth=1.1, linestyles=":",
        alpha=0.9, zorder=2,
    )
    ax.plot(
        energy, sys_bottom, ls="", marker="_", markersize=6,
        markeredgewidth=1.1, color=color, zorder=2,
    )
    ax.plot(
        energy, sys_top, ls="", marker="_", markersize=6,
        markeredgewidth=1.1, color=color, zorder=2,
    )
    ax.errorbar(
        energy, flux, yerr=[stat_lo, stat_up],
        fmt=EXP_MARKERS["KASCADE"], markersize=6.5,
        markerfacecolor="white", markeredgecolor=color,
        markeredgewidth=1.4, color=color,
        elinewidth=1.2, capsize=2.5, capthick=1.2,
        zorder=4,
    )


def plot_spectrum(d, data):
    samples = d["samples"]
    sample_idx = posterior_indices(len(samples))

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 10))

        for species in SPECIES:
            energy = energy_grid(data, species)
            draw_posterior(ax, samples, sample_idx, species, energy)
            draw_data(ax, data, samples, species)
        draw_kascade_fe(ax)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(
            0.85 * np.min(data["E"]),
            PLOT_EMAX,
        )
        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_ylabel(
            r"$E^{2.6}\,I(E)$ "
            r"[GeV$^{1.6}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        ax.set_ylim(*PLOT_YLIM)

        model_handles = [
            Line2D([], [], color="0.2", lw=3.0, label="total"),
        ]
        if DOPLOTPOPS:
            model_handles += [
                Line2D([], [], color="0.2", lw=1.8, ls="--", label="LE"),
                Line2D([], [], color="0.2", lw=1.8, ls=":", label="HE"),
            ]
        species_handles = [
            Line2D(
                [], [], color=SPECIES_COLORS[species], lw=3.0,
                label=species,
            )
            for species in SPECIES
        ]
        experiment_handles = [
            Line2D(
                [], [], ls="", marker=EXP_MARKERS[exp], markersize=7,
                markerfacecolor="white", markeredgecolor="0.3",
                markeredgewidth=1.4, label=exp,
            )
            for exp in EXPERIMENTS
        ]
        experiment_handles.append(
            Line2D(
                [], [], ls="", marker=EXP_MARKERS["KASCADE"], markersize=7,
                markerfacecolor="white", markeredgecolor="0.3",
                markeredgewidth=1.4, label="KASCADE Fe (comparison)",
            )
        )
        uncertainty_handles = [
            Line2D(
                [], [], ls="", marker="_", markersize=9,
                markeredgewidth=1.3, color="0.3", label="statistical",
            ),
            Line2D(
                [], [], ls=":", marker="_", markersize=7,
                markeredgewidth=1.1, lw=1.1,
                color="0.4", label="systematic",
            ),
        ]
        ax.legend(
            handles=species_handles + model_handles
            + experiment_handles + uncertainty_handles,
            loc="lower left", fontsize=17, ncol=2,
        )

        ax.set_title("Two-population Galactic model")
        savefig(fig, "EVA_mcmc_twopop_spectrum.pdf")


def plot_corner(d):
    try:
        import corner
    except ImportError as exc:
        raise RuntimeError(
            "the 'corner' package is required for the corner plot; install it "
            "or run with --only spectrum"
        ) from exc

    labels = list(d["labels"])
    samples = d["samples"]
    if len(samples) > N_CORNER_SAMPLES:
        idx = posterior_indices(
            len(samples), size=N_CORNER_SAMPLES, seed=1
        )
        samples = samples[idx]

    with plt.style.context("default"):
        fig = corner.corner(
            samples,
            labels=labels,
            show_titles=True,
            title_fmt=".2f",
            quantiles=[0.16, 0.5, 0.84],
            max_n_ticks=3,
            label_kwargs={"fontsize": 11},
            title_kwargs={"fontsize": 9},
        )
        fig.set_size_inches(1.6 * len(labels), 1.6 * len(labels))
        savefig(fig, "EVA_mcmc_twopop_corner.pdf", dpi=170)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the two-population MCMC output."
    )
    parser.add_argument(
        "--only",
        choices=("all", "spectrum", "corner"),
        default="all",
        help="select which figure to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    d = np.load(OUTPUT, allow_pickle=False)
    validate_output(d)

    if args.only in ("all", "spectrum"):
        data = reload_and_validate_data(d)
        plot_spectrum(d, data)
    if args.only in ("all", "corner"):
        plot_corner(d)


if __name__ == "__main__":
    main()
