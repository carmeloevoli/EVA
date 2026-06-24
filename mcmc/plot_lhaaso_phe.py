"""Plots for the two-component (H+He, shared rigidity-break) fit.

Reads the .npz from mcmc_lhaaso_phe.py and makes:
  * spectrum: I_H, I_He and I_H+I_He bands over the H, He and light data
        -> figures/EVA_mcmc_lhaaso_phe_spectrum.pdf
  * helium posterior over DAMPE, CALET and LHAASO He measurements
        -> figures/EVA_mcmc_lhaaso_phe_helium_spectrum.pdf
  * p/He posterior over the CALET rigidity ratio
        -> figures/EVA_mcmc_lhaaso_phe_p_he_ratio.pdf
  * the three rigidity-break posteriors
        -> figures/EVA_mcmc_lhaaso_phe_breaks.pdf
  * a paired H/He slope fingerprint
        -> figures/EVA_mcmc_lhaaso_phe_slope_fingerprint.pdf
  * corner plot (default matplotlib style)
        -> figures/EVA_mcmc_lhaaso_phe_corner.pdf

Run with:  python plot_lhaaso_phe.py
"""

import os

import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import kiss_reader
from mcmc_lhaaso_phe import (OUTPUT, components, MIN_ENERGY, SCALE_EXPERIMENTS,
                             SCALE_START, R_INDEX, NDIM, A1_H, A1_HE, A2_H,
                             A2_HE, A3_H, A3_HE, A4_H, A4_HE,
                             CROSS_OBSERVABLE_SYS_RHO, FIT_DATASET_TAG)

STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EVA.mplstyle")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PLOT_EMAX = 1.3e7
PLOT_SLOPE = 2.6

OBS_COLORS = {"H": "tab:blue", "He": "tab:red", "light": "k"}
EXP_MARKERS = {"DAMPE": "o", "CALET": "s", "CREAM": "^", "LHAASO": "D"}
EXP_COLORS = {"DAMPE": "tab:red", "CALET": "tab:orange",
              "LHAASO": "tab:green"}
HE_DATASETS = [("DAMPE", "DAMPE"), ("CALET", "CALET"),
               ("LHAASO", "LHAASO_QGSJET-II-04")]
CALET_RATIO_FILE = "CALET_H_He_rigidity.txt"
BREAK_INFO = [(0, "tab:blue", "softening"), (1, "tab:green", "hardening"),
              (2, "tab:red", "knee")]
SLOPE_LABELS = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$\alpha_4$"]
H_SLOPE_IDX = (A1_H, A2_H, A3_H, A4_H)
HE_SLOPE_IDX = (A1_HE, A2_HE, A3_HE, A4_HE)


def savefig(fig, name, dpi=300):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


def validate_output(d):
    nsamples, ndim = d["samples"].shape
    if ndim != NDIM:
        raise RuntimeError(
            f"{OUTPUT} has {ndim} parameters, but the current model expects "
            f"{NDIM}. Re-run mcmc_lhaaso_phe.py before plotting."
        )
    if len(d["labels"]) != ndim:
        raise RuntimeError(
            f"{OUTPUT} has {nsamples} samples with {ndim} parameters, but the "
            "stored labels have a different length."
        )
    covariance_keys = ("cross_observable_sys_experiments",
                       "cross_observable_sys_rho")
    if not all(key in d.files for key in covariance_keys):
        raise RuntimeError(
            f"{OUTPUT} predates the H/light systematic correlations. "
            "Re-run mcmc_lhaaso_phe.py before plotting."
        )
    stored_rho = dict(zip(d["cross_observable_sys_experiments"],
                          d["cross_observable_sys_rho"]))
    if (set(stored_rho) != set(CROSS_OBSERVABLE_SYS_RHO)
            or any(not np.isclose(stored_rho[exp], rho)
                   for exp, rho in CROSS_OBSERVABLE_SYS_RHO.items())):
        raise RuntimeError(
            f"{OUTPUT} was generated with cross-observable correlations "
            f"{stored_rho}, but the current configuration is "
            f"{CROSS_OBSERVABLE_SYS_RHO}. Re-run mcmc_lhaaso_phe.py."
        )
    if "fit_dataset_tag" not in d.files or str(d["fit_dataset_tag"]) != FIT_DATASET_TAG:
        raise RuntimeError(
            f"{OUTPUT} was generated with a different fit dataset. "
            "Re-run mcmc_lhaaso_phe.py before plotting."
        )


def _band(ax, Egrid, curves, color, scale, label, ls="-"):
    lo95, lo68, med, up68, up95 = np.percentile(curves, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(Egrid, lo95 * scale, up95 * scale, color=color, alpha=0.12)
    ax.fill_between(Egrid, lo68 * scale, up68 * scale, color=color, alpha=0.25)
    ax.plot(Egrid, med * scale, color=color, lw=3.0, ls=ls, zorder=10, label=label)


def load_he_data():
    keys = ("E", "y", "err_lo", "err_up", "exp")
    out = {k: [] for k in keys}
    for label, exp_name in HE_DATASETS:
        e, y, slo, sup, ylo, yup = kiss_reader.load_experiment(
            exp_name, "He", MIN_ENERGY, PLOT_EMAX)
        out["E"].append(e)
        out["y"].append(y)
        out["err_lo"].append(np.sqrt(slo ** 2 + ylo ** 2))
        out["err_up"].append(np.sqrt(sup ** 2 + yup ** 2))
        out["exp"].append(np.full(e.size, label))
    return {k: np.concatenate(v) for k, v in out.items()}


def load_calet_ratio():
    path = os.path.join(kiss_reader.KISS_TABLES_DIR, CALET_RATIO_FILE)
    header = kiss_reader.read_header(path)
    if (header.get("X Quantity") != "rigidity"
            or header.get("Y Quantity") != "H_He"):
        raise ValueError(
            f"{CALET_RATIO_FILE} is not an H/He ratio in rigidity"
        )
    x, y, stat_lo, stat_up, sys_lo, sys_up = np.loadtxt(
        path, usecols=range(6), unpack=True
    )
    err_lo = np.sqrt(stat_lo ** 2 + sys_lo ** 2)
    err_up = np.sqrt(stat_up ** 2 + sys_up ** 2)
    return x, y, err_lo, err_up


def plot_spectrum(d):
    samples = d["samples"]
    E, y, elo, eup = d["data_E"], d["data_y"], d["data_err_lo"], d["data_err_up"]
    exp, obs = d["data_exp"], d["data_obs"]
    he_data = load_he_data()
    ms = {name: np.median(samples[:, SCALE_START + i])
          for i, name in enumerate(SCALE_EXPERIMENTS)}

    Egrid = np.logspace(np.log10(MIN_ENERGY), np.log10(PLOT_EMAX), 400)
    scale = Egrid ** PLOT_SLOPE
    idx = np.random.default_rng(0).integers(len(samples), size=2000)
    comp = np.array([components(samples[i], Egrid) for i in idx])  # (n,2,nE)
    IH, IHe = comp[:, 0, :], comp[:, 1, :]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 9.5))
        _band(ax, Egrid, IH, "tab:blue", scale, "H")
        _band(ax, Egrid, IHe, "tab:red", scale, "He", ls="--")
        _band(ax, Egrid, IH + IHe, "tab:purple", scale, "H+He")

        for ob in ("H", "He", "light"):
            for name in dict.fromkeys(exp):
                m = (obs == ob) & (exp == name)
                if not m.any():
                    continue
                f = ms.get(name, 1.0)
                Em = f * E[m]
                ax.errorbar(Em, (y[m] / f) * Em ** PLOT_SLOPE,
                            yerr=[(elo[m] / f) * Em ** PLOT_SLOPE,
                                  (eup[m] / f) * Em ** PLOT_SLOPE],
                            fmt=EXP_MARKERS.get(name, "o"), markersize=7,
                            markerfacecolor="white", markeredgecolor=OBS_COLORS[ob],
                            markeredgewidth=1.5, color=OBS_COLORS[ob], elinewidth=1.5,
                            capsize=3, alpha=0.7, zorder=3)

        # he_exp = he_data["exp"]
        # for name in dict.fromkeys(he_exp):
        #     m = he_exp == name
        #     f = ms.get(name, 1.0)
        #     Em = f * he_data["E"][m]
        #     ax.errorbar(Em, (he_data["y"][m] / f) * Em ** PLOT_SLOPE,
        #                 yerr=[(he_data["err_lo"][m] / f) * Em ** PLOT_SLOPE,
        #                       (he_data["err_up"][m] / f) * Em ** PLOT_SLOPE],
        #                 fmt=EXP_MARKERS.get(name, "o"), markersize=7,
        #                 markerfacecolor="white", markeredgecolor="0.45",
        #                 markeredgewidth=1.5, color="0.45", elinewidth=1.3,
        #                 capsize=3, alpha=0.65, zorder=4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(MIN_ENERGY, PLOT_EMAX)
        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_ylabel(r"$E^{2.6}\,I(E)$ [GeV$^{1.6}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        comp_handles = [Line2D([], [], color="tab:blue", lw=3, label="H"),
                        Line2D([], [], color="tab:red", lw=3, ls="--", label="He"),
                        Line2D([], [], color="tab:purple", lw=3, label="H+He")]
        obs_handles = [
            Line2D([], [], ls="", marker="o", markerfacecolor="white",
                   markeredgecolor=OBS_COLORS[ob], markeredgewidth=1.5,
                   color=OBS_COLORS[ob], label=label)
            for ob, label in (("H", "H data"), ("He", "He data"),
                              ("light", "H+He data"))
        ]
        data_handles = [Line2D([], [], ls="", marker=mk, markerfacecolor="white",
                               markeredgecolor="0.3", markeredgewidth=1.5, label=e)
                        for e, mk in EXP_MARKERS.items()]
        ax.legend(handles=comp_handles + obs_handles + data_handles, loc="lower left",
                  fontsize=17, ncol=2)
        savefig(fig, "EVA_mcmc_lhaaso_phe_spectrum.pdf")


def plot_helium_spectrum(d):
    samples = d["samples"]
    he_data = load_he_data()
    median_scale = {
        name: np.median(samples[:, SCALE_START + i])
        for i, name in enumerate(SCALE_EXPERIMENTS)
    }

    Egrid = np.logspace(np.log10(MIN_ENERGY), np.log10(PLOT_EMAX), 400)
    scale = Egrid ** PLOT_SLOPE
    rng = np.random.default_rng(1)
    idx = rng.choice(len(samples), size=min(2000, len(samples)), replace=False)
    curves = np.array([components(samples[i], Egrid)[1] for i in idx])

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 9.5))
        _band(ax, Egrid, curves, "tab:red", scale, "He posterior")

        for name in dict.fromkeys(he_data["exp"]):
            m = he_data["exp"] == name
            f = median_scale.get(name, 1.0)
            Em = f * he_data["E"][m]
            color = EXP_COLORS.get(name, "0.35")
            ax.errorbar(
                Em, (he_data["y"][m] / f) * Em ** PLOT_SLOPE,
                yerr=[
                    (he_data["err_lo"][m] / f) * Em ** PLOT_SLOPE,
                    (he_data["err_up"][m] / f) * Em ** PLOT_SLOPE,
                ],
                fmt=EXP_MARKERS.get(name, "o"), markersize=8,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.7, color=color, elinewidth=1.5,
                capsize=3.5, alpha=0.85, zorder=3,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(MIN_ENERGY, PLOT_EMAX)
        ax.set_xlabel(r"$E_{\rm He}$ [GeV]")
        ax.set_ylabel(
            r"$E_{\rm He}^{2.6}\,I_{\rm He}(E_{\rm He})$ "
            r"[GeV$^{1.6}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        handles = [Line2D([], [], color="tab:red", lw=3,
                          label="He posterior")]
        handles += [
            Line2D([], [], ls="", marker=EXP_MARKERS[name],
                   markerfacecolor="white", markeredgecolor=EXP_COLORS[name],
                   markeredgewidth=1.7, label=name)
            for name, _ in HE_DATASETS
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=18)
        savefig(fig, "EVA_mcmc_lhaaso_phe_helium_spectrum.pdf")


def plot_p_he_ratio(d):
    samples = d["samples"]
    R_data, ratio_data, err_lo, err_up = load_calet_ratio()
    calet_scale = np.median(
        samples[:, SCALE_START + SCALE_EXPERIMENTS.index("CALET")]
    )

    Z_h = kiss_reader.SPECIES["H"]["Z"]
    A_h = kiss_reader.SPECIES["H"]["A"]
    Z_he = kiss_reader.SPECIES["He"]["Z"]
    A_he = kiss_reader.SPECIES["He"]["A"]
    mass_nucleon = kiss_reader.PROTON_MASS
    mass_h = A_h * mass_nucleon
    mass_he = A_he * mass_nucleon
    R_max_h = np.sqrt(PLOT_EMAX ** 2 - mass_h ** 2) / Z_h
    R_max_he = np.sqrt(PLOT_EMAX ** 2 - mass_he ** 2) / Z_he
    R_grid = np.logspace(np.log10(0.75 * R_data.min()),
                         np.log10(min(R_max_h, R_max_he)), 400)
    E_h = np.sqrt((Z_h * R_grid) ** 2 + mass_h ** 2)
    E_he = np.sqrt((Z_he * R_grid) ** 2 + mass_he ** 2)
    dE_h_dR = Z_h ** 2 * R_grid / E_h
    dE_he_dR = Z_he ** 2 * R_grid / E_he

    rng = np.random.default_rng(2)
    idx = rng.choice(len(samples), size=min(2000, len(samples)), replace=False)
    ratio_curves = []
    for i in idx:
        I_h = components(samples[i], E_h)[0]
        I_he = components(samples[i], E_he)[1]
        ratio_curves.append((I_h * dE_h_dR) / (I_he * dE_he_dR))
    ratio_curves = np.asarray(ratio_curves)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 8.8))
        _band(ax, R_grid, ratio_curves, "tab:purple", 1.0,
              r"$p/{\rm He}$ posterior")
        ax.errorbar(
            calet_scale * R_data, ratio_data, yerr=[err_lo, err_up],
            fmt=EXP_MARKERS["CALET"], markersize=8,
            markerfacecolor="white", markeredgecolor=EXP_COLORS["CALET"],
            markeredgewidth=1.7, color=EXP_COLORS["CALET"],
            elinewidth=1.6, capsize=3.5, label="CALET", zorder=3,
        )
        ax.set_xscale("log")
        ax.set_xlim(R_grid[0], R_grid[-1])
        ax.set_xlabel(r"$R$ [GV]")
        ax.set_ylabel(r"$p/{\rm He}$")
        ax.legend(loc="best", fontsize=19)
        savefig(fig, "EVA_mcmc_lhaaso_phe_p_he_ratio.pdf")


def plot_breaks(d):
    samples = d["samples"]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 8))
        for j, color, name in BREAK_INFO:
            c = samples[:, R_INDEX + j]
            q16, q50, q84 = np.percentile(c, [16, 50, 84])
            ax.hist(c, bins=70, density=True, histtype="stepfilled", color=color, alpha=0.30)
            ax.hist(c, bins=70, density=True, histtype="step", color=color, lw=2.5,
                    label=rf"{name}: ${q50:.2f}^{{+{q84 - q50:.2f}}}_{{-{q50 - q16:.2f}}}$")
            ax.axvline(q50, color=color, lw=1.5, ls="--")
        ax.set_xlabel(r"$\log_{10}(R_b\,/\,{\rm GV})$")
        ax.set_ylabel("posterior density")
        ax.legend(fontsize=20)
        savefig(fig, "EVA_mcmc_lhaaso_phe_breaks.pdf")


def _sign_style(delta):
    p_pos = np.mean(delta > 0.0)
    strength = 2.0 * abs(p_pos - 0.5)
    if strength < 0.12:
        return "0.45", 0.40, p_pos
    color = "tab:red" if p_pos > 0.5 else "tab:blue"
    return color, 0.25 + 0.65 * strength, p_pos


def _smooth_density(values, grid):
    counts, edges = np.histogram(values, bins=70, range=(grid[0], grid[-1]),
                                 density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x = np.linspace(-3.0, 3.0, 25)
    kernel = np.exp(-0.5 * x ** 2)
    kernel /= kernel.sum()
    density = np.convolve(counts, kernel, mode="same")
    return np.interp(grid, centers, density, left=0.0, right=0.0)


def _split_violin(ax, x0, values, grid, width=0.34):
    density = _smooth_density(values, grid)
    if density.max() <= 0.0:
        return
    half_width = width * density / density.max()
    neg = grid <= 0.0
    pos = grid >= 0.0
    ax.fill_betweenx(grid[neg], x0 - half_width[neg], x0 + half_width[neg],
                     color="tab:blue", alpha=0.28, lw=0.0)
    ax.fill_betweenx(grid[pos], x0 - half_width[pos], x0 + half_width[pos],
                     color="tab:red", alpha=0.28, lw=0.0)

    q025, q16, q50, q84, q975 = np.percentile(values, [2.5, 16, 50, 84, 97.5])
    ax.plot([x0, x0], [q025, q975], color="0.25", lw=1.1, alpha=0.70)
    ax.plot([x0, x0], [q16, q84], color="0.05", lw=2.4, alpha=0.90)
    ax.plot(x0, q50, marker="o", ms=6.5, color="white", mec="0.05", mew=1.5)


def plot_slope_fingerprint(d):
    samples = d["samples"]
    h = samples[:, H_SLOPE_IDX]
    he = samples[:, HE_SLOPE_IDX]
    delta = h - he
    x = np.arange(1, 5)
    x_h = x - 0.06
    x_he = x + 0.06

    h_q = np.percentile(h, [2.5, 16, 50, 84, 97.5], axis=0)
    he_q = np.percentile(he, [2.5, 16, 50, 84, 97.5], axis=0)

    span = np.nanpercentile(np.abs(delta), 99.0)
    span = max(0.05, 1.15 * span)
    grid = np.linspace(-span, span, 320)

    with plt.style.context(STYLE):
        fig, (ax_top, ax_delta) = plt.subplots(
            2, 1, figsize=(11.5, 9.2), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.25], "hspace": 0.05}
        )

        ax_top.fill_between(x_h, h_q[0], h_q[4], color="tab:blue", alpha=0.10)
        ax_top.fill_between(x_h, h_q[1], h_q[3], color="tab:blue", alpha=0.24)
        ax_top.plot(x_h, h_q[2], marker="o", color="tab:blue", lw=3.0, label="H")

        ax_top.fill_between(x_he, he_q[0], he_q[4], color="0.35", alpha=0.10)
        ax_top.fill_between(x_he, he_q[1], he_q[3], color="0.35", alpha=0.24)
        ax_top.plot(x_he, he_q[2], marker="s", color="0.35", lw=3.0, label="He")

        for j in range(4):
            color, alpha, p_pos = _sign_style(delta[:, j])
            ax_top.plot([x_h[j], x_he[j]], [h_q[2, j], he_q[2, j]],
                        color=color, alpha=alpha, lw=5.0, solid_capstyle="round",
                        zorder=2)
            ax_delta.text(x[j], 0.92 * span, rf"$P_+={100 * p_pos:.0f}\%$",
                          ha="center", va="top", fontsize=13, color=color)
            _split_violin(ax_delta, x[j], delta[:, j], grid)

        ax_top.set_ylabel(r"slope $\alpha$")
        ax_top.set_xlim(0.55, 4.45)
        ax_top.legend(loc="best", fontsize=19)
        ax_top.grid(axis="y", ls=":", lw=0.8, alpha=0.45)

        ax_delta.axhline(0.0, color="0.25", lw=1.4, ls="--")
        ax_delta.set_ylabel(r"$\Delta\alpha$" "\n" r"H $-$ He")
        ax_delta.set_xlabel("spectral segment")
        ax_delta.set_ylim(-span, span)
        ax_delta.set_xticks(x)
        ax_delta.set_xticklabels(SLOPE_LABELS)
        ax_delta.grid(axis="y", ls=":", lw=0.8, alpha=0.45)

        color_handles = [
            Line2D([], [], color="tab:blue", lw=6, label=r"H harder: $\Delta\alpha<0$"),
            Line2D([], [], color="tab:red", lw=6, label=r"H softer: $\Delta\alpha>0$"),
        ]
        ax_delta.legend(handles=color_handles, loc="lower right", fontsize=15)
        savefig(fig, "EVA_mcmc_lhaaso_phe_slope_fingerprint.pdf")


def plot_corner(d):
    labels = list(d["labels"])
    with plt.style.context("default"):
        fig = corner.corner(d["samples"], labels=labels, show_titles=True,
                            title_fmt=".2f", quantiles=[0.16, 0.5, 0.84],
                            max_n_ticks=3, label_kwargs={"fontsize": 11},
                            title_kwargs={"fontsize": 9})
        fig.set_size_inches(1.6 * len(labels), 1.6 * len(labels))
        savefig(fig, "EVA_mcmc_lhaaso_phe_corner.pdf", dpi=170)


def report(d):
    s = d["samples"]
    print("\nRigidity breaks log10(R/GV)  (E_b(H)=R, E_b(He)=2R):")
    for j, _, name in BREAK_INFO:
        q16, q50, q84 = np.percentile(s[:, R_INDEX + j], [16, 50, 84])
        print(f"  {name:10s}: {q50:.2f} (+{q84 - q50:.2f}/-{q50 - q16:.2f})  "
              f"R={10 ** q50 / 1e6:.2f} PV -> E_He={2 * 10 ** q50 / 1e6:.2f} PeV")
    h_idx = (A1_H, A2_H, A3_H, A4_H)
    he_idx = (A1_HE, A2_HE, A3_HE, A4_HE)
    h_slopes = [np.median(s[:, i]) for i in h_idx]
    he_slopes = [np.median(s[:, i]) for i in he_idx]
    print("slopes H : " + " ".join(f"a{i + 1}={v:.3f}" for i, v in enumerate(h_slopes)))
    print("slopes He: " + " ".join(f"a{i + 1}={v:.3f}" for i, v in enumerate(he_slopes)))
    for i, (ih, ihe) in enumerate(zip(h_idx, he_idx), start=1):
        delta = s[:, ih] - s[:, ihe]
        print(f"alpha{i}_H-alpha{i}_He={np.median(delta):+.3f} "
              f"({abs(np.median(delta)) / delta.std():.1f} sigma)")

    d23H = s[:, A3_H] - s[:, A2_H]
    d23He = s[:, A3_HE] - s[:, A2_HE]
    d34H = s[:, A4_H] - s[:, A3_H]
    d34He = s[:, A4_HE] - s[:, A3_HE]
    print(f"hardening H: Delta=a3_H-a2_H={np.median(d23H):+.3f}, "
          f"P(hardening)={np.mean(d23H < 0) * 100:.0f}%")
    print(f"hardening He: Delta=a3_He-a2_He={np.median(d23He):+.3f}, "
          f"P(hardening)={np.mean(d23He < 0) * 100:.0f}%")
    print(f"final steepening H: Delta=a4_H-a3_H={np.median(d34H):+.3f}, "
          f"P(steepening)={np.mean(d34H > 0) * 100:.0f}%")
    print(f"final steepening He: Delta=a4_He-a3_He={np.median(d34He):+.3f}, "
          f"P(steepening)={np.mean(d34He > 0) * 100:.0f}%")
    he_h = 10 ** np.median(s[:, 1] - s[:, 0])
    print(f"normalization He/H (at common rigidity) = {he_h:.2f}")


def main():
    d = np.load(OUTPUT, allow_pickle=False)
    validate_output(d)
    plot_spectrum(d)
    plot_helium_spectrum(d)
    plot_p_he_ratio(d)
    plot_breaks(d)
    plot_slope_fingerprint(d)
    plot_corner(d)
    report(d)


if __name__ == "__main__":
    main()
