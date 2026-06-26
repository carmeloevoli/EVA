"""Compare two spectral shapes for the LHAASO H or He spectrum.

The fitted hypotheses are

    exponential cutoff:
        I(E) = K (E/E0)^(-alpha) exp[-(E - E0)/Ecut]

    smoothed broken power law:
        I(E) = K (E/E0)^(-alpha)
               { [1 + (E/Eb)^(1/w)]
                 / [1 + (E0/Eb)^(1/w)] }^(-DeltaAlpha w)

The normalization ``K`` is the intensity at the pivot energy ``E0`` in both
models.  By default the fit starts at 3e5 GeV, excluding the first three
LHAASO bins.  Statistical errors are independent, while systematic errors use
a common correlation coefficient ``rho``:

    C = diag(stat^2) + (1-rho) diag(sys^2) + rho outer(sys, sys).

Thus ``rho=0`` treats systematics as independent and ``rho=1`` as fully
correlated.  The default is ``rho=0.5``.  The asymmetric error side appropriate
to each residual is used.

The models are not nested, so a raw Delta-chi2 does not follow the usual Wilks
chi-square distribution.  ``--statistic aic`` and ``--statistic bic`` provide
penalized rankings.  ``--statistic bootstrap`` uses pseudo-data generated from
the less-favored model to calibrate the observed Delta-chi2 and report a
one-sided p-value and Gaussian-equivalent significance.

Examples
--------
Quick information-criterion comparison:

    python test_knee_shape.py --statistic aic

Calibrated non-nested significance:

    python test_knee_shape.py --statistic bootstrap --n-bootstrap 5000

Use a different LHAASO hadronic-interaction reconstruction:

    python test_knee_shape.py --hadronic-model EPOS-LHC --statistic bootstrap

Run the same test for helium and save a separate figure:

    python test_knee_shape.py --species He --statistic bootstrap
"""

import argparse
import os
from dataclasses import dataclass
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from matplotlib.lines import Line2D

import kiss_reader


STYLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "EVA.mplstyle"
)
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures"
)

HADRONIC_MODELS = {
    "QGSJET-II-04": "LHAASO_QGSJET-II-04",
    "EPOS-LHC": "LHAASO_EPOS-LHC",
    "SIBYLL-2.3d": "LHAASO_SIBYLL-2.3d",
}

MODEL_LABELS = {
    "cutoff": "power law + exponential cutoff",
    "break": "power law + break",
}
MODEL_COLORS = {
    "cutoff": "tab:blue",
    "break": "tab:red",
}

DEFAULT_SMOOTHNESS = 0.1
DEFAULT_PLOT_SLOPE = 2.7
DEFAULT_MIN_ENERGY = {
    "H": 3.0e5,
    "He": 6.0e5,
}
DEFAULT_SYS_CORRELATION = 0.5


@dataclass
class SpectrumData:
    energy: np.ndarray
    flux: np.ndarray
    error_lo: np.ndarray
    error_up: np.ndarray
    stat_lo: np.ndarray
    stat_up: np.ndarray
    sys_lo: np.ndarray
    sys_up: np.ndarray
    pivot: float


@dataclass
class FitResult:
    model: str
    names: tuple
    values: np.ndarray
    errors: np.ndarray
    chi2: float
    npar: int
    npoints: int
    valid: bool

    @property
    def ndof(self):
        return self.npoints - self.npar


def default_output(species):
    return os.path.join(
        FIGURES_DIR, f"EVA_test_knee_shape_{species}.pdf"
    )


def load_spectrum_data(
    hadronic_model, species, emin=None, emax=None
):
    """Load one LHAASO H or He reconstruction."""
    experiment = HADRONIC_MODELS[hadronic_model]
    energy, flux, stat_lo, stat_up, sys_lo, sys_up = (
        kiss_reader.load_experiment(experiment, species)
    )

    if emin is None:
        emin = DEFAULT_MIN_ENERGY[species]

    mask = np.ones(energy.size, dtype=bool)
    if emin is not None:
        mask &= energy >= emin
    if emax is not None:
        mask &= energy <= emax

    arrays = [
        array[mask]
        for array in (
            energy, flux, stat_lo, stat_up, sys_lo, sys_up
        )
    ]
    energy, flux, stat_lo, stat_up, sys_lo, sys_up = arrays
    if energy.size < 6:
        raise ValueError(
            "the selected energy range contains fewer than six data points"
        )

    error_lo = np.hypot(stat_lo, sys_lo)
    error_up = np.hypot(stat_up, sys_up)
    pivot = 10.0 ** np.mean(np.log10(energy))
    return SpectrumData(
        energy=energy,
        flux=flux,
        error_lo=error_lo,
        error_up=error_up,
        stat_lo=stat_lo,
        stat_up=stat_up,
        sys_lo=sys_lo,
        sys_up=sys_up,
        pivot=pivot,
    )


def cutoff_model(energy, params, pivot):
    """Power law multiplied by a simple exponential cutoff."""
    log10_k, alpha, log10_ecut = params
    ecut = 10.0 ** log10_ecut
    log_flux = (
        np.log(10.0) * log10_k
        - alpha * np.log(energy / pivot)
        - (energy - pivot) / ecut
    )
    return np.exp(log_flux)


def break_model(energy, params, pivot, smoothness):
    """One-break SBPL, normalized to K at the pivot energy."""
    log10_k, alpha, delta_alpha, log10_eb = params
    eb = 10.0 ** log10_eb
    log_transition = np.logaddexp(
        0.0, np.log(energy / eb) / smoothness
    )
    log_transition_pivot = np.logaddexp(
        0.0, np.log(pivot / eb) / smoothness
    )
    log_flux = (
        np.log(10.0) * log10_k
        - alpha * np.log(energy / pivot)
        - delta_alpha * smoothness
        * (log_transition - log_transition_pivot)
    )
    return np.exp(log_flux)


def evaluate_model(model, energy, params, pivot, smoothness):
    if model == "cutoff":
        return cutoff_model(energy, params, pivot)
    if model == "break":
        return break_model(energy, params, pivot, smoothness)
    raise ValueError(f"unknown model '{model}'")


def covariance_matrix(data, prediction, observed, sys_correlation):
    """Covariance with independent statistics and partly correlated systematics."""
    use_upper = prediction > observed
    stat = np.where(use_upper, data.stat_up, data.stat_lo)
    sys = np.where(use_upper, data.sys_up, data.sys_lo)
    covariance = np.diag(
        stat ** 2 + (1.0 - sys_correlation) * sys ** 2
    )
    covariance += sys_correlation * np.outer(sys, sys)
    return covariance


def residual_chi2(
    prediction, observed, data, sys_correlation
):
    """Generalized chi-square using the residual-dependent covariance."""
    residual = observed - prediction
    covariance = covariance_matrix(
        data, prediction, observed, sys_correlation
    )
    return residual @ np.linalg.solve(covariance, residual)


def power_law_initial_guess(data):
    slope, intercept = np.polyfit(
        np.log(data.energy / data.pivot), np.log(data.flux), 1
    )
    return np.log10(np.exp(intercept)), -slope


def fit_shape(
    model,
    data,
    smoothness=DEFAULT_SMOOTHNESS,
    sys_correlation=DEFAULT_SYS_CORRELATION,
    observed=None,
    start=None,
    run_hesse=True,
):
    """Fit one shape with Minuit and return a compact fit result."""
    y = data.flux if observed is None else np.asarray(observed)
    log10_k0, alpha0 = power_law_initial_guess(data)
    log10_emin = np.log10(np.min(data.energy))
    log10_emax = np.log10(np.max(data.energy))

    if model == "cutoff":
        names = ("log10K", "alpha", "log10Ecut")
        initial = (
            np.array(start, dtype=float)
            if start is not None
            else np.array([log10_k0, alpha0, log10_emax - 0.2])
        )

        def objective(log10K, alpha, log10Ecut):
            prediction = cutoff_model(
                data.energy, (log10K, alpha, log10Ecut), data.pivot
            )
            return residual_chi2(
                prediction, y, data, sys_correlation
            )

        minuit = Minuit(
            objective,
            log10K=initial[0],
            alpha=initial[1],
            log10Ecut=initial[2],
        )
        minuit.limits["log10K"] = (log10_k0 - 3.0, log10_k0 + 3.0)
        minuit.limits["alpha"] = (1.0, 5.0)
        minuit.limits["log10Ecut"] = (
            log10_emin - 0.5, log10_emax + 2.0
        )
    elif model == "break":
        names = ("log10K", "alpha", "delta_alpha", "log10Eb")
        initial = (
            np.array(start, dtype=float)
            if start is not None
            else np.array([
                log10_k0, alpha0 - 0.2, 0.7, log10_emax - 0.55
            ])
        )

        def objective(log10K, alpha, delta_alpha, log10Eb):
            prediction = break_model(
                data.energy,
                (log10K, alpha, delta_alpha, log10Eb),
                data.pivot,
                smoothness,
            )
            return residual_chi2(
                prediction, y, data, sys_correlation
            )

        minuit = Minuit(
            objective,
            log10K=initial[0],
            alpha=initial[1],
            delta_alpha=initial[2],
            log10Eb=initial[3],
        )
        minuit.limits["log10K"] = (log10_k0 - 3.0, log10_k0 + 3.0)
        minuit.limits["alpha"] = (1.0, 5.0)
        minuit.limits["delta_alpha"] = (0.0, 3.0)
        minuit.limits["log10Eb"] = (log10_emin, log10_emax)
    else:
        raise ValueError(f"unknown model '{model}'")

    minuit.errordef = Minuit.LEAST_SQUARES
    minuit.migrad()
    if not minuit.valid:
        minuit.simplex()
        minuit.migrad()
    if run_hesse and minuit.valid:
        minuit.hesse()

    values = np.array([minuit.values[name] for name in names])
    errors = np.array([
        minuit.errors[name] if run_hesse else np.nan
        for name in names
    ])
    return FitResult(
        model=model,
        names=names,
        values=values,
        errors=errors,
        chi2=float(minuit.fval),
        npar=len(names),
        npoints=data.energy.size,
        valid=bool(minuit.valid and np.isfinite(minuit.fval)),
    )


def statistic_value(result, statistic):
    """Return a score for which smaller values indicate a better fit."""
    if statistic in ("chi2", "bootstrap"):
        return result.chi2
    if statistic == "aic":
        return result.chi2 + 2.0 * result.npar
    if statistic == "bic":
        return result.chi2 + result.npar * np.log(result.npoints)
    raise ValueError(f"unknown statistic '{statistic}'")


def compare_models(cutoff_fit, break_fit, statistic):
    """Return score difference and the preferred model.

    Delta is defined as score(cutoff) - score(break), hence positive values
    favor the broken power law.
    """
    cutoff_score = statistic_value(cutoff_fit, statistic)
    break_score = statistic_value(break_fit, statistic)
    delta = cutoff_score - break_score
    preferred = "break" if delta > 0.0 else "cutoff"
    return cutoff_score, break_score, delta, preferred


def bootstrap_significance(
    data,
    cutoff_fit,
    break_fit,
    n_bootstrap,
    smoothness,
    sys_correlation,
    seed,
):
    """Calibrate Delta-chi2 under the less-favored observed hypothesis."""
    observed_delta = cutoff_fit.chi2 - break_fit.chi2
    null_model = "cutoff" if observed_delta > 0.0 else "break"
    null_fit = cutoff_fit if null_model == "cutoff" else break_fit
    null_prediction = evaluate_model(
        null_model,
        data.energy,
        null_fit.values,
        data.pivot,
        smoothness,
    )

    rng = np.random.default_rng(seed)
    stat = 0.5 * (data.stat_lo + data.stat_up)
    sys = 0.5 * (data.sys_lo + data.sys_up)
    toy_covariance = np.diag(
        stat ** 2 + (1.0 - sys_correlation) * sys ** 2
    )
    toy_covariance += sys_correlation * np.outer(sys, sys)
    deltas = []
    failures = 0
    for _ in range(n_bootstrap):
        toy = rng.multivariate_normal(null_prediction, toy_covariance)

        toy_cutoff = fit_shape(
            "cutoff",
            data,
            smoothness=smoothness,
            sys_correlation=sys_correlation,
            observed=toy,
            start=cutoff_fit.values,
            run_hesse=False,
        )
        toy_break = fit_shape(
            "break",
            data,
            smoothness=smoothness,
            sys_correlation=sys_correlation,
            observed=toy,
            start=break_fit.values,
            run_hesse=False,
        )
        if not (toy_cutoff.valid and toy_break.valid):
            failures += 1
            continue
        deltas.append(toy_cutoff.chi2 - toy_break.chi2)

    deltas = np.asarray(deltas)
    if deltas.size == 0:
        raise RuntimeError("all bootstrap fits failed")

    if observed_delta > 0.0:
        tail_count = np.count_nonzero(deltas >= observed_delta)
    else:
        tail_count = np.count_nonzero(deltas <= observed_delta)

    # The +1 correction avoids reporting a zero p-value from finite toys.
    p_value = (tail_count + 1.0) / (deltas.size + 1.0)
    sigma = max(0.0, NormalDist().inv_cdf(1.0 - p_value))
    p_error = np.sqrt(
        p_value * (1.0 - p_value) / (deltas.size + 1.0)
    )
    return {
        "observed_delta": observed_delta,
        "null_model": null_model,
        "p_value": p_value,
        "p_error": p_error,
        "sigma": sigma,
        "tail_count": int(tail_count),
        "deltas": deltas,
        "failures": failures,
    }


def format_fit(result):
    lines = [
        f"{MODEL_LABELS[result.model]}:",
        f"  chi2 / dof = {result.chi2:.3f} / {result.ndof}",
    ]
    for name, value, error in zip(
        result.names, result.values, result.errors
    ):
        lines.append(f"  {name:12s} = {value:10.5f} +/- {error:.5f}")

    if result.model == "cutoff":
        lines.append(f"  Ecut         = {10.0 ** result.values[2]:.4g} GeV")
    else:
        lines.append(
            f"  alpha_high   = "
            f"{result.values[1] + result.values[2]:.5f}"
        )
        lines.append(f"  Eb           = {10.0 ** result.values[3]:.4g} GeV")
    return "\n".join(lines)


def report_comparison(cutoff_fit, break_fit, statistic, bootstrap=None):
    score_name = "chi2" if statistic == "bootstrap" else statistic
    cutoff_score, break_score, delta, preferred = compare_models(
        cutoff_fit, break_fit, statistic
    )
    other = "cutoff" if preferred == "break" else "break"

    print("\n" + format_fit(cutoff_fit))
    print("\n" + format_fit(break_fit))
    print(f"\nComparison using {score_name.upper()} (smaller is better):")
    print(f"  cutoff score = {cutoff_score:.3f}")
    print(f"  break score  = {break_score:.3f}")
    print(f"  Delta(cutoff - break) = {delta:+.3f}")
    print(
        f"  preferred: {MODEL_LABELS[preferred]} over "
        f"{MODEL_LABELS[other]}"
    )

    if statistic in ("aic", "bic"):
        preferred_weight = 1.0 / (1.0 + np.exp(-abs(delta) / 2.0))
        relative_support = np.exp(-abs(delta) / 2.0)
        qualifier = "Akaike" if statistic == "aic" else "approximate BIC"
        print(f"  {qualifier} weight of preferred model: "
              f"{preferred_weight:.4f}")
        print(f"  relative support for the other model: "
              f"{relative_support:.4g}")
        print("  note: information-criterion weights are not p-values or sigma")
    elif statistic == "chi2":
        print(
            "  note: these models are non-nested; Delta-chi2 alone has no "
            "Wilks-theorem sigma interpretation"
        )

    if bootstrap is not None:
        null_label = MODEL_LABELS[bootstrap["null_model"]]
        print("\nParametric-bootstrap calibration:")
        print(f"  null hypothesis: {null_label}")
        print(
            f"  successful toys: {bootstrap['deltas'].size} "
            f"(failed: {bootstrap['failures']})"
        )
        if bootstrap["tail_count"] == 0:
            print(f"  one-sided p < {bootstrap['p_value']:.5g}")
            print(
                f"  Gaussian-equivalent significance > "
                f"{bootstrap['sigma']:.3f} sigma"
            )
        else:
            print(
                f"  one-sided p = {bootstrap['p_value']:.5g} "
                f"+/- {bootstrap['p_error']:.2g}"
            )
            print(
                f"  Gaussian-equivalent significance = "
                f"{bootstrap['sigma']:.3f} sigma"
            )


def plot_fits(
    data,
    cutoff_fit,
    break_fit,
    hadronic_model,
    species,
    statistic,
    output,
    smoothness,
    sys_correlation,
    plot_slope,
    bootstrap=None,
):
    energy_grid = np.logspace(
        np.log10(np.min(data.energy)),
        np.log10(np.max(data.energy)),
        500,
    )
    cutoff_grid = cutoff_model(
        energy_grid, cutoff_fit.values, data.pivot
    )
    break_grid = break_model(
        energy_grid, break_fit.values, data.pivot, smoothness
    )
    cutoff_data = cutoff_model(
        data.energy, cutoff_fit.values, data.pivot
    )
    break_data = break_model(
        data.energy, break_fit.values, data.pivot, smoothness
    )

    cutoff_score, break_score, delta, preferred = compare_models(
        cutoff_fit, break_fit, statistic
    )
    score_name = "chi2" if statistic == "bootstrap" else statistic

    with plt.style.context(STYLE):
        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=(14.5, 12.5), sharex=True,
            gridspec_kw={"height_ratios": (3.0, 1.0)},
        )

        scale = data.energy ** plot_slope
        sys_bottom = (data.flux - data.sys_lo) * scale
        sys_top = (data.flux + data.sys_up) * scale
        ax.vlines(
            data.energy,
            sys_bottom,
            sys_top,
            color="black",
            linewidth=1.1,
            linestyles=":",
            alpha=0.9,
            zorder=3,
        )
        ax.plot(
            data.energy,
            sys_bottom,
            ls="",
            marker="_",
            markersize=7,
            markeredgewidth=1.1,
            color="black",
            zorder=3,
        )
        ax.plot(
            data.energy,
            sys_top,
            ls="",
            marker="_",
            markersize=7,
            markeredgewidth=1.1,
            color="black",
            zorder=3,
        )
        ax.errorbar(
            data.energy,
            data.flux * scale,
            yerr=[
                data.stat_lo * scale,
                data.stat_up * scale,
            ],
            fmt="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            color="black",
            elinewidth=1.3,
            capsize=3.0,
            zorder=4,
        )
        cutoff_line, = ax.plot(
            energy_grid,
            cutoff_grid * energy_grid ** plot_slope,
            color=MODEL_COLORS["cutoff"],
            lw=3.2,
        )
        break_line, = ax.plot(
            energy_grid,
            break_grid * energy_grid ** plot_slope,
            color=MODEL_COLORS["break"],
            lw=3.2,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        particle_symbol = "p" if species == "H" else r"{\rm He}"
        particle_name = "proton" if species == "H" else "helium"
        ax.set_ylabel(
            rf"$E^{{{plot_slope:g}}} I_{{{particle_symbol}}}(E)$ "
            rf"[GeV$^{{{plot_slope - 1.0:g}}}$ "
            r"m$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        legend_handles = [
            Line2D(
                [], [], ls="", marker="o", markersize=7,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.5,
                label=f"LHAASO {hadronic_model}",
            ),
            cutoff_line,
            break_line,
            Line2D(
                [], [], ls="", marker="_", markersize=10,
                markeredgewidth=1.3, color="black",
                label="statistical",
            ),
            Line2D(
                [], [], ls=":", marker="_", markersize=8,
                markeredgewidth=1.1, lw=1.1, color="black",
                label="systematic",
            ),
        ]
        legend_handles[1].set_label(
            rf"cutoff: $\chi^2/{cutoff_fit.ndof}"
            rf"={cutoff_fit.chi2:.1f}/{cutoff_fit.ndof}$"
        )
        legend_handles[2].set_label(
            rf"break: $\chi^2/{break_fit.ndof}"
            rf"={break_fit.chi2:.1f}/{break_fit.ndof}$"
        )
        ax.legend(
            handles=legend_handles, loc="best", fontsize=16, ncol=2
        )

        ax.set_title(
            f"LHAASO {particle_name} shape comparison "
            f"({hadronic_model}, "
            rf"$\rho_{{\rm sys}}={sys_correlation:g}$)"
        )
        score_label = r"\chi^2" if score_name == "chi2" else score_name.upper()
        comparison_text = (
            f"{MODEL_LABELS[preferred]} preferred\n"
            rf"$\Delta {score_label}={delta:+.2f}$"
        )
        if bootstrap is not None:
            relation = ">" if bootstrap["tail_count"] == 0 else "="
            p_relation = "<" if bootstrap["tail_count"] == 0 else "="
            comparison_text += (
                "\n"
                rf"$p{p_relation}{bootstrap['p_value']:.3g}$, "
                rf"$Z{relation}{bootstrap['sigma']:.2f}\sigma$"
            )
        ax.text(
            0.98, 0.96, comparison_text,
            transform=ax.transAxes,
            ha="right", va="top", fontsize=16,
        )
        ecut_pev = 10.0 ** cutoff_fit.values[2] / 1.0e6
        ecut_error_pev = (
            np.log(10.0) * ecut_pev * cutoff_fit.errors[2]
        )
        eb_pev = 10.0 ** break_fit.values[3] / 1.0e6
        eb_error_pev = (
            np.log(10.0) * eb_pev * break_fit.errors[3]
        )
        ax.text(
            0.02, 0.27,
            rf"$E_{{\rm cut}}={ecut_pev:.2f}"
            rf"\pm{ecut_error_pev:.2f}\ {{\rm PeV}}$",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=19,
            color=MODEL_COLORS["cutoff"],
        )
        ax.text(
            0.02, 0.21,
            rf"$E_b={eb_pev:.2f}"
            rf"\pm{eb_error_pev:.2f}\ {{\rm PeV}}$",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=19,
            color=MODEL_COLORS["break"],
        )

        cutoff_sigma = np.where(
            cutoff_data > data.flux, data.error_up, data.error_lo
        )
        break_sigma = np.where(
            break_data > data.flux, data.error_up, data.error_lo
        )
        ax_res.axhline(0.0, color="0.3", lw=1.5)
        ax_res.plot(
            data.energy,
            (data.flux - cutoff_data) / cutoff_sigma,
            marker="o",
            color=MODEL_COLORS["cutoff"],
            lw=1.5,
            label="cutoff",
        )
        ax_res.plot(
            data.energy,
            (data.flux - break_data) / break_sigma,
            marker="s",
            color=MODEL_COLORS["break"],
            lw=1.5,
            label="break",
        )
        ax_res.set_xscale("log")
        ax_res.set_xlabel(r"$E$ [GeV]")
        ax_res.set_ylabel(r"pull")
        ax_res.set_ylim(-3.5, 3.5)

        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit LHAASO H or He with an exponential-cutoff power law and "
            "a smoothed broken power law, then compare the shapes."
        )
    )
    parser.add_argument(
        "--statistic",
        choices=("chi2", "aic", "bic", "bootstrap"),
        default="bootstrap",
        help=(
            "model-comparison statistic; bootstrap calibrates Delta-chi2 "
            "with pseudo-experiments (default: bootstrap)"
        ),
    )
    parser.add_argument(
        "--species",
        choices=("H", "He"),
        default="H",
        help="LHAASO species to fit (default: H)",
    )
    parser.add_argument(
        "--hadronic-model",
        choices=tuple(HADRONIC_MODELS),
        default="QGSJET-II-04",
        help="LHAASO hadronic-interaction reconstruction to fit",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=None,
        help=(
            "minimum fitted total energy in GeV "
            "(default: 3e5 for H, 6e5 for He)"
        ),
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=None,
        help="maximum fitted total energy in GeV (default: table maximum)",
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=DEFAULT_SMOOTHNESS,
        help="fixed SBPL smoothness w (default: %(default)s)",
    )
    parser.add_argument(
        "--sys-correlation",
        type=float,
        default=DEFAULT_SYS_CORRELATION,
        help=(
            "common correlation coefficient for systematic errors; "
            "0 is independent and 1 fully correlated (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="number of pseudo-experiments for --statistic bootstrap",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for the bootstrap",
    )
    parser.add_argument(
        "--plot-slope",
        type=float,
        default=DEFAULT_PLOT_SLOPE,
        help="multiply the plotted flux by E**slope",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "output PDF path (default: species-specific file in "
            "mcmc/figures)"
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="print the comparison without creating a figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoothness <= 0.0:
        raise ValueError("--smoothness must be positive")
    if not 0.0 <= args.sys_correlation <= 1.0:
        raise ValueError("--sys-correlation must lie between 0 and 1")
    if args.statistic == "bootstrap" and args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")

    data = load_spectrum_data(
        args.hadronic_model,
        args.species,
        emin=args.emin,
        emax=args.emax,
    )
    particle_name = "proton" if args.species == "H" else "helium"
    print(
        f"loaded {data.energy.size} LHAASO {args.hadronic_model} "
        f"{particle_name} "
        f"points from {data.energy[0]:.4g} to {data.energy[-1]:.4g} GeV"
    )
    print(f"pivot energy E0 = {data.pivot:.4g} GeV")
    print(
        f"systematic-error correlation coefficient rho = "
        f"{args.sys_correlation:g}"
    )

    cutoff_fit = fit_shape(
        "cutoff",
        data,
        smoothness=args.smoothness,
        sys_correlation=args.sys_correlation,
    )
    break_fit = fit_shape(
        "break",
        data,
        smoothness=args.smoothness,
        sys_correlation=args.sys_correlation,
    )
    if not (cutoff_fit.valid and break_fit.valid):
        raise RuntimeError("one or both observed-data fits did not converge")

    bootstrap = None
    if args.statistic == "bootstrap":
        bootstrap = bootstrap_significance(
            data,
            cutoff_fit,
            break_fit,
            n_bootstrap=args.n_bootstrap,
            smoothness=args.smoothness,
            sys_correlation=args.sys_correlation,
            seed=args.seed,
        )

    report_comparison(
        cutoff_fit, break_fit, args.statistic, bootstrap=bootstrap
    )
    if not args.no_plot:
        output = (
            default_output(args.species)
            if args.output is None
            else args.output
        )
        plot_fits(
            data,
            cutoff_fit,
            break_fit,
            args.hadronic_model,
            args.species,
            args.statistic,
            output,
            args.smoothness,
            args.sys_correlation,
            args.plot_slope,
            bootstrap=bootstrap,
        )


if __name__ == "__main__":
    main()
