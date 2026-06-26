"""Joint MCMC of a two-component (H + He) model from direct data to LHAASO.

Independent rigidity-break patterns describe H and He, while each component
also has its own normalization and spectral slopes:

    I_H(E)  = K_H  * f_H(E)        (Z_H  = 1,  rigidity R = E)
    I_He(E) = K_He * f_He(E / 2)   (Z_He = 2,  rigidity R = E / 2)

The three H and He breaks are fitted separately in rigidity.  A posteriori,
Z-scaling is recovered if R_{i,He} / R_{i,H} is consistent with 1, equivalently
if E_{i,He} / E_{i,H} is consistent with 2.  The model is constrained
simultaneously by:

    * the proton measurement   I_H            (DAMPE, CALET, CREAM, LHAASO)
    * the light measurement    I_H + I_He     (DAMPE, CALET, LHAASO)

Energy-scale nuisances are per experiment (shared across observables), DAMPE =
reference.  Statistical errors remain independent; systematic errors are
correlated within energy blocks for each observable.  DAMPE and LHAASO H and
H+He are not cross-correlated.  CALET H+He is reconstructed from its H and He
tables, so its covariance with the reused H measurements is retained.

Run with:  python mcmc_lhaaso_phe.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

import numpy as np
import emcee

import kiss_reader
from sbpl import sbpl

NCORES = 10

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Proton measurement: (label, kiss_reader experiment).
H_DATASETS = [("DAMPE", "DAMPE"), ("CALET", "CALET"), ("CREAM", "CREAM"),
              ("LHAASO", "LHAASO_QGSJET-II-04")]
# Light (H+He) measurement: (label, total-energy table file).
LIGHT_DATASETS = [
    ("DAMPE", "DAMPE_light_totalEnergy.txt"),
    ("LHAASO", "LHAASO_QGSJET-II-04_light_totalEnergy.txt"),
]
DERIVED_LIGHT_EXPERIMENTS = ["CALET"]
FIT_DATASET_TAG = (
    "H:DAMPE,CALET,CREAM,LHAASO;He:none;"
    "light:DAMPE,CALET-derived,LHAASO"
)

SCALE_EXPERIMENTS = ["CALET", "CREAM", "LHAASO"]      # DAMPE = reference
CORRELATED_EXPERIMENTS = ["DAMPE", "CALET", "CREAM"]
# Kept as output metadata and for compatibility with the plotting script.
# No external H/light correlation coefficient is imposed; the covariance of
# the derived CALET light data is handled explicitly in build_sys_groups().
CROSS_OBSERVABLE_SYS_RHO = {}

MIN_ENERGY = 1.0e3     # GeV
MAX_ENERGY = 1.0e7     # GeV
R0 = 1.0e3             # GV, pivot rigidity where f(R0) = 1
W_FIXED = 0.05
SYS_BINS_PER_DECADE = 2
Z_HE = 2

# theta = [ log10K_H, log10K_He, a1_H, a1_He, a2_H, a2_He,
#           a3_H, a3_He, a4_H, a4_He,
#           log10R1..3_H, log10R1..3_He, f per scale experiment ]
# The spectral slopes and rigidity breaks differ between H and He.
LABELS = [r"$\log_{10} K_{\rm H}$", r"$\log_{10} K_{\rm He}$",
          r"$\alpha_{1,\rm H}$", r"$\alpha_{1,\rm He}$",
          r"$\alpha_{2,\rm H}$", r"$\alpha_{2,\rm He}$",
          r"$\alpha_{3,\rm H}$", r"$\alpha_{3,\rm He}$",
          r"$\alpha_{4,\rm H}$", r"$\alpha_{4,\rm He}$",
          r"$\log_{10} R_{1,\rm H}$",
          r"$\log_{10} R_{2,\rm H}$",
          r"$\log_{10} R_{3,\rm H}$",
          r"$\log_{10} R_{1,\rm He}$",
          r"$\log_{10} R_{2,\rm He}$",
          r"$\log_{10} R_{3,\rm He}$"]
A1_H, A1_HE, A2_H, A2_HE = 2, 3, 4, 5
A3_H, A3_HE, A4_H, A4_HE = 6, 7, 8, 9
H_R_INDEX = 10
HE_R_INDEX = 13
R_INDEX = H_R_INDEX
SCALE_START = 16
LABELS += [rf"$f_{{\rm {e}}}$" for e in SCALE_EXPERIMENTS]
NDIM = len(LABELS)

THETA0 = np.array([-4.07, -4.92, 2.59, 2.51, 2.90, 2.90, 2.63, 2.63,
                   3.43, 3.43,
                   4.22, 5.34, 6.40,
                   4.22, 5.34, 6.40]
                  + [1.12, 1.15, 0.83])

PRIOR_BOUNDS = [
    (-8.0, 2.0),     # log10K_H
    (-8.0, 2.0),     # log10K_He
    (2.0, 4.0),      # alpha1_H
    (2.0, 4.0),      # alpha1_He
    (2.0, 4.0),      # alpha2_H
    (2.0, 4.0),      # alpha2_He
    (2.0, 4.0),      # alpha3_H
    (2.0, 4.0),      # alpha3_He
    (2.0, 4.5),      # alpha4_H
    (2.0, 4.5),      # alpha4_He
    (3.5, 5.0),      # log10 R1_H (~10 TV)
    (4.5, 6.0),      # log10 R2_H (~100 TV)
    (5.8, 6.8),      # log10 R3_H (~3 PV)
    (3.5, 5.0),      # log10 R1_He (~10 TV)
    (4.5, 6.0),      # log10 R2_He (~100 TV)
    (5.8, 6.8),      # log10 R3_He (~3 PV)
    (0.7, 1.3), (0.7, 1.3), (0.7, 1.3),   # widened energy-scale priors
]

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/EVA_mcmc_lhaaso_phe.npz")
KISS_TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiss_tables")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _read_light(filename, emin, emax):
    path = os.path.join(KISS_TABLES_DIR, filename)
    x, y, slo, sup, ylo, yup = np.loadtxt(path, usecols=range(6), unpack=True)
    mask = (x >= emin) & (x <= emax)
    return (a[mask] for a in (x, y, slo, sup, ylo, yup))


def _loglog_interpolate_with_errors(x_new, x, y, *errors):
    """Log-log interpolate y and propagate independent endpoint errors."""
    if np.any(x_new < x[0]) or np.any(x_new > x[-1]):
        raise ValueError("interpolation requested outside the source range")

    hi = np.searchsorted(x, x_new, side="right")
    hi = np.clip(hi, 1, x.size - 1)
    lo = hi - 1
    t = ((np.log(x_new) - np.log(x[lo]))
         / (np.log(x[hi]) - np.log(x[lo])))
    y_new = np.exp((1.0 - t) * np.log(y[lo]) + t * np.log(y[hi]))

    d_lo = y_new * (1.0 - t) / y[lo]
    d_hi = y_new * t / y[hi]
    propagated = [
        np.sqrt((d_lo * error[lo]) ** 2 + (d_hi * error[hi]) ** 2)
        for error in errors
    ]
    return (y_new, *propagated)


def _build_calet_light(emin, emax):
    """Construct CALET H+He on the proton grid with propagated errors.

    Helium is interpolated in log(E)-log(I) space.  Statistical and systematic
    endpoint errors are propagated through that interpolation, then the H and
    interpolated-He errors are added in quadrature, separately for their lower
    and upper values.  This reconstruction assumes the tabulated H and He
    errors are independent because no cross-species covariance is available.
    """
    h = kiss_reader.load_experiment("CALET", "H", emin, emax)
    he = kiss_reader.load_experiment("CALET", "He", 0.0, np.inf)
    e_h, y_h, h_slo, h_sup, h_ylo, h_yup = h
    e_he, y_he, he_slo, he_sup, he_ylo, he_yup = he

    overlap = (e_h >= e_he[0]) & (e_h <= e_he[-1])
    e_h = e_h[overlap]
    y_h = y_h[overlap]
    h_slo, h_sup = h_slo[overlap], h_sup[overlap]
    h_ylo, h_yup = h_ylo[overlap], h_yup[overlap]

    y_he_i, he_slo_i, he_sup_i, he_ylo_i, he_yup_i = (
        _loglog_interpolate_with_errors(
            e_h, e_he, y_he, he_slo, he_sup, he_ylo, he_yup
        )
    )
    y_light = y_h + y_he_i
    stat_lo = np.hypot(h_slo, he_slo_i)
    stat_up = np.hypot(h_sup, he_sup_i)
    sys_lo = np.hypot(h_ylo, he_ylo_i)
    sys_up = np.hypot(h_yup, he_yup_i)
    return (e_h, y_light, stat_lo, stat_up, sys_lo, sys_up,
            0.5 * (h_slo + h_sup), 0.5 * (h_ylo + h_yup))


def load_data():
    keys = ("E", "y", "stat", "sys",
            "stat_lo", "stat_up", "sys_lo", "sys_up",
            "err_lo", "err_up", "light_h_stat", "light_h_sys",
            "exp", "obs")
    out = {k: [] for k in keys}

    def add(e, y, slo, sup, ylo, yup, label, obs,
            light_h_stat=None, light_h_sys=None):
        out["E"].append(e)
        out["y"].append(y)
        out["stat"].append(0.5 * (slo + sup))
        out["sys"].append(0.5 * (ylo + yup))
        out["stat_lo"].append(slo)
        out["stat_up"].append(sup)
        out["sys_lo"].append(ylo)
        out["sys_up"].append(yup)
        out["err_lo"].append(np.sqrt(slo ** 2 + ylo ** 2))
        out["err_up"].append(np.sqrt(sup ** 2 + yup ** 2))
        out["light_h_stat"].append(
            np.zeros(e.size) if light_h_stat is None else light_h_stat
        )
        out["light_h_sys"].append(
            np.zeros(e.size) if light_h_sys is None else light_h_sys
        )
        out["exp"].append(np.full(e.size, label))
        out["obs"].append(np.full(e.size, obs))

    for label, exp_name in H_DATASETS:
        e, y, slo, sup, ylo, yup = kiss_reader.load_experiment(
            exp_name, "H", MIN_ENERGY, MAX_ENERGY)
        add(e, y, slo, sup, ylo, yup, label, "H")
    for label, fname in LIGHT_DATASETS:
        e, y, slo, sup, ylo, yup = _read_light(fname, MIN_ENERGY, MAX_ENERGY)
        add(e, y, slo, sup, ylo, yup, label, "light")
    e, y, slo, sup, ylo, yup, h_stat, h_sys = _build_calet_light(
        MIN_ENERGY, MAX_ENERGY
    )
    add(e, y, slo, sup, ylo, yup, "CALET", "light",
        light_h_stat=h_stat, light_h_sys=h_sys)

    return {k: np.concatenate(v) for k, v in out.items()}


def build_sys_groups(data):
    """Return index blocks and their statistical/systematic covariance pieces.

    DAMPE, CALET, and CREAM use fully correlated half-decade blocks within
    each observable.  LHAASO points are independent.  DAMPE and LHAASO H and
    H+He are not cross-correlated.  CALET H and derived H+He share the proton
    statistical and systematic contributions, which are included here.
    """
    rbin = np.floor(SYS_BINS_PER_DECADE * np.log10(data["E"])).astype(int)
    grouped_indices = {}
    for i in range(data["E"].size):
        exp = data["exp"][i]
        if exp == "CALET" and data["obs"][i] in ("H", "light"):
            key = (exp, rbin[i])
        elif exp in CORRELATED_EXPERIMENTS:
            key = (exp, data["obs"][i], rbin[i])
        else:
            key = ("indep", i)
        grouped_indices.setdefault(key, []).append(i)

    groups = []
    for indices in grouped_indices.values():
        idx = np.asarray(indices)
        stat_cov = np.diag(data["stat"][idx] ** 2)
        sys_cov = np.diag(data["sys"][idx] ** 2)
        exp = data["exp"][idx[0]]

        if exp == "CALET":
            obs = data["obs"][idx]
            h_mode = np.zeros(idx.size)
            he_mode = np.zeros(idx.size)
            h_rows = obs == "H"
            light_rows = obs == "light"
            h_mode[h_rows] = data["sys"][idx[h_rows]]
            h_mode[light_rows] = data["light_h_sys"][idx[light_rows]]
            he_mode[light_rows] = np.sqrt(np.maximum(
                data["sys"][idx[light_rows]] ** 2
                - h_mode[light_rows] ** 2,
                0.0,
            ))
            sys_cov = np.outer(h_mode, h_mode) + np.outer(he_mode, he_mode)

            for a in range(idx.size):
                for b in range(a + 1, idx.size):
                    if {obs[a], obs[b]} != {"H", "light"}:
                        continue
                    h_local = a if obs[a] == "H" else b
                    light_local = b if obs[b] == "light" else a
                    h_global = idx[h_local]
                    light_global = idx[light_local]

                    if np.isclose(
                            data["E"][h_global], data["E"][light_global],
                            rtol=1e-12, atol=0.0):
                        shared_stat = (
                            data["stat"][h_global]
                            * data["light_h_stat"][light_global]
                        )
                        stat_cov[h_local, light_local] = shared_stat
                        stat_cov[light_local, h_local] = shared_stat
        elif exp in CORRELATED_EXPERIMENTS:
            sys_cov = np.outer(data["sys"][idx], data["sys"][idx])

        groups.append((idx, stat_cov, sys_cov))
    return groups


# ---------------------------------------------------------------------------
# Model and probability
# ---------------------------------------------------------------------------

def shape(R, alphas, log10_breaks):
    """Rigidity shape f(R) with component-specific slopes."""
    breaks = 10.0 ** np.asarray(log10_breaks)
    return sbpl(R, 1.0, alphas, breaks, W_FIXED, E0=R0)


def components(theta, E):
    """Return (I_H, I_He) at energies E."""
    h_alphas = [theta[A1_H], theta[A2_H], theta[A3_H], theta[A4_H]]
    he_alphas = [theta[A1_HE], theta[A2_HE], theta[A3_HE], theta[A4_HE]]
    h_breaks = theta[H_R_INDEX:H_R_INDEX + 3]
    he_breaks = theta[HE_R_INDEX:HE_R_INDEX + 3]
    I_H = 10.0 ** theta[0] * shape(E, h_alphas, h_breaks)
    I_He = 10.0 ** theta[1] * shape(E / Z_HE, he_alphas, he_breaks)
    return I_H, I_He


def energy_scales(theta, exp):
    f = np.ones(exp.shape)
    for name, value in zip(SCALE_EXPERIMENTS, theta[SCALE_START:]):
        f[exp == name] = value
    return f


def log_prior(theta):
    for value, (lo, hi) in zip(theta, PRIOR_BOUNDS):
        if not (lo < value < hi):
            return -np.inf
    for start in (H_R_INDEX, HE_R_INDEX):
        r1, r2, r3 = theta[start:start + 3]
        if not (r1 < r2 < r3):
            return -np.inf
    return 0.0


def log_likelihood(theta, E, y, stat, sys, exp, obs, groups):
    f = energy_scales(theta, exp)
    I_H, I_He = components(theta, f * E)
    pred = np.empty_like(y)
    pred[obs == "H"] = I_H[obs == "H"]
    pred[obs == "He"] = I_He[obs == "He"]
    pred[obs == "light"] = (I_H + I_He)[obs == "light"]
    known = np.isin(obs, ("H", "He", "light"))
    if not np.all(known):
        raise ValueError(f"unknown observables: {np.unique(obs[~known])}")
    pred *= f
    r = y - pred
    chi2 = 0.0
    for idx, stat_cov, sys_cov in groups:
        rb = r[idx]
        cov = stat_cov + sys_cov
        chi2 += rb @ np.linalg.solve(cov, rb)
    return -0.5 * chi2


def log_probability(theta, E, y, stat, sys, exp, obs, groups):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, E, y, stat, sys, exp, obs, groups)


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------

def print_parameter_recap():
    if not (len(LABELS) == len(THETA0) == len(PRIOR_BOUNDS) == NDIM):
        raise RuntimeError(
            "inconsistent parameter configuration: LABELS, THETA0, "
            "PRIOR_BOUNDS, and NDIM must have the same length"
        )

    print("\nParameter recap:")
    print(f"  NDIM = {NDIM}")
    print("  Slopes and rigidity breaks are independent for H and He.")
    print("  Z-scaling is checked a posteriori from R_He / R_H.")
    print("  No standalone helium observable is fitted.")
    print("  The CALET He table enters only through the derived H+He spectrum.")
    print("  Priors are uniform open intervals: lo < theta < hi.")
    print("  DAMPE/LHAASO H-light systematic correlations: none")
    print("  CALET light is derived from H+He; shared-H covariance is propagated.")
    print(f"  Fit datasets: {FIT_DATASET_TAG}")
    print(f"  H rigidity breaks: theta[{H_R_INDEX}:{H_R_INDEX + 3}]")
    print(f"  He rigidity breaks: theta[{HE_R_INDEX}:{HE_R_INDEX + 3}]")
    print(f"  Energy-scale nuisances: theta[{SCALE_START}:{NDIM}] "
          f"for {', '.join(SCALE_EXPERIMENTS)}")
    print("\n  idx  parameter                       theta0        prior")
    print("  ---  ------------------------------  --------  ----------------")
    for i, (label, theta0, (lo, hi)) in enumerate(zip(LABELS, THETA0, PRIOR_BOUNDS)):
        print(f"  {i:3d}  {label:30s}  {theta0:8.3f}  ({lo:g}, {hi:g})")


def run_mcmc(data, nwalkers=48, nsteps=30000, burnin=6000, seed=42):
    rng = np.random.default_rng(seed)
    pos = THETA0 + 1e-3 * rng.standard_normal((nwalkers, NDIM))
    groups = build_sys_groups(data)
    args = (data["E"], data["y"], data["stat"], data["sys"],
            data["exp"], data["obs"], groups)
    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, NDIM, log_probability, args=args, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    tau = sampler.get_autocorr_time(quiet=True)
    print(f"\nacceptance={np.mean(sampler.acceptance_fraction):.3f}, "
          f"max tau={np.nanmax(tau):.0f}, chain/tau={nsteps / np.nanmax(tau):.0f} "
          f"(slowest: {LABELS[int(np.nanargmax(tau))]})")
    return sampler.get_chain(discard=burnin, thin=15, flat=True)


def print_summary(flat):
    print("\nPosterior (median, 16th/84th):")
    for i, label in enumerate(LABELS):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        print(f"  {label:26s} = {q50:9.3f}  (+{q84 - q50:.3f} / -{q50 - q16:.3f})")


def print_z_scaling_summary(flat):
    print("\nPosterior Z-scaling check:")
    print("  Z-scaling means R_He/R_H = 1, or E_b,He/E_b,H = 2.")
    names = ("break 1", "break 2", "break 3")
    h_log_r = flat[:, H_R_INDEX:H_R_INDEX + 3]
    he_log_r = flat[:, HE_R_INDEX:HE_R_INDEX + 3]
    for j, name in enumerate(names):
        log_r_ratio = he_log_r[:, j] - h_log_r[:, j]
        energy_ratio = Z_HE * 10.0 ** log_r_ratio
        d16, d50, d84 = np.percentile(log_r_ratio, [16, 50, 84])
        e16, e50, e84 = np.percentile(energy_ratio, [16, 50, 84])
        print(
            f"  {name}: log10(R_He/R_H) = {d50:+.3f} "
            f"(+{d84 - d50:.3f}/-{d50 - d16:.3f}); "
            f"E_He/E_H = {e50:.2f} "
            f"(+{e84 - e50:.2f}/-{e50 - e16:.2f})"
        )


def main():
    print_parameter_recap()
    data = load_data()
    nH = int((data["obs"] == "H").sum())
    nHe = int((data["obs"] == "He").sum())
    nL = int((data["obs"] == "light").sum())
    print(f"loaded {data['E'].size} points (H: {nH}, He: {nHe}, light: {nL})")
    flat = run_mcmc(data)
    print(f"{flat.shape[0]} samples after burn-in/thinning")
    print_summary(flat)
    print_z_scaling_summary(flat)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.savez(OUTPUT, samples=flat, labels=np.array(LABELS),
             R0=R0, w_fixed=W_FIXED, min_energy=MIN_ENERGY, max_energy=MAX_ENERGY,
             model_kind=np.array("independent_h_he_rigidity_breaks"),
             h_r_index=H_R_INDEX, he_r_index=HE_R_INDEX,
             r_index=R_INDEX, scale_start=SCALE_START,
             z_he=Z_HE,
             scale_experiments=np.array(SCALE_EXPERIMENTS),
             cross_observable_sys_experiments=np.array(
                 list(CROSS_OBSERVABLE_SYS_RHO), dtype=str),
             cross_observable_sys_rho=np.array(
                 list(CROSS_OBSERVABLE_SYS_RHO.values()), dtype=float),
             fit_dataset_tag=np.array(FIT_DATASET_TAG),
             derived_light_experiments=np.array(
                 DERIVED_LIGHT_EXPERIMENTS, dtype=str),
             data_E=data["E"], data_y=data["y"],
             data_stat_lo=data["stat_lo"], data_stat_up=data["stat_up"],
             data_sys_lo=data["sys_lo"], data_sys_up=data["sys_up"],
             data_light_h_stat=data["light_h_stat"],
             data_light_h_sys=data["light_h_sys"],
             data_err_lo=data["err_lo"], data_err_up=data["err_up"],
             data_exp=data["exp"], data_obs=data["obs"])
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
