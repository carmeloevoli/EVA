"""MCMC fit of the direct cosmic-ray spectra with smoothed-broken power laws.

Each species is modelled with a single-break SBPL.  The first species (the
reference, H) carries the free break energy; every other species shares the
*same break in rigidity*, i.e. its break energy is fixed to

    E_b(species) = (Z_species / Z_ref) * E_b(ref)

so no extra break parameter is introduced.  The break smoothness ``w`` is held
fixed.  Energy-scale nuisance factors are shared across species: the experiments
in SCALE_EXPERIMENTS get a fitted factor ``f`` (applied to both energy and flux
via the Jacobian), all others are the reference (f = 1).

Per species the minimum fitted energy is Z * E_MIN_P (a fixed rigidity cut).

The posterior and everything needed for plotting is written to a ``.npz``;
plotting lives in ``plot_direct.py``.

Run with:  python mcmc_direct.py
"""

import os

# Keep per-worker BLAS single-threaded so the multiprocessing pool does not
# oversubscribe the cores (the heavy work is the many small covariance solves).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

import numpy as np
import emcee

import kiss_reader
from sbpl import sbpl

NCORES = 16   # processes for the emcee pool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPECIES = ["H", "He", "C", "O", "Fe"]
Z = {"H": 1, "He": 2, "C": 6, "O": 8, "Fe": 26}

# Each species has its own normalization K, but the two slopes (alpha1, alpha2)
# are shared within a slope group: H and He are alone, C/O/Fe share one pair.
SLOPE_GROUP = {"H": "H", "He": "He", "C": "heavy", "O": "heavy", "Fe": "heavy"}

# The break is parametrized as a rigidity break R_b per break group; the energy
# break of a species follows the rigidity scaling, E_b(s) = Z(s) * R_b(group).
# Three independent breaks: protons, helium, and the heavy nuclei.
BREAK_GROUP = {"H": "H", "He": "He", "C": "heavy", "O": "heavy", "Fe": "heavy"}

EXPERIMENTS = ["DAMPE", "CALET", "CREAM", "ISS-CREAM"]
SCALE_EXPERIMENTS = ["CALET", "CREAM", "ISS-CREAM"]   # fitted energy scales

# Which species each experiment contributes.  ISS-CREAM (the evolution of the
# balloon CREAM) provides H, replacing the balloon CREAM proton data.
EXP_SPECIES = {
    "DAMPE": ["H", "He", "C", "O", "Fe"],
    "CALET": ["H", "He", "C", "O", "Fe"],
    "CREAM": ["He", "C", "O", "Fe"],
    "ISS-CREAM": ["H"],
}

E_MIN_P = 1.0e3                  # GeV, proton energy cut; per species it is Z * E_MIN_P
MAX_ENERGY = 1.0e6               # GeV
E0_P = 1.0e3                     # GeV, proton pivot; per species the pivot is Z * E0_P
W_FIXED = 0.1                    # fixed break smoothness

# Systematic errors are treated as fully correlated within energy ranges (a
# coherent shift per range), following Eq. 8 of arXiv:1909.12860: the per-range
# multiplicative nuisance with a Gaussian prior of width = relative systematic
# is marginalized analytically into a block covariance.  Ranges are ~2/decade.
SYS_BINS_PER_DECADE = 2

# Initial guesses.
K0 = {"H": -4.0, "He": -5.0, "C": -7.2, "O": -7.3, "Fe": -8.4}  # at the Z*E0_P pivot
SLOPE0 = {"H": [2.7, 2.9], "He": [2.5, 2.7], "heavy": [2.6, 2.7]}
RB0 = {"H": 4.1, "He": 4.1, "heavy": 4.1}  # initial log10(R_b) per break group
F0 = 1.0                         # initial energy-scale factor

# Flat-prior bounds.
K_BOUNDS = (-10.0, 2.0)
A1_BOUNDS = (2.0, 3.5)
A2_BOUNDS = (2.0, 4.0)
RB_BOUNDS = (3.0, 6.0)           # log10(R_b) rigidity break
F_BOUNDS = (0.8, 1.2)


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------
# theta = [ log10K per species, (alpha1, alpha2) per slope group,
#           log10Rb per break group, f per scale experiment ]

SLOPE_GROUPS = list(dict.fromkeys(SLOPE_GROUP[s] for s in SPECIES))
BREAK_GROUPS = list(dict.fromkeys(BREAK_GROUP[s] for s in SPECIES))

LABELS, PRIOR_BOUNDS, THETA0 = [], [], []
K_INDEX, SLOPE_INDEX, RB_INDEX = {}, {}, {}

for s in SPECIES:
    K_INDEX[s] = len(LABELS)
    LABELS += [rf"$\log_{{10}} K_{{\rm {s}}}$"]
    PRIOR_BOUNDS += [K_BOUNDS]
    THETA0 += [K0[s]]

for g in SLOPE_GROUPS:
    SLOPE_INDEX[g] = len(LABELS)
    LABELS += [rf"$\alpha_{{1,\rm {g}}}$", rf"$\alpha_{{2,\rm {g}}}$"]
    PRIOR_BOUNDS += [A1_BOUNDS, A2_BOUNDS]
    THETA0 += SLOPE0[g]

for g in BREAK_GROUPS:
    RB_INDEX[g] = len(LABELS)
    LABELS += [rf"$\log_{{10}} R_{{b,\rm {g}}}$"]
    PRIOR_BOUNDS += [RB_BOUNDS]
    THETA0 += [RB0[g]]

SCALE_START = len(LABELS)
for e in SCALE_EXPERIMENTS:
    LABELS += [rf"$f_{{\rm {e}}}$"]
    PRIOR_BOUNDS += [F_BOUNDS]
    THETA0 += [F0]

THETA0 = np.array(THETA0)
NDIM = len(LABELS)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/EVA_mcmc_direct.npz")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    """Load all species/experiments above their rigidity cut (Z * E_MIN_P).

    Returns concatenated arrays with, per point, the species and experiment
    name plus the symmetric (fit) and asymmetric (plot) errors.
    """
    keys = ("E", "y", "stat", "sys", "stat_lo", "stat_up", "sys_lo", "sys_up",
            "exp", "sp")
    out = {k: [] for k in keys}
    for name in EXPERIMENTS:
        for s in EXP_SPECIES[name]:
            emin = Z[s] * E_MIN_P
            e, f, slo, sup, ylo, yup = kiss_reader.load_experiment(name, s, emin, MAX_ENERGY)
            out["E"].append(e)
            out["y"].append(f)
            out["stat"].append(0.5 * (slo + sup))   # symmetrized, used in the fit
            out["sys"].append(0.5 * (ylo + yup))
            out["stat_lo"].append(slo)
            out["stat_up"].append(sup)
            out["sys_lo"].append(ylo)
            out["sys_up"].append(yup)
            out["exp"].append(np.full(e.size, name))
            out["sp"].append(np.full(e.size, s))
    return {k: np.concatenate(v) for k, v in out.items()}


def build_sys_groups(data):
    """Group point indices into (experiment, species, energy-range) blocks within
    which the systematic uncertainty is treated as fully correlated."""
    rbin = np.floor(SYS_BINS_PER_DECADE * np.log10(data["E"])).astype(int)
    groups = {}
    for i in range(data["E"].size):
        key = (data["exp"][i], data["sp"][i], rbin[i])
        groups.setdefault(key, []).append(i)
    return [np.array(v) for v in groups.values()]


# ---------------------------------------------------------------------------
# Model and probability
# ---------------------------------------------------------------------------

def unpack(theta):
    """Return (spec, scales): spec[s] = (log10K, alpha1, alpha2, log10Eb) with
    slopes shared within the slope group and the energy break set by the break
    group's rigidity break, E_b(s) = Z(s) * R_b(group); scales[exp] = factor."""
    spec = {}
    for s in SPECIES:
        log10K = theta[K_INDEX[s]]
        gi = SLOPE_INDEX[SLOPE_GROUP[s]]
        a1, a2 = theta[gi], theta[gi + 1]
        log10Rb = theta[RB_INDEX[BREAK_GROUP[s]]]
        log10Eb = log10Rb + np.log10(Z[s])
        spec[s] = (log10K, a1, a2, log10Eb)
    scales = {e: theta[SCALE_START + i] for i, e in enumerate(SCALE_EXPERIMENTS)}
    return spec, scales


def species_model(theta, s, E):
    """SBPL intensity of species ``s`` at energies ``E`` for parameters theta."""
    log10K, a1, a2, log10Eb = unpack(theta)[0][s]
    return sbpl(E, 10.0 ** log10K, [a1, a2], 10.0 ** log10Eb, W_FIXED, E0=Z[s] * E0_P)


def log_prior(theta):
    for value, (lo, hi) in zip(theta, PRIOR_BOUNDS):
        if not (lo < value < hi):
            return -np.inf
    return 0.0


def log_likelihood(theta, E, y, stat, sys, exp, sp, groups):
    spec, scales = unpack(theta)

    # Energy-scale factor per point.  An energy scale f sends (E, phi) ->
    # (f E, phi / f); applying it to the model instead of the data, the
    # prediction at a measured energy E_i is  f * I(f E_i).
    f = np.ones_like(E)
    for name, value in scales.items():
        f[exp == name] = value

    pred = np.empty_like(E)
    for s in SPECIES:
        mask = sp == s
        log10K, a1, a2, log10Eb = spec[s]
        pred[mask] = f[mask] * sbpl(f[mask] * E[mask], 10.0 ** log10K, [a1, a2],
                                    10.0 ** log10Eb, W_FIXED, E0=Z[s] * E0_P)

    r = y - pred

    # Block covariance (Eq. 8 of arXiv:1909.12860, systematics marginalized):
    # statistical errors on the diagonal; within each energy range the
    # systematic errors are fully correlated (a single coherent shift).
    chi2 = 0.0
    for idx in groups:
        rb = r[idx]
        sg = sys[idx]
        cov = np.diag(stat[idx] ** 2) + np.outer(sg, sg)
        chi2 += rb @ np.linalg.solve(cov, rb)
    return -0.5 * chi2


def log_probability(theta, E, y, stat, sys, exp, sp, groups):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, E, y, stat, sys, exp, sp, groups)


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------

def run_mcmc(data, nwalkers=64, nsteps=25000, burnin=5000, seed=42):
    rng = np.random.default_rng(seed)
    pos = THETA0 + 1e-3 * rng.standard_normal((nwalkers, NDIM))

    groups = build_sys_groups(data)
    args = (data["E"], data["y"], data["stat"], data["sys"],
            data["exp"], data["sp"], groups)

    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, NDIM, log_probability, args=args, pool=pool,
        )
        sampler.run_mcmc(pos, nsteps, progress=True)

    print(f"\nmean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"autocorrelation time: max={np.nanmax(tau):.1f}, mean={np.nanmean(tau):.1f} steps")
    print(f"chain length / max tau = {nsteps / np.nanmax(tau):.0f} "
          f"(>50 recommended); post-burn-in independent samples ~ "
          f"{nwalkers * (nsteps - burnin) / np.nanmax(tau):.0f}")
    worst = int(np.nanargmax(tau))
    print(f"slowest parameter: {LABELS[worst]} (tau={tau[worst]:.1f})")

    return sampler.get_chain(discard=burnin, thin=15, flat=True)


def print_summary(flat):
    print("\nPosterior (median and 16th/84th percentiles):")
    for i, label in enumerate(LABELS):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        print(f"  {label:22s} = {q50:8.3f}  (+{q84 - q50:.3f} / -{q50 - q16:.3f})")


# ---------------------------------------------------------------------------

def main():
    data = load_data()
    print(f"loaded {data['E'].size} data points "
          f"({', '.join(SPECIES)} from {', '.join(EXPERIMENTS)})")

    flat = run_mcmc(data)
    print(f"{flat.shape[0]} posterior samples after burn-in/thinning")
    print_summary(flat)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.savez(
        OUTPUT,
        samples=flat,
        labels=np.array(LABELS),
        data_E=data["E"],
        data_y=data["y"],
        data_stat_lo=data["stat_lo"],
        data_stat_up=data["stat_up"],
        data_sys_lo=data["sys_lo"],
        data_sys_up=data["sys_up"],
        data_exp=data["exp"],
        data_sp=data["sp"],
    )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
