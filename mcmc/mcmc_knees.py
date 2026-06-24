"""MCMC fit of the all-particle cosmic-ray spectrum across the knees.

A three-break smoothed broken power law is fitted from below the first knee
(~PeV) to beyond the second knee (~100 PeV).  The three breaks describe:

    break 1 (~few PeV)   : the first knee   (steepening, alpha1 -> alpha2)
    break 2 (~10-20 PeV) : a hardening      (alpha2 -> alpha3, alpha3 < alpha2)
    break 3 (~100 PeV)   : the second knee  (steepening, alpha3 -> alpha4)

Data: LHAASO + TALE + IceTop/IceCube all-particle spectra.  Statistical and
systematic errors are simply added in quadrature (Gaussian likelihood).

The posterior and everything needed for plotting is written to a ``.npz``;
plotting lives in ``plot_knees.py``.

Run with:  python mcmc_knees.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

import numpy as np
import emcee

from sbpl import sbpl

NCORES = 16

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# (experiment label, table file).  All are all-particle, total-energy tables.
DATASETS = [
    ("LHAASO", "LHAASO_QGSJET-II-04_allParticle_totalEnergy.txt"),
    ("TALE", "TALE_allParticle_totalEnergy.txt"),
    ("IceTop-IceCube", "IceTop_IceCube_SIBYLL-2.1_allParticle_totalEnergy.txt"),
    ("KASCADE", "KASCADE_Kuznetsov2024_allParticle_totalEnergy.txt"),
    ("TUNKA-133", "TUNKA-133_allParticle_totalEnergy.txt"),
]

MIN_ENERGY = 3.0e5     # GeV (~0.3 PeV, below the first knee)
MAX_ENERGY = 1.0e9     # GeV (~1 EeV, below the ankle)
E0 = 1.0e6             # GeV, pivot energy (1 PeV) for the normalization K
W_FIXED = 0.1          # fixed break smoothness (shared by the three breaks)

# Experiments with a fitted energy scale; LHAASO is the reference (f = 1).
SCALE_EXPERIMENTS = ["TALE", "IceTop-IceCube", "KASCADE", "TUNKA-133"]

# theta = [ log10K, alpha1..4, log10E1..3, f per scale experiment ]
SPECTRAL_LABELS = [r"$\log_{10} K$",
                   r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$\alpha_4$",
                   r"$\log_{10} E_1$", r"$\log_{10} E_2$", r"$\log_{10} E_3$"]
NDIM_SPECTRAL = len(SPECTRAL_LABELS)
SCALE_START = NDIM_SPECTRAL
LABELS = SPECTRAL_LABELS + [rf"$f_{{\rm {e.split('-')[0]}}}$" for e in SCALE_EXPERIMENTS]
NDIM = len(LABELS)

E1_INDEX = 5         # the three break energies are theta[5], theta[6], theta[7]

# Initial guess: first knee ~4 PeV, hardening ~16 PeV, second knee ~100 PeV.
THETA0 = np.array([-11.6, 2.7, 3.1, 2.9, 3.3, 6.6, 7.2, 8.0]
                  + [1.0] * len(SCALE_EXPERIMENTS))

PRIOR_BOUNDS = [
    (-15.0, -8.0),   # log10K
    (2.0, 3.5),      # alpha1 (below first knee)
    (2.5, 4.0),      # alpha2 (after first knee)
    (2.5, 4.0),      # alpha3 (after hardening)
    (2.5, 4.5),      # alpha4 (after second knee)
    (6.0, 7.3),      # log10 E1  (first knee)
    (6.6, 7.9),      # log10 E2  (hardening)
    (7.5, 8.7),      # log10 E3  (second knee)
] + [(0.8, 1.2)] * len(SCALE_EXPERIMENTS)   # energy-scale factors

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/EVA_mcmc_knees.npz")
KISS_TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiss_tables")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_allparticle(filename, emin, emax):
    """Read an all-particle total-energy table; return E, y, err_lo, err_up with
    statistical and systematic errors added in quadrature."""
    path = os.path.join(KISS_TABLES_DIR, filename)
    x, y, slo, sup, ylo, yup = np.loadtxt(path, usecols=range(6), unpack=True)
    err_lo = np.sqrt(slo ** 2 + ylo ** 2)
    err_up = np.sqrt(sup ** 2 + yup ** 2)
    mask = (x >= emin) & (x <= emax)
    return x[mask], y[mask], err_lo[mask], err_up[mask]


def load_data():
    """Load and concatenate the all-particle datasets within the fit range."""
    out = {k: [] for k in ("E", "y", "err_lo", "err_up", "yerr", "exp")}
    for name, fname in DATASETS:
        e, y, lo, up = load_allparticle(fname, MIN_ENERGY, MAX_ENERGY)
        out["E"].append(e)
        out["y"].append(y)
        out["err_lo"].append(lo)
        out["err_up"].append(up)
        out["yerr"].append(0.5 * (lo + up))
        out["exp"].append(np.full(e.size, name))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Model and probability
# ---------------------------------------------------------------------------

def model(theta, E):
    """Three-break SBPL all-particle intensity (spectral parameters only)."""
    log10K = theta[0]
    alphas = theta[1:5]
    breaks = 10.0 ** theta[5:8]
    return sbpl(E, 10.0 ** log10K, alphas, breaks, W_FIXED, E0=E0)


def energy_scales(theta, exp):
    """Per-point energy-scale factor f (LHAASO = 1)."""
    f = np.ones(exp.shape)
    for name, value in zip(SCALE_EXPERIMENTS, theta[SCALE_START:]):
        f[exp == name] = value
    return f


def log_prior(theta):
    for value, (lo, hi) in zip(theta, PRIOR_BOUNDS):
        if not (lo < value < hi):
            return -np.inf
    # break energies must be ordered: E1 < E2 < E3
    e1, e2, e3 = theta[E1_INDEX:E1_INDEX + 3]
    if not (e1 < e2 < e3):
        return -np.inf
    return 0.0


def log_likelihood(theta, E, y, yerr, exp):
    # An energy scale f sends (E, phi) -> (f E, phi / f); applied to the model,
    # the prediction at a measured energy E_i is  f * I(f E_i).
    f = energy_scales(theta, exp)
    pred = f * model(theta, f * E)
    return -0.5 * np.sum(((y - pred) / yerr) ** 2)


def log_probability(theta, E, y, yerr, exp):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, E, y, yerr, exp)


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------

def run_mcmc(data, nwalkers=48, nsteps=30000, burnin=6000, seed=42):
    rng = np.random.default_rng(seed)
    pos = THETA0 + 1e-3 * rng.standard_normal((nwalkers, NDIM))

    args = (data["E"], data["y"], data["yerr"], data["exp"])
    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, NDIM, log_probability, args=args, pool=pool,
        )
        sampler.run_mcmc(pos, nsteps, progress=True)

    print(f"\nmean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"autocorrelation time: max={np.nanmax(tau):.1f}, mean={np.nanmean(tau):.1f} steps")
    print(f"chain length / max tau = {nsteps / np.nanmax(tau):.0f} (>50 recommended)")
    print(f"slowest parameter: {LABELS[int(np.nanargmax(tau))]} (tau={np.nanmax(tau):.1f})")

    return sampler.get_chain(discard=burnin, thin=15, flat=True)


def print_summary(flat):
    print("\nPosterior (median and 16th/84th percentiles):")
    for i, label in enumerate(LABELS):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        print(f"  {label:16s} = {q50:9.3f}  (+{q84 - q50:.3f} / -{q50 - q16:.3f})")


def main():
    data = load_data()
    print(f"loaded {data['E'].size} all-particle points from "
          f"{', '.join(n for n, _ in DATASETS)}")

    flat = run_mcmc(data)
    print(f"{flat.shape[0]} posterior samples after burn-in/thinning")
    print_summary(flat)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.savez(
        OUTPUT,
        samples=flat,
        labels=np.array(LABELS),
        E0=E0,
        w_fixed=W_FIXED,
        min_energy=MIN_ENERGY,
        max_energy=MAX_ENERGY,
        data_E=data["E"],
        data_y=data["y"],
        data_elo=data["err_lo"],
        data_eup=data["err_up"],
        data_exp=data["exp"],
    )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
