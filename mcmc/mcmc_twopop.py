"""Joint MCMC fit of a two-population Galactic cosmic-ray model.

The differential intensity of species ``s`` is the sum of a low-energy (LE)
and, where present, a high-energy (HE) Galactic population:

    I_s(E) = I_s^LE(E) + I_s^HE(E)

    I_s^p(E) = K_s^p (R/R0)^(-alpha_g^p)
               { [1 + (R/Rb_p)^(1/w_p)]
                 / [1 + (R0/Rb_p)^(1/w_p)] }^(-DeltaAlpha_p w_p)

where ``R = pc/(Ze)``.  The ratio of transition factors ensures that
``K_s^p`` is exactly the population intensity at the pivot rigidity ``R0``.
Below the break the index is ``alpha_g^p`` and above it asymptotes to
``alpha_g^p + DeltaAlpha_p``.

Population content:

    LE: H, He, C, O, Fe
    HE: H, He, C, O, Fe

The break rigidity and steepening are shared by all species in a population.
LE slopes use three groups (H, He, and a common C/O/Fe heavy group); HE has
separate H and He slopes.  HE C, O, and Fe share the helium rigidity shape
and differ from it only through three fitted normalizations.  The SBPL
smoothing is fixed.

Data:

    DAMPE: H, He, C, O, Fe
    CALET: H, He, C, O, Fe
    LHAASO QGSJET-II-04: H, He

DAMPE is the reference energy scale.  CALET and LHAASO have one fitted energy
scale each, shared by all their species.  Statistical errors are independent.
DAMPE and CALET systematic errors are fully correlated within half-decade
energy blocks for a given species; LHAASO bins are treated as independent.

Run with:

    python mcmc_twopop.py

Optional controls:

    python mcmc_twopop.py --ncores 10 --nwalkers 72 --nsteps 30000
"""

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

import emcee
import numpy as np

import kiss_reader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NCORES = 10

SPECIES = ["H", "He", "C", "O", "Fe"]
Z = {s: kiss_reader.SPECIES[s]["Z"] for s in SPECIES}
A = {s: kiss_reader.SPECIES[s]["A"] for s in SPECIES}

POPULATIONS = ["LE", "HE"]
POP_SPECIES = {
    "LE": ["H", "He", "C", "O", "Fe"],
    "HE": ["H", "He", "C", "O", "Fe"],
}

SLOPE_GROUP = {
    "LE": {"H": "H", "He": "He", "C": "heavy", "O": "heavy", "Fe": "heavy"},
    "HE": {"H": "H", "He": "He", "C": "He", "O": "He", "Fe": "He"},
}

EXPERIMENTS = ["DAMPE", "CALET", "LHAASO"]
EXP_READER_NAME = {
    "DAMPE": "DAMPE",
    "CALET": "CALET",
    "LHAASO": "LHAASO_QGSJET-II-04",
}
EXP_SPECIES = {
    "DAMPE": ["H", "He", "C", "O", "Fe"],
    "CALET": ["H", "He", "C", "O", "Fe"],
    "LHAASO": ["H", "He"],
}

# DAMPE defines the reference energy scale.
SCALE_EXPERIMENTS = ["CALET", "LHAASO"]
CORRELATED_EXPERIMENTS = ["DAMPE", "CALET"]

MIN_RIGIDITY = 1.0e3       # GV
MAX_ENERGY = 1.5e7         # GeV, total particle energy
R0 = 1.0e3                 # GV
SYS_BINS_PER_DECADE = 2
BREAK_SMOOTHING = {"LE": 0.1, "HE": 0.1}

# Initial values.  K is the population intensity at R0.
LOG10K0 = {
    "LE": {"H": -4.05, "He": -4.85, "C": -7.05, "O": -7.10, "Fe": -8.25},
    "HE": {
        "H": -5.25,
        "He": -5.85,
        "C": -8.05,
        "O": -8.10,
        "Fe": -9.25,
    },
}
ALPHA0 = {
    "LE": {"H": 2.75, "He": 2.65, "heavy": 2.65},
    "HE": {"H": 2.40, "He": 2.35},
}
LOG10_RB0 = {"LE": 4.0, "HE": 6.0}  # 10 TV and 1 PV
DELTA_ALPHA0 = {"LE": 1.0, "HE": 1.0}

LOG10K_BOUNDS = (-11.0, -2.0)
ALPHA_BOUNDS = {
    "LE": (2.0, 3.5),
    "HE": (1.5, 3.5),
}
LOG10_RB_BOUNDS = {
    "LE": (3.4, 4.7),       # 2.5--50 TV
    "HE": (5.4, 6.6),       # 0.25--4 PV
}
DELTA_ALPHA_BOUNDS = {
    "LE": (0.1, 5.0),
    "HE": (0.1, 5.0),
}
SCALE_BOUNDS = {
    "CALET": (0.8, 1.2),
    "LHAASO": (0.7, 1.3),
}

OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output/EVA_mcmc_twopop.npz",
)


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------

LABELS = []
PRIOR_BOUNDS = []
THETA0 = []

K_INDEX = {pop: {} for pop in POPULATIONS}
ALPHA_INDEX = {pop: {} for pop in POPULATIONS}
RB_INDEX = {}
DELTA_ALPHA_INDEX = {}

for pop in POPULATIONS:
    for s in POP_SPECIES[pop]:
        K_INDEX[pop][s] = len(LABELS)
        LABELS.append(rf"$\log_{{10}} K^{{\rm {pop}}}_{{\rm {s}}}$")
        PRIOR_BOUNDS.append(LOG10K_BOUNDS)
        THETA0.append(LOG10K0[pop][s])

for pop in POPULATIONS:
    groups = list(dict.fromkeys(SLOPE_GROUP[pop].values()))
    for group in groups:
        ALPHA_INDEX[pop][group] = len(LABELS)
        LABELS.append(rf"$\alpha^{{\rm {pop}}}_{{\rm {group}}}$")
        PRIOR_BOUNDS.append(ALPHA_BOUNDS[pop])
        THETA0.append(ALPHA0[pop][group])

for pop in POPULATIONS:
    RB_INDEX[pop] = len(LABELS)
    LABELS.append(rf"$\log_{{10}} R_b^{{\rm {pop}}}$")
    PRIOR_BOUNDS.append(LOG10_RB_BOUNDS[pop])
    THETA0.append(LOG10_RB0[pop])

for pop in POPULATIONS:
    DELTA_ALPHA_INDEX[pop] = len(LABELS)
    LABELS.append(rf"$\Delta\alpha^{{\rm {pop}}}$")
    PRIOR_BOUNDS.append(DELTA_ALPHA_BOUNDS[pop])
    THETA0.append(DELTA_ALPHA0[pop])

SCALE_START = len(LABELS)
for exp in SCALE_EXPERIMENTS:
    LABELS.append(rf"$f_{{\rm {exp}}}$")
    PRIOR_BOUNDS.append(SCALE_BOUNDS[exp])
    THETA0.append(1.0)

THETA0 = np.asarray(THETA0, dtype=float)
NDIM = len(LABELS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    """Load all configured individual-species spectra."""
    keys = (
        "E", "y", "stat", "sys",
        "stat_lo", "stat_up", "sys_lo", "sys_up",
        "err_lo", "err_up", "exp", "sp",
    )
    out = {key: [] for key in keys}

    for exp in EXPERIMENTS:
        reader_name = EXP_READER_NAME[exp]
        for s in EXP_SPECIES[exp]:
            emin = Z[s] * MIN_RIGIDITY
            e, y, stat_lo, stat_up, sys_lo, sys_up = (
                kiss_reader.load_experiment(
                    reader_name, s, emin, MAX_ENERGY
                )
            )
            if e.size == 0:
                raise RuntimeError(
                    f"no {exp} {s} points in the configured fit range"
                )

            out["E"].append(e)
            out["y"].append(y)
            out["stat"].append(0.5 * (stat_lo + stat_up))
            out["sys"].append(0.5 * (sys_lo + sys_up))
            out["stat_lo"].append(stat_lo)
            out["stat_up"].append(stat_up)
            out["sys_lo"].append(sys_lo)
            out["sys_up"].append(sys_up)
            out["err_lo"].append(np.hypot(stat_lo, sys_lo))
            out["err_up"].append(np.hypot(stat_up, sys_up))
            out["exp"].append(np.full(e.size, exp))
            out["sp"].append(np.full(e.size, s))

    return {key: np.concatenate(value) for key, value in out.items()}


def build_covariance_groups(data):
    """Build fixed covariance blocks for the Gaussian likelihood."""
    energy_block = np.floor(
        SYS_BINS_PER_DECADE * np.log10(data["E"])
    ).astype(int)
    grouped = {}

    for i in range(data["E"].size):
        exp = data["exp"][i]
        if exp in CORRELATED_EXPERIMENTS:
            key = (exp, data["sp"][i], energy_block[i])
        else:
            key = ("independent", i)
        grouped.setdefault(key, []).append(i)

    groups = []
    for indices in grouped.values():
        idx = np.asarray(indices, dtype=int)
        covariance = np.diag(data["stat"][idx] ** 2)
        if data["exp"][idx[0]] in CORRELATED_EXPERIMENTS:
            covariance += np.outer(data["sys"][idx], data["sys"][idx])
        else:
            covariance += np.diag(data["sys"][idx] ** 2)
        groups.append((idx, covariance))
    return groups


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def rigidity(E, s):
    """Exact particle rigidity in GV from total energy in GeV."""
    mass = A[s] * kiss_reader.PROTON_MASS
    momentum = np.sqrt(np.maximum(np.asarray(E) ** 2 - mass ** 2, 0.0))
    return momentum / Z[s]


def population_component(theta, pop, s, E):
    """Differential intensity of one species from one population."""
    if s not in POP_SPECIES[pop]:
        return np.zeros_like(np.asarray(E, dtype=float))

    R = rigidity(E, s)
    log10K = theta[K_INDEX[pop][s]]
    group = SLOPE_GROUP[pop][s]
    alpha = theta[ALPHA_INDEX[pop][group]]
    rb = 10.0 ** theta[RB_INDEX[pop]]
    delta_alpha = theta[DELTA_ALPHA_INDEX[pop]]
    smoothing = BREAK_SMOOTHING[pop]

    log_transition = np.logaddexp(
        0.0, np.log(R / rb) / smoothing
    )
    log_transition_r0 = np.logaddexp(
        0.0, np.log(R0 / rb) / smoothing
    )
    log_shape = (
        -alpha * np.log(R / R0)
        - delta_alpha * smoothing
        * (log_transition - log_transition_r0)
    )
    return 10.0 ** log10K * np.exp(log_shape)


def species_components(theta, s, E):
    """Return the LE and HE intensities of species ``s``."""
    return tuple(
        population_component(theta, pop, s, E) for pop in POPULATIONS
    )


def species_model(theta, s, E):
    """Return the total LE+HE intensity of species ``s``."""
    le, he = species_components(theta, s, E)
    return le + he


def energy_scales(theta, exp):
    factors = np.ones(exp.shape, dtype=float)
    for i, name in enumerate(SCALE_EXPERIMENTS):
        factors[exp == name] = theta[SCALE_START + i]
    return factors


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------

def log_prior(theta):
    for value, (low, high) in zip(theta, PRIOR_BOUNDS):
        if not (low < value < high):
            return -np.inf

    if not theta[RB_INDEX["LE"]] < theta[RB_INDEX["HE"]]:
        return -np.inf
    return 0.0


def log_likelihood(theta, E, y, exp, sp, groups):
    scale = energy_scales(theta, exp)
    shifted_energy = scale * E
    prediction = np.empty_like(y)

    for s in SPECIES:
        mask = sp == s
        prediction[mask] = (
            scale[mask]
            * species_model(theta, s, shifted_energy[mask])
        )

    residual = y - prediction
    chi2 = 0.0
    for idx, covariance in groups:
        block_residual = residual[idx]
        chi2 += block_residual @ np.linalg.solve(
            covariance, block_residual
        )
    return -0.5 * chi2


def log_probability(theta, E, y, exp, sp, groups):
    prior = log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    return prior + log_likelihood(theta, E, y, exp, sp, groups)


# ---------------------------------------------------------------------------
# MCMC and output
# ---------------------------------------------------------------------------

def print_parameter_recap():
    if not (
        len(LABELS) == len(PRIOR_BOUNDS) == THETA0.size == NDIM
    ):
        raise RuntimeError("inconsistent parameter configuration")

    print("\nTwo-population parameter recap:")
    print(f"  NDIM = {NDIM}")
    print("  LE species: " + ", ".join(POP_SPECIES["LE"]))
    print("  HE species: " + ", ".join(POP_SPECIES["HE"]))
    print("  HE C/O/Fe share the HE-He rigidity shape; only K is independent.")
    print("  Both populations use SBPL rigidity breaks.")
    print("  Fixed smoothing: w_LE = w_HE = 0.1")
    print("  Delta-alpha priors: uniform in (0.1, 5.0)")
    print("  Rb_LE prior: 10^[3.4, 4.7] GV; initial 10 TV")
    print("  Rb_HE prior: 10^[5.4, 6.6] GV; initial 1 PV")
    print("  DAMPE reference scale; fitted scales: "
          + ", ".join(SCALE_EXPERIMENTS))
    print("\n  idx  parameter                           theta0       prior")
    print("  ---  ----------------------------------  --------  ----------------")
    for i, (label, initial, bounds) in enumerate(
            zip(LABELS, THETA0, PRIOR_BOUNDS)):
        print(
            f"  {i:3d}  {label:34s}  {initial:8.3f}  "
            f"({bounds[0]:g}, {bounds[1]:g})"
        )


def run_mcmc(data, nwalkers=72, nsteps=30000, burnin=6000,
             thin=15, seed=42, ncores=NCORES):
    if nwalkers < 2 * NDIM:
        raise ValueError(
            f"emcee needs at least 2*NDIM={2 * NDIM} walkers"
        )
    if not 0 <= burnin < nsteps:
        raise ValueError("burnin must satisfy 0 <= burnin < nsteps")

    rng = np.random.default_rng(seed)
    position = THETA0 + 1.0e-3 * rng.standard_normal((nwalkers, NDIM))
    groups = build_covariance_groups(data)
    args = (
        data["E"], data["y"], data["exp"], data["sp"], groups,
    )

    with Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, NDIM, log_probability, args=args, pool=pool
        )
        sampler.run_mcmc(position, nsteps, progress=True)

    acceptance = np.mean(sampler.acceptance_fraction)
    print(f"\nmean acceptance fraction: {acceptance:.3f}")

    tau = sampler.get_autocorr_time(quiet=True)
    finite_tau = np.isfinite(tau)
    if np.any(finite_tau):
        slowest = np.nanargmax(tau)
        max_tau = tau[slowest]
        print(
            f"max autocorrelation time: {max_tau:.1f} steps "
            f"({LABELS[slowest]})"
        )
        print(f"chain length / max tau: {nsteps / max_tau:.0f}")
    else:
        print("autocorrelation time could not be estimated")

    return sampler.get_chain(discard=burnin, thin=thin, flat=True)


def print_summary(samples):
    print("\nPosterior (median, 16th/84th percentiles):")
    for i, label in enumerate(LABELS):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(
            f"  {label:28s} = {q50:9.3f} "
            f"(+{q84 - q50:.3f} / -{q50 - q16:.3f})"
        )


def save_output(samples, data, args):
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.savez(
        OUTPUT,
        samples=samples,
        labels=np.asarray(LABELS),
        species=np.asarray(SPECIES),
        populations=np.asarray(POPULATIONS),
        le_species=np.asarray(POP_SPECIES["LE"]),
        he_species=np.asarray(POP_SPECIES["HE"]),
        R0=R0,
        min_rigidity=MIN_RIGIDITY,
        max_energy=MAX_ENERGY,
        model_kind=np.asarray("two_population_sbpl"),
        break_smoothing_le=BREAK_SMOOTHING["LE"],
        break_smoothing_he=BREAK_SMOOTHING["HE"],
        delta_alpha_index_le=DELTA_ALPHA_INDEX["LE"],
        delta_alpha_index_he=DELTA_ALPHA_INDEX["HE"],
        k_index_le=np.asarray([K_INDEX["LE"][s] for s in POP_SPECIES["LE"]]),
        k_index_he=np.asarray([K_INDEX["HE"][s] for s in POP_SPECIES["HE"]]),
        alpha_index_le_h=ALPHA_INDEX["LE"]["H"],
        alpha_index_le_he=ALPHA_INDEX["LE"]["He"],
        alpha_index_le_heavy=ALPHA_INDEX["LE"]["heavy"],
        alpha_index_he_h=ALPHA_INDEX["HE"]["H"],
        alpha_index_he_he=ALPHA_INDEX["HE"]["He"],
        rb_index_le=RB_INDEX["LE"],
        rb_index_he=RB_INDEX["HE"],
        scale_start=SCALE_START,
        scale_experiments=np.asarray(SCALE_EXPERIMENTS),
        fit_dataset_tag=np.asarray(
            "DAMPE:H,He,C,O,Fe;CALET:H,He,C,O,Fe;"
            "LHAASO-QGSJET-II-04:H,He"
        ),
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burnin=args.burnin,
        thin=args.thin,
        seed=args.seed,
        data_E=data["E"],
        data_y=data["y"],
        data_stat_lo=data["stat_lo"],
        data_stat_up=data["stat_up"],
        data_sys_lo=data["sys_lo"],
        data_sys_up=data["sys_up"],
        data_err_lo=data["err_lo"],
        data_err_up=data["err_up"],
        data_exp=data["exp"],
        data_sp=data["sp"],
    )
    print(f"saved {OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the two-population Galactic cosmic-ray model."
    )
    parser.add_argument("--ncores", type=int, default=NCORES)
    parser.add_argument("--nwalkers", type=int, default=72)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=6000)
    parser.add_argument("--thin", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    print_parameter_recap()
    data = load_data()

    print(f"\nloaded {data['E'].size} points:")
    for exp in EXPERIMENTS:
        counts = [
            f"{s}={np.sum((data['exp'] == exp) & (data['sp'] == s))}"
            for s in EXP_SPECIES[exp]
        ]
        print(f"  {exp}: " + ", ".join(counts))

    samples = run_mcmc(
        data,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burnin=args.burnin,
        thin=args.thin,
        seed=args.seed,
        ncores=args.ncores,
    )
    print(f"{samples.shape[0]} samples after burn-in/thinning")
    print_summary(samples)
    save_output(samples, data, args)


if __name__ == "__main__":
    main()
