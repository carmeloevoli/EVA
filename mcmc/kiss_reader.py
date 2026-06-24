"""Readers for the KISS cosmic-ray data tables.

The tables in ``kiss_tables/`` store the differential intensity as measured by
each experiment, expressed as a function of the *original* energy variable used
in the publication.  Depending on the dataset this can be:

    * ``totalEnergy``             -- total energy of the particle  [GeV]
    * ``kineticEnergy``           -- total kinetic energy          [GeV]
    * ``kineticEnergyPerNucleon`` -- kinetic energy per nucleon    [GeV/n]
    * ``rigidity``                -- rigidity R = pc / Ze          [GV]

The functions below read those files and return the spectrum as a function of
the *total energy* of the particle, applying the appropriate Jacobian so that
the intensity is always dN/dE_tot in [(GeV m^2 s sr)^-1].

File format (whitespace separated, ``#`` comment header):
    x, y, y_stat_err_low, y_stat_err_high, y_sys_err_low, y_sys_err_high
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants and species properties
# ---------------------------------------------------------------------------

PROTON_MASS = 0.938272  # GeV, mass per nucleon used for the rest-mass energy

# Mass number (A) and charge number (Z) of the supported species.  The rest
# mass of the nucleus is approximated as A * PROTON_MASS, which is accurate to
# better than ~1% and well below the experimental uncertainties.
SPECIES = {
    "H":  {"Z": 1,  "A": 1},
    "He": {"Z": 2,  "A": 4},
    "C":  {"Z": 6,  "A": 12},
    "N":  {"Z": 7,  "A": 14},
    "O":  {"Z": 8,  "A": 16},
    "Ne": {"Z": 10, "A": 20},
    "Mg": {"Z": 12, "A": 24},
    "Si": {"Z": 14, "A": 28},
    "Fe": {"Z": 26, "A": 56},
}

KISS_TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiss_tables")

# Energy variables that can appear as the trailing token of a table filename.
X_QUANTITIES = ("totalEnergy", "kineticEnergyPerNucleon", "kineticEnergy", "rigidity")


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def read_header(path):
    """Return the ``#Key: value`` header of a KISS table as a dictionary."""
    header = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if ":" in line:
                key, value = line[1:].split(":", 1)
                header[key.strip()] = value.strip()
    return header


# ---------------------------------------------------------------------------
# Energy-variable conversions to total energy
# ---------------------------------------------------------------------------

def _to_total_energy(x, y, x_quantity, Z, A):
    """Convert (x, y) from the original energy variable to total energy.

    Returns ``(E_tot, jacobian)`` where ``E_tot`` is the total energy [GeV] and
    ``jacobian = dx/dE_tot`` is the factor by which the differential intensity
    must be multiplied: ``dN/dE_tot = (dN/dx) * dx/dE_tot``.
    """
    mass = A * PROTON_MASS  # rest-mass energy of the nucleus [GeV]

    if x_quantity == "totalEnergy":
        E = x
        jacobian = np.ones_like(x)
    elif x_quantity == "kineticEnergy":
        # x is the total kinetic energy: E_tot = E_kin + m, dE_tot = dE_kin
        E = x + mass
        jacobian = np.ones_like(x)
    elif x_quantity == "kineticEnergyPerNucleon":
        # x = E_kin / A  ->  E_tot = A * (x + m_nucleon), dx/dE_tot = 1 / A
        E = A * (x + PROTON_MASS)
        jacobian = np.full_like(x, 1.0 / A)
    elif x_quantity == "rigidity":
        # x = R [GV], pc = Z * R, E_tot = sqrt((Z R)^2 + m^2)
        # dE/dR = Z^2 R / E  ->  dR/dE = E / (Z^2 R)
        E = np.sqrt((Z * x) ** 2 + mass ** 2)
        jacobian = E / (Z ** 2 * x)
    else:
        raise ValueError(f"unsupported X quantity '{x_quantity}'")

    return E, jacobian


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def load_spectrum(filename, min_energy=0.0, max_energy=np.inf):
    """Load a KISS table and return the spectrum as a function of total energy.

    Parameters
    ----------
    filename : str
        Name of the table inside ``kiss_tables/`` (e.g. ``DAMPE_H_kineticEnergy.txt``),
        or an absolute/relative path to a table file.
    min_energy, max_energy : float
        Bounds, in *total energy* [GeV], used to select the data points to
        return.

    Returns
    -------
    E : ndarray
        Total energy of the particle [GeV].
    y : ndarray
        Differential intensity dN/dE_tot [(GeV m^2 s sr)^-1].
    stat_lo, stat_up, sys_lo, sys_up : ndarray
        Lower and upper statistical and systematic uncertainties on ``y``.
    """
    path = filename if os.path.isfile(filename) else os.path.join(KISS_TABLES_DIR, filename)

    header = read_header(path)
    x_quantity = header.get("X Quantity")
    species = header.get("Y Quantity")
    if species not in SPECIES:
        raise ValueError(
            f"species '{species}' from {filename} is not in the SPECIES table"
        )
    Z, A = SPECIES[species]["Z"], SPECIES[species]["A"]

    x, y, stat_lo, stat_up, sys_lo, sys_up = np.loadtxt(
        path, usecols=(0, 1, 2, 3, 4, 5), unpack=True
    )

    E, jacobian = _to_total_energy(x, y, x_quantity, Z, A)
    y = y * jacobian
    stat_lo, stat_up = stat_lo * jacobian, stat_up * jacobian
    sys_lo, sys_up = sys_lo * jacobian, sys_up * jacobian

    mask = (E >= min_energy) & (E <= max_energy)
    order = np.argsort(E[mask])  # rigidity conversion can reorder points
    s = lambda a: a[mask][order]
    return s(E), s(y), s(stat_lo), s(stat_up), s(sys_lo), s(sys_up)


def load_experiment(experiment, species, min_energy=0.0, max_energy=np.inf):
    """Load a spectrum by experiment and species, resolving the table filename.

    Searches ``kiss_tables/`` for a file ``<experiment>_<species>_<xquantity>.txt``
    and returns the spectrum in total energy, as :func:`load_spectrum` does.
    """
    # Accept only single-species spectra of the form
    # "<experiment>_<species>_<xquantity>.txt"; this rejects ratio tables such
    # as "CALET_H_He_rigidity.txt".
    expected = {f"{experiment}_{species}_{q}.txt" for q in X_QUANTITIES}
    matches = [f for f in os.listdir(KISS_TABLES_DIR) if f in expected]
    if not matches:
        raise FileNotFoundError(
            f"no table found for experiment '{experiment}' and species '{species}'"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous tables for '{experiment}' '{species}': {sorted(matches)}"
        )
    return load_spectrum(matches[0], min_energy, max_energy)
