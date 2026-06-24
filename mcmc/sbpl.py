"""Smoothed broken power-law (SBPL) model for the cosmic-ray intensity.

For ``n`` breaks the intensity reads

    phi(E) = K (E/E0)^(-alpha[0])
             * prod_{i=1}^{n} [1 + (E/Eb[i-1])^(1/w[i-1])]^(-(alpha[i] - alpha[i-1]) w[i-1])

i.e. the spectrum is a power law with slope ``alpha[0]`` below the first break
and steepens/hardens to ``alpha[i]`` above the i-th break ``Eb[i-1]``; each
``w`` controls the smoothness of the corresponding transition (small ``w`` ->
sharp break).  ``E0`` is the pivot energy at which ``K`` is defined.

With ``n`` breaks there are ``n + 1`` slopes, ``n`` break energies and ``n``
smoothness parameters.
"""

import numpy as np


def sbpl(E, K, alphas, breaks, ws, E0=1.0):
    """Smoothed broken power-law intensity with an arbitrary number of breaks.

    Parameters
    ----------
    E : array_like
        Energy [GeV].
    K : float
        Normalization (intensity at ``E = E0`` in the ``alphas[0]`` regime).
    alphas : sequence of float
        Spectral indices, ordered from low to high energy. Length ``n + 1``
        for ``n`` breaks.
    breaks : float or sequence of float
        Break energies ``Eb`` [GeV]. Length ``n``.
    ws : float or sequence of float
        Smoothness of each break (smaller is sharper). A scalar is broadcast to
        all breaks; otherwise length ``n``.
    E0 : float, optional
        Pivot energy [GeV] for the normalization (default 1).

    Returns
    -------
    ndarray
        The intensity ``phi(E)``, same shape as ``E``.
    """
    E = np.asarray(E, dtype=float)
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    breaks = np.atleast_1d(np.asarray(breaks, dtype=float))

    n = breaks.size
    if alphas.size != n + 1:
        raise ValueError(
            f"expected {n + 1} slopes for {n} breaks, got {alphas.size}"
        )
    ws = np.broadcast_to(np.asarray(ws, dtype=float), (n,))

    log_phi = np.log(K) - alphas[0] * np.log(E / E0)

    # Each break multiplies a smoothed transition factor; the log of the bracket
    # is evaluated with logaddexp for numerical stability:
    # log[1 + (E/Eb)^(1/w)] = logaddexp(0, (1/w) * log(E/Eb))
    for i in range(n):
        log_bracket = np.logaddexp(0.0, np.log(E / breaks[i]) / ws[i])
        log_phi = log_phi - (alphas[i + 1] - alphas[i]) * ws[i] * log_bracket

    return np.exp(log_phi)
