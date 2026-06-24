"""Corner plot of the direct-spectrum MCMC posterior.

Kept separate from ``plot_direct.py`` because the corner plot needs the default
matplotlib style: the large fonts of ``EVA.mplstyle`` (used for the publication
spectra) make a dense 16-parameter corner unreadable.

Run with:  python plot_corner.py
"""

import os

import numpy as np
import corner
import matplotlib.pyplot as plt

from mcmc_direct import OUTPUT

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def plot_corner(samples, labels):
    fig = corner.corner(
        samples, labels=list(labels), show_titles=True, title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84], max_n_ticks=3,
        label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10},
    )
    n = len(labels)
    fig.set_size_inches(1.6 * n, 1.6 * n)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "EVA_mcmc_direct_corner.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


def main():
    d = np.load(OUTPUT, allow_pickle=False)
    plot_corner(d["samples"], d["labels"])


if __name__ == "__main__":
    main()
