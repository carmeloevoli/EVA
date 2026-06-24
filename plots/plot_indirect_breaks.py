import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import plot_data, set_axes, savefig
from plot_galactic import XLABEL

YLABEL = r'E$^{3}$ I [GeV$^{2}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'

# LHAASO_EPOS-LHC_all_energy.txt
# LHAASO_QGSJET-II-04_all_energy.txt
# LHAASO_SIBYLL-23_all_energy.txt
# TA_all_energy.txt

# TIBET_QGSJET+HD_all_energy.txt
# TIBET_QGSJET+PD_all_energy.txt
# TIBET_SIBYLL+HD_all_energy.txt

def plot_GAMMA():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e6, 1e9], ylim=[1e6, 6e6], xscale='log', yscale='linear')

    plot_data(ax, 'GAMMA_SIBYLL_all_energy.txt', 3.0, 1, 'o', 'tab:blue', '', zorder=3)

    x, y, yerr = np.loadtxt('output/GAMMA_all_breaks.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    savefig(fig, f'EVA_indirect_breaks_GAMMA.pdf')

   
def plot_TALE():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e6, 1e9], ylim=[5e5, 5e6], xscale='log', yscale='linear')

    plot_data(ax, 'TALE_QGSJET-II-04_all_energy.txt', 3.0, 1, 'o', 'tab:blue', '', zorder=3)

    x, y, yerr = np.loadtxt('output/TALE_all_breaks.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    savefig(fig, f'EVA_indirect_breaks_TALE.pdf')


def plot_TUNKA():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e6, 1e9], ylim=[1e6, 6e6], xscale='log', yscale='linear')

    plot_data(ax, 'TUNKA-133_QGSJET-01_all_energy.txt', 3.0, 1, 'o', 'tab:blue', '', zorder=3)

    x, y, yerr = np.loadtxt('output/TUNKA_all_breaks.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    savefig(fig, f'EVA_indirect_breaks_TUNKA.pdf')


def plot_ICECUBE():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e6, 2e9], ylim=[5e5, 5e6], xscale='log', yscale='linear')

    plot_data(ax, 'ICECUBE_SIBYLL_21_all_energy.txt', 3.0, 1, 'o', 'tab:blue', '', zorder=3)

    x, y, yerr = np.loadtxt('output/ICECUBE_all_breaks.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    savefig(fig, f'EVA_indirect_breaks_ICECUBE.pdf')

def plot_KASCADE():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e6, 2e8], ylim=[9e5, 8e6], xscale='log', yscale='linear')

    plot_data(ax, 'KASCADE_QGSJET-01_all_energy.txt', 3.0, 1, 'o', 'tab:blue', '', zorder=3)

    x, y, yerr = np.loadtxt('output/KASCADE_QGSJET_all_breaks.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.12, zorder=1)

    savefig(fig, f'EVA_indirect_breaks_KASCADE.pdf')


def plot_GRANDE():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e6, 3e9], ylim=[9e5, 1e7], xscale='log', yscale='log')

    plot_data(ax, 'KGRANDE_QGSJET-II-04_all_energy.txt', 3.0, 1, 'o', 'tab:blue', '', zorder=3)

    x, y, yerr = np.loadtxt('output/KGRANDE_all_breaks.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.12, zorder=1)

    savefig(fig, f'EVA_indirect_breaks_GRANDE.pdf')


def plot_TA():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[3e6, 1e9], ylim=[1e6, 4e6], xscale='log', yscale='linear')

    plot_data(ax, 'TA_all_energy.txt', 3.0, 1, 'o', 'tab:blue', '', zorder=3)

    x, y, yerr = np.loadtxt('output/TA_all_breaks.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.12, zorder=1)

    savefig(fig, f'EVA_indirect_breaks_TA.pdf')

# def plot_AUGER():
#     fig, ax = plt.subplots(figsize=(11.50, 8.5))
#     set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e6, 1e10], ylim=[5e5, 1e7], xscale='log', yscale='log')

#     plot_data(ax, 'AUGER_all_energy.txt', 3.0, 1, 'o', 'tab:blue', '', zorder=3)

# #    x, y, yerr = np.loadtxt('output/AUGER_all_energy.txt', usecols=(0,1,2), unpack=True)

# #    ax.plot(x, y, color='tab:red', zorder=10)
# #    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

#     savefig(fig, f'EVA_indirect_breaks_AUGER.pdf')


if __name__== "__main__":
    #plot_TALE()
    #plot_GAMMA()
    #plot_TUNKA()
    #plot_ICECUBE()
    #plot_GRANDE()
    #plot_KASCADE()
    #plot_TA()
