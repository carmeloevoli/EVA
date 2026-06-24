import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, savefig

def grapes_break(E, Z):
    Rb, dslope = 3e5, 0.29
    R = E / float(Z)
    s = 5.0
    return (1. + (R / Rb) ** s) ** (dslope / s)

XLABEL = r'E [GeV]'
YLABEL = r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'

fCREAM = 0.925
fCALET = 0.937
fDAMPE = 1.004

def plot_exclusion_region(ax, Z: float = 1.):
    ax.fill_between([1e0 * Z, 5e2 * Z], 0, 1e5, color='tab:gray', alpha=0.25, zorder=0)
    ax.fill_between([2e5 * Z, 1e6 * Z], 0, 1e5, color='tab:gray', alpha=0.25, zorder=0)

def plot_direct_H():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e2, 1e6], ylim=[1e3, 22e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_H_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CALET_H_energy.txt', 2.7, fCALET, 'o', 'tab:blue', 'CALET', 4)
    plot_data(ax, 'DAMPE_H_energy.txt', 2.7, fDAMPE, 'o', 'tab:red', 'DAMPE', 5)
    plot_data(ax, 'CREAM_H_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 2)
    plot_data(ax, 'ISS-CREAM_H_energy.txt', 2.7, 1., 'o', 'tab:purple', 'ISS-CREAM', 3)

    plot_exclusion_region(ax)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,1,2), unpack=True)
    ax.plot(E, I, color='y', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:blue', alpha=0.3, zorder=10)

    ax.legend(fontsize=16)
    savefig(fig, 'EVA_direct_H.pdf')

def plot_direct_He():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e2, 1e6], ylim=[1e3, 22e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_He_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CALET_He_energy.txt', 2.7, fCALET, 'o', 'tab:blue', 'CALET', 4)
    plot_data(ax, 'DAMPE_He_energy.txt', 2.7, fDAMPE, 'o', 'tab:red', 'DAMPE', 5)
    plot_data(ax, 'CREAM_He_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 2)

    plot_exclusion_region(ax, Z=2)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,3,4), unpack=True)
    ax.plot(E, I, color='y', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:blue', alpha=0.3, zorder=10)

    ax.legend(fontsize=16)
    savefig(fig, 'EVA_direct_He.pdf')

def plot_direct_C():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e2, 1e6], ylim=[0, 7e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_C_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CALET_C_energy.txt', 2.7, fCALET, 'o', 'tab:blue', 'CALET', 4)
    plot_data(ax, 'CREAM_C_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 2)

    plot_exclusion_region(ax, Z=6)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,5,6), unpack=True)
    ax.plot(E, I, color='y', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:blue', alpha=0.3, zorder=10)

    ax.legend(fontsize=16)
    savefig(fig, 'EVA_direct_C.pdf')

def plot_direct_O():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e2, 1e6], ylim=[0, 7e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_O_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CALET_O_energy.txt', 2.7, fCALET, 'o', 'tab:blue', 'CALET', 4)
    plot_data(ax, 'CREAM_O_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 2)

    plot_exclusion_region(ax, Z=8)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,7,8), unpack=True)
    ax.plot(E, I, color='y', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:blue', alpha=0.3, zorder=10)

    ax.legend(fontsize=16)
    savefig(fig, 'EVA_direct_O.pdf')

def plot_direct_Mg():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e2, 1e6], ylim=[0, 7e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_Mg_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CREAM_Mg_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 2)

    plot_exclusion_region(ax, Z=12)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,9,10), unpack=True)
    ax.plot(E, I, color='y', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:blue', alpha=0.3, zorder=10)

    ax.legend(fontsize=16)
    savefig(fig, 'EVA_direct_Mg.pdf')

def plot_direct_Si():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e2, 1e6], ylim=[0, 7e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_Si_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CREAM_Si_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 2)

    plot_exclusion_region(ax, Z=14)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,11,12), unpack=True)
    ax.plot(E, I, color='y', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:blue', alpha=0.3, zorder=10)

    ax.legend(fontsize=16)
    savefig(fig, 'EVA_direct_Si.pdf')

def plot_direct_Fe():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e2, 1e6], ylim=[0, 7e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_Fe_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CALET_Fe_energy.txt', 2.7, fCALET, 'o', 'tab:blue', 'CALET', 4)
    plot_data(ax, 'CREAM_Fe_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 2)

    plot_exclusion_region(ax, Z=26)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,13,14), unpack=True)
    ax.plot(E, I, color='y', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:blue', alpha=0.3, zorder=10)

    ax.legend(fontsize=16)
    savefig(fig, 'EVA_direct_Fe.pdf')

def plot_direct_recap():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[3e3, 3e6], ylim=[4e2, 4e4], xscale='log', yscale='log')
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_H_energy.txt', 2.7, 1., 'o', 'tab:blue', 'AMS-02', 1)
    plot_data(ax, 'CALET_H_energy.txt', 2.7, fCALET, 'o', 'tab:blue', 'CALET', 4)
    plot_data(ax, 'DAMPE_H_energy.txt', 2.7, fDAMPE, 'o', 'tab:blue', 'DAMPE', 5)
    plot_data(ax, 'CREAM_H_energy.txt', 2.7, fCREAM, 'o', 'tab:blue', 'CREAM', 2)
    plot_data(ax, 'ISS-CREAM_H_energy.txt', 2.7, 1., 'o', 'tab:blue', 'ISS-CREAM', 3)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,1,2), unpack=True)
    I *= grapes_break(E, 1.)
    y = I[5000]
    ax.text(1.1e6, y, 'H', color='tab:blue', fontsize=18)
    ax.plot(E, I, color='tab:blue', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:blue', alpha=0.15, zorder=10)


    plot_data(ax, 'AMS-02_He_energy.txt', 2.7, 1., 'o', 'tab:green', 'AMS-02', 1)
    plot_data(ax, 'CALET_He_energy.txt', 2.7, fCALET, 'o', 'tab:green', 'CALET', 4)
    plot_data(ax, 'DAMPE_He_energy.txt', 2.7, fDAMPE, 'o', 'tab:green', 'DAMPE', 5)
    plot_data(ax, 'CREAM_He_energy.txt', 2.7, fCREAM, 'o', 'tab:green', 'CREAM', 2)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,3,4), unpack=True)
    I *= grapes_break(E, 2.)
    y = I[5000]
    ax.text(1.1e6, y, 'He', color='tab:green', fontsize=18)
    ax.plot(E, I, color='tab:green', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:green', alpha=0.15, zorder=10)


    plot_data(ax, 'AMS-02_C_energy.txt', 2.7, 1., 'o', 'tab:red', 'AMS-02', 1)
    plot_data(ax, 'CALET_C_energy.txt', 2.7, fCALET, 'o', 'tab:red', 'CALET', 4)
    plot_data(ax, 'CREAM_C_energy.txt', 2.7, fCREAM, 'o', 'tab:red', 'CREAM', 2)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,5,6), unpack=True)
    I *= grapes_break(E, 6.)
    y = I[5000]
    ax.text(1.1e6, y, 'C', color='tab:red', fontsize=18)
    ax.plot(E, I, color='tab:red', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:red', alpha=0.15, zorder=10)

    
    plot_data(ax, 'AMS-02_O_energy.txt', 2.7, 1., 'o', 'tab:orange', 'AMS-02', 1)
    plot_data(ax, 'CALET_O_energy.txt', 2.7, fCALET, 'o', 'tab:orange', 'CALET', 4)
    plot_data(ax, 'CREAM_O_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 2)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,7,8), unpack=True)
    I *= grapes_break(E, 8.)
    y = I[5000]
    ax.text(1.1e6, y, 'O', color='tab:orange', fontsize=18)
    ax.plot(E, I, color='tab:orange', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:orange', alpha=0.15, zorder=10)


    plot_data(ax, 'AMS-02_Mg_energy.txt', 2.7, 1., 'o', 'tab:purple', 'AMS-02', 1)
    plot_data(ax, 'CREAM_Mg_energy.txt', 2.7, fCREAM, 'o', 'tab:purple', 'CREAM', 2)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,9,10), unpack=True)
    y = I[5000]
    I *= grapes_break(E, 12.)
    ax.text(1.1e6, y, 'Mg', color='tab:purple', fontsize=18)
    ax.plot(E, I, color='tab:purple', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:purple', alpha=0.3, zorder=10)

    plot_data(ax, 'AMS-02_Si_energy.txt', 2.7, 1., 'o', 'tab:olive', 'AMS-02', 1)
    plot_data(ax, 'CREAM_Si_energy.txt', 2.7, fCREAM, 'o', 'tab:olive', 'CREAM', 2)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,11,12), unpack=True)
    I *= grapes_break(E, 14.)
    y = I[5000]
    ax.text(1.1e6, y, 'Si', color='tab:olive', fontsize=18)
    ax.plot(E, I, color='tab:olive', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:olive', alpha=0.3, zorder=10)

    plot_data(ax, 'AMS-02_Fe_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CALET_Fe_energy.txt', 2.7, fCALET, 'o', 'tab:gray', 'CALET', 4)
    plot_data(ax, 'CREAM_Fe_energy.txt', 2.7, fCREAM, 'o', 'tab:gray', 'CREAM', 2)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,13,14), unpack=True)
    I *= grapes_break(E, 26.)
    y = I[5000]
    ax.text(1.1e6, y, 'Fe', color='tab:gray', fontsize=18)
    ax.plot(E, I, color='tab:gray', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='tab:gray', alpha=0.3, zorder=10)

#   ax.legend(fontsize=16)
    savefig(fig, 'EVA_direct_recap.pdf')


if __name__== "__main__":
    # plot_direct_H()
    # plot_direct_He()
    # plot_direct_C()
    # plot_direct_O()
    # plot_direct_Mg()
    # plot_direct_Si()
    # plot_direct_Fe()
    plot_direct_recap()