import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, savefig
from plot_galactic import fCREAM, fCALET, fDAMPE, XLABEL, YLABEL

def grapes_break(E, Z):
    Rb, dslope = 3e5, 0.29
    R = E / float(Z)
    s = 5.0
    return (1. + (R / Rb) ** s) ** (dslope / s)

def plot_GRAPES():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e3, 3e6], ylim=[4e3, 18e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    #plot_data(ax, 'DAMPE_light_energy.txt', 2.7, fDAMPE, 'o', 'tab:red', 'DAMPE', 5)
    #plot_data(ax, 'CREAM_light_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 3)

    plot_data(ax, 'GRAPES_H_energy.txt', 2.7, 1., 'o', 'r', 'GRAPES', 10)

    E, I_H, I_H_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,1,2), unpack=True)
    I_H *= grapes_break(E, 1.)

    ax.plot(E, I_H, color='y', zorder=11)
    ax.fill_between(E, I_H - I_H_err, I_H + I_H_err, color='y', alpha=0.1, zorder=9)
    
    ax.legend(fontsize=25)

    plot_data(ax, 'AMS-02_H_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'CALET_H_energy.txt', 2.7, fCALET, 'o', 'tab:gray', 'CALET', 4)
    plot_data(ax, 'DAMPE_H_energy.txt', 2.7, fDAMPE, 'o', 'tab:gray', 'DAMPE', 5)
    #plot_data(ax, 'CREAM_H_energy.txt', 2.7, fCREAM, 'o', 'tab:gray', 'CREAM', 2)
    #plot_data(ax, 'ISS-CREAM_H_energy.txt', 2.7, 1., 'o', 'tab:gray', 'ISS-CREAM', 3)

    savefig(fig, 'EVA_GRAPES_H.pdf')

def plot_DAMPE():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e3, 3e6], ylim=[10e3, 40e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'DAMPE_light_energy.txt', 2.7, fDAMPE, 'o', 'tab:red', 'DAMPE', 5)
    plot_data(ax, 'CREAM_light_energy.txt', 2.7, fCREAM, 'o', 'tab:orange', 'CREAM', 3)

    E, I_H, I_H_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,1,2), unpack=True)
    I_H *= grapes_break(E, 1.)

    E, I_He, I_He_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,3,4), unpack=True)
    I_He *= grapes_break(E, 2.)

    I = I_H + I_He
    I_err = I_H_err + I_He_err

    ax.plot(E, I, color='y', zorder=11)
    ax.fill_between(E, I - I_err, I + I_err, color='y', alpha=0.1, zorder=9)
    
    ax.legend(fontsize=25)

    savefig(fig, 'EVA_DAMPE_light.pdf')


if __name__== "__main__":
    plot_GRAPES()
    plot_DAMPE()