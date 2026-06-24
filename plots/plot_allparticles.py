import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, plot_data_statonly, savefig
from plot_galactic_GRAPES import grapes_break

XLABEL = r'E [GeV]'
YLABEL = r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'

def get_galactic_individual(norm, doBreak = True):
    E, I_H, I_H_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,1,2), unpack=True)
    E, I_He, I_He_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,3,4), unpack=True)
    E, I_C, I_C_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,5,6), unpack=True)
    E, I_O, I_O_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,7,8), unpack=True)
    E, I_Mg, I_Mg_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,9,10), unpack=True)
    E, I_Si, I_Si_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,11,12), unpack=True)
    E, I_Fe, I_Fe_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,13,14), unpack=True)

    I_H *= norm
    I_He *= norm
    I_C *= norm
    I_O *= norm
    I_Mg *= norm
    I_Si *= norm
    I_Fe *= norm

    if doBreak:
        I_H *= grapes_break(E, 1) 
        I_He *= grapes_break(E, 2)
        I_C *= grapes_break(E, 6)
        I_O *= grapes_break(E, 8)
        I_Mg *= grapes_break(E, 12)
        I_Si *= grapes_break(E, 14)
        I_Fe *= grapes_break(E, 26)

    I_CO = I_C + I_O
    I_CO_err = I_C_err + I_O_err

    I_MgSi = I_Mg + I_Si
    I_MgSi_err = I_Mg_err + I_Si_err

    I_all = I_H + I_He + I_C + I_O + I_Mg + I_Si + I_Fe

    lnA = np.log(4.) * I_He + np.log(12.) * I_C + np.log(16.) * I_O + np.log(24.) * I_Mg + np.log(28.) * I_Si + np.log(56.) * I_Fe
    lnA /= I_all

    return E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA

def plot_galactic_all():
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e3, 2e6], ylim=[3e4, 6.4e4], xscale='log', yscale='linear')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    plot_data(ax, 'DAMPE_all_energy.txt', 2.7, 1, 'o', 'r', 'DAMPE (PRELIMINARY)', 9)
    #plot_data_statonly(ax, 'GAMMA_SIBYLL_all_energy.txt', 2.7, 1, 'o', 'tab:orange', 'GAMMA', 4)
    plot_data_statonly(ax, 'HAWC_QGSJET-II-04_all_energy.txt', 2.7, 1, 'o', 'tab:brown', 'HAWC', 5)
    plot_data_statonly(ax, 'NUCLEON_all_energy.txt', 2.7, 1, 'o', 'tab:pink', 'NUCLEON', 6)

    plot_data(ax, 'LHAASO_QGSJET-II-04_all_energy.txt', 2.7, 1, 'o', 'c', 'LHAASO', 8)

    norm = 1.27
    E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA = get_galactic_individual(norm)
    ax.plot(E, I_all, color='b', zorder=10)

    E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA = get_galactic_individual(norm, False)
    ax.plot(E, I_all, color='b', ls='--', zorder=10)
    
    ax.legend(fontsize=16)

    savefig(fig, 'EVA_all_galactic.pdf')

def plot_galactic_individual():
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[3e3, 3e6], ylim=[3e3, 6.4e4], xscale='log', yscale='log')
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    plot_data(ax, 'DAMPE_all_energy.txt', 2.7, 1, 'o', 'r', 'DAMPE (PRELIMINARY)', 9)
    plot_data_statonly(ax, 'GAMMA_SIBYLL_all_energy.txt', 2.7, 1, 'o', 'tab:orange', 'GAMMA', 4)
    plot_data_statonly(ax, 'HAWC_QGSJET-II-04_all_energy.txt', 2.7, 1, 'o', 'tab:brown', 'HAWC', 5)
    plot_data_statonly(ax, 'NUCLEON_all_energy.txt', 2.7, 1, 'o', 'tab:pink', 'NUCLEON', 6)

    plot_data(ax, 'LHAASO_QGSJET-II-04_all_energy.txt', 2.7, 1, 'o', 'c', 'LHAASO', 8)

    norm = 1.27
    E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA = get_galactic_individual(norm)
    
    ax.plot(E, I_H, ls='--', color='tab:blue', zorder=10)
    ax.fill_between(E, (I_H - I_H_err), (I_H + I_H_err), color='tab:gray', alpha=0.15, zorder=0)

    ax.plot(E, I_He, ls='--', color='tab:green', zorder=10)
    ax.fill_between(E, (I_He - I_He_err), (I_He + I_He_err), color='tab:gray', alpha=0.15, zorder=0)

    ax.plot(E, I_CO, ls='--', color='tab:orange', zorder=10)
    ax.fill_between(E, (I_CO - I_CO_err), (I_CO + I_CO_err), color='tab:gray', alpha=0.15, zorder=0)

    ax.plot(E, I_MgSi, ls='--', color='tab:olive', zorder=10)
    ax.fill_between(E, (I_MgSi - I_MgSi_err), (I_MgSi + I_MgSi_err), color='tab:gray', alpha=0.15, zorder=0)

    ax.plot(E, I_Fe, ls='--', color='tab:red', zorder=10)
    ax.fill_between(E, (I_Fe - I_Fe_err), (I_Fe + I_Fe_err), color='tab:gray', alpha=0.15, zorder=0)

    ax.vlines(1.3e4, 1e1, 1e6, ls=':', lw=1, color='tab:blue')
    ax.vlines(1.3e4 * 26, 1e1, 1e6, ls=':', lw=1, color='tab:red')
    #ax.legend(fontsize=16)

    savefig(fig, 'EVA_individual_galactic.pdf')

def plot_galactic_lnA():
    def add_lnA_Irene(ax, filename, color='tab:gray'):
        E, y, y_lo, y_up = np.loadtxt(filename, usecols=(0,1,2,3), unpack=True)
        ax.errorbar(E, y, yerr=[y_lo, y_up], fmt='o', markeredgecolor=color, color=color, 
            capsize=4.0, markersize=5.8, elinewidth=1.8, capthick=1.8, zorder=1)
    
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=r'$\langle$ ln A $\rangle$', xlim =[3e3, 3e6], ylim=[1.0, 2.4], xscale='log', yscale='linear')
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    plot_data(ax, 'LHAASO_QGSJET-II-04_lnA_energy.txt', 0., 1, 'o', 'r', 'LHAASO', 7)
    plot_data(ax, 'LHAASO_EPOS-LHC_lnA_energy.txt', 0., 1, 'o', 'tab:orange', 'LHAASO', 8)
    plot_data(ax, 'LHAASO_SIBYLL-23_lnA_energy.txt', 0., 1, 'o', 'tab:purple', 'LHAASO', 9)

    ax.text(4e5, 1.1, 'LHAASO', color='r')
    
    #add_lnA_Irene(ax, '../data/lake/Atic02_lnA_DataThief.txt')
    add_lnA_Irene(ax, '../data/lake/JACEE_lnA_DataThief.txt')
    #add_lnA_Irene(ax, '../data/lake/NUCLEON-IC_lnA_DataThief.txt')
    add_lnA_Irene(ax, '../data/lake/NUCLEON-KLEM_lnA_DataThief.txt')
    #add_lnA_Irene(ax, '../data/lake/RUNJOB_lnA_DataThief.txt')
    #add_lnA_Irene(ax, '../data/lake/SOKOL_lnA_DataThief.txt')

    # E, lnA, lnA_err = np.loadtxt('output/galactic_lnA_fit.txt', usecols=(0,1,2), unpack=True)
    # ax.plot(E, lnA, color='tab:blue')
    # ax.fill_between(E, lnA - lnA_err, lnA + lnA_err, color='tab:gray', alpha=0.18)

    norm = 1.
    E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA = get_galactic_individual(norm)
    ax.plot(E, lnA, color='tab:blue')

    E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA = get_galactic_individual(norm, False)
    ax.plot(E, lnA, ls='--', color='tab:blue')

    ax.hlines(np.log(4), 1e2, 1e7, ls='--', color='tab:green')
    ax.text(1.1e4, 1.25, 'He', color='tab:green', fontsize=24)

    savefig(fig, 'EVA_lnA_galactic.pdf')

if __name__== "__main__":
    plot_galactic_all()
    plot_galactic_lnA()
    plot_galactic_individual()