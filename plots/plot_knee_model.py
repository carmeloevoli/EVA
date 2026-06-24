import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data_statonly, plot_data, savefig

def grapes_break(E, Z):
    Rb, dslope = 3e5, 0.29
    R = E / float(Z)
    s = 10.0
    return (1. + (R / Rb) ** s) ** (dslope / s)

def knee1st(E, Z):
    Rb, dslope, norm = 2.6e6, -1.3, 0.9 * 1.15
    R = E / float(Z)
    s = 10.0
    return norm * (1. + (R / Rb) ** s) ** (dslope / s)

XLABEL = r'E [GeV]'
YLABEL = r'E$^{3}$ I [GeV$^{2}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'

def get_galactic_individual(norm, doBreak = True):
    E, I_H, I_H_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,1,2), unpack=True)
    E, I_He, I_He_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,3,4), unpack=True)
    E, I_C, I_C_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,5,6), unpack=True)
    E, I_O, I_O_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,7,8), unpack=True)
    E, I_Mg, I_Mg_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,9,10), unpack=True)
    E, I_Si, I_Si_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,11,12), unpack=True)
    E, I_Fe, I_Fe_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,13,14), unpack=True)

    if doBreak:
        I_H *= grapes_break(E, 1) * knee1st(E, 1)
        I_He *= grapes_break(E, 2) * knee1st(E, 2)
        I_C *= grapes_break(E, 6) * knee1st(E, 6)
        I_O *= grapes_break(E, 8) * knee1st(E, 8)
        I_Mg *= grapes_break(E, 12) * knee1st(E, 12)
        I_Si *= grapes_break(E, 14) * knee1st(E, 14)
        I_Fe *= grapes_break(E, 26) * knee1st(E, 26)

    I_CO = I_C + I_O
    I_CO_err = I_C_err + I_O_err

    I_MgSi = I_Mg + I_Si
    I_MgSi_err = I_Mg_err + I_Si_err

    I_all = I_H + I_He + I_C + I_O + I_Mg + I_Si + I_Fe

    lnA = np.log(4.) * I_He + np.log(12.) * I_C + np.log(16.) * I_O + np.log(24.) * I_Mg + np.log(28.) * I_Si + np.log(56.) * I_Fe
    lnA /= I_all

    return E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA

def plot_knee():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[5e5, 3e8], ylim=[5e5, 6e6], xscale='log', yscale='log')
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data_statonly(ax, 'LHAASO_QGSJET-II-04_all_energy.txt', 3., 1, 'o', 'navy', 'LHAASO', 22)
    plot_data_statonly(ax, 'GAMMA_SIBYLL_all_energy.txt', 3., 1, 'o', 'm', 'GAMMA', 16)
    plot_data_statonly(ax, 'TALE_QGSJET-II-04_all_energy.txt', 3., 0.9, 'o', 'y', 'TALE', 15)
    plot_data_statonly(ax, 'TUNKA-133_QGSJET-01_all_energy.txt', 3., 1, 'o', 'tab:brown', 'TUNKA-133', 14)
    plot_data_statonly(ax, 'KASCADE_QGSJET-01_all_energy.txt', 3., 1.01, 'o', 'tab:gray', 'KASCADE SIBYLL 2005')
    plot_data_statonly(ax, 'TA_all_energy.txt', 3., 0.9, 'o', 'b', 'Telescope Array', 20)
    #plot_data(ax, 'TIBET_SIBYLL+HD_all_energy.txt', 3., 0.95, 'o', 'g', 'Tibet-III', 18)
    plot_data_statonly(ax, 'ICECUBE_SIBYLL_21_all_energy.txt', 3., 1, 'o', 'c', 'IceCube+IceTop', 17)
    
    E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA = get_galactic_individual(1.0)

    ax.plot(E, np.power(E, 0.3) * I_all, color='r', zorder=30)

    ax.plot(E, np.power(E, 0.3) * I_H, color='tab:blue', zorder=30)
    ax.text(1e6, 6e5, 'H', color='tab:blue', fontsize=24)

    ax.plot(E, np.power(E, 0.3) * I_He, color='tab:green', zorder=30)
    ax.text(4e6, 1.6e6, 'He', color='tab:green', fontsize=24)

    ax.plot(E, np.power(E, 0.3) * I_CO, color='tab:orange', zorder=30)
    ax.text(1.5e7, 1.3e6, 'CO', color='tab:orange', fontsize=24)

    ax.plot(E, np.power(E, 0.3) * I_MgSi, color='tab:olive', zorder=30)

    ax.plot(E, np.power(E, 0.3) * I_Fe, color='tab:gray', zorder=30)
    ax.text(6.0e7, 1.5e6, 'Fe', color='tab:gray', fontsize=24)

    #ax.legend(fontsize=12)
    savefig(fig, 'EVA_knee_model.pdf')

def plot_lnA_knee():
    def get_lna_data(ax, exp, color, label):
        filename = '../data/lake/lnA_epos.txt'
        f = open(filename, 'r')
        E = []
        lnA = []
        lnAerr = []
        for line in f:
            s = line.split(" ")
            if s[0] == exp:
                E.append(float(s[1]) / 1e9)
                lnA.append(float(s[2]))
                lnAerr.append(float(s[3]))
        ax.errorbar(E, lnA, yerr=lnAerr, fmt='o', markeredgecolor=color, color=color, capsize=4.5, markersize=7, elinewidth=2.2, capthick=2.2, zorder=1, label=label)

    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=r'$\langle$ ln A $\rangle$', xscale='log', xlim=[5e5, 3e8], ylim=[0, np.log(56)])

    ax.set_yticks([np.log(1), np.log(4), np.log(14), np.log(24), np.log(56)])
    ax.set_yticklabels(['H', 'He', 'N', 'Mg', 'Fe'], fontsize=25)
    ax.tick_params(axis='y', which='minor', length=0, color='k')

    get_lna_data(ax, 'tunka', 'tab:olive', 'Tunka') 
    get_lna_data(ax, 'yakutskSmall', 'tab:pink', 'Yakutsk Small')
    get_lna_data(ax, 'yakutskBig', 'tab:cyan', 'Yakutsk Big')
    get_lna_data(ax, 'casaBlanca', 'tab:brown', 'Casa-Blanca')
    get_lna_data(ax, 'augerXmax', 'r', 'Auger Xmax')
    get_lna_data(ax, 'hiresMIA', 'tab:orange', 'Hi-Res MIA')
    get_lna_data(ax, 'hires', 'tab:gray', 'Hi-Res')
    get_lna_data(ax, 'telArray', 'tab:blue', 'Telescope Array')
    
    E, I_H, I_H_err, I_He, I_He_err, I_CO, I_CO_err, I_MgSi, I_MgSi_err, I_Fe, I_Fe_err, I_all, lnA = get_galactic_individual(1.0)

    plot_data(ax, 'LHAASO_QGSJET-II-04_lnA_energy.txt', 0, 1, 'o', 'navy', 'LHAASO', 22)
    plot_data(ax, 'LHAASO_EPOS-LHC_lnA_energy.txt', 0, 1, 'o', 'tab:blue', 'LHAASO', 22)
    plot_data(ax, 'LHAASO_SIBYLL-23_lnA_energy.txt', 0, 1, 'o', 'tab:cyan', 'LHAASO', 22)

    ax.plot(E, 0.88 * lnA)

    # ax.annotate('1st knee',xy=(4e6, 0.5),xytext=(4e6, 0.2),
    #             arrowprops=dict(color='r'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
    #             ha='center', va='bottom', fontsize=15, color='r')

    # ax.annotate('1st ankle',xy=(20e6, 0.5),xytext=(20e6, 0.2),
    #             arrowprops=dict(color='b'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
    #             ha='center', va='bottom', fontsize=15, color='b')

    # ax.annotate('2nd knee',xy=(100e6, 0.5),xytext=(100e6, 0.2),
    #             arrowprops=dict(color='r'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
    #             ha='center', va='bottom', fontsize=15, color='r')

    # ax.annotate('2nd ankle',xy=(5e9, 0.5),xytext=(5e9, 0.2),
    #             arrowprops=dict(color='b'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
    #             ha='center', va='bottom', fontsize=15, color='b')
    
    # ax.fill_between([1e5, 1e11], np.log(1), np.log(4), color='tab:olive', alpha=0.10)
    # ax.fill_between([1e5, 1e11], np.log(4), np.log(14), color='tab:orange', alpha=0.10)
    # ax.fill_between([1e5, 1e11], np.log(14), np.log(24), color='tab:red', alpha=0.10)
    # ax.fill_between([1e5, 1e11], np.log(24), np.log(56), color='tab:brown', alpha=0.10)

    # x = 8e5
    # y = np.linspace(2.25, 3.65, 9)
    # fontsize = 15
    # ax.text(x, y[0], 'Casa-Blanca', fontsize=fontsize, color='tab:brown')
    # ax.text(x, y[1], 'Hi-Res', fontsize=fontsize, color='tab:gray')
    # ax.text(x, y[2], 'Hi-Res MIA', fontsize=fontsize, color='tab:orange')
    # ax.text(x, y[3], 'LHAASO', fontsize=fontsize, color='navy')
    # ax.text(x, y[4], 'Pierre Auger Xmax', fontsize=fontsize, color='r')
    # ax.text(x, y[5], 'Telescope Array', fontsize=fontsize, color='b')
    # ax.text(x, y[6], 'TUNKA', fontsize=fontsize, color='tab:olive')
    # ax.text(x, y[7], 'Yakutsk Small', fontsize=fontsize, color='tab:pink')
    # ax.text(x, y[8], 'Yakutsk Big', fontsize=fontsize, color='tab:cyan')

    savefig(fig, 'EVA_lnA_knee.pdf')
    
if __name__== "__main__":
    plot_knee()
    plot_lnA_knee()