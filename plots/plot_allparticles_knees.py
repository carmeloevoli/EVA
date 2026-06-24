import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data_statonly, plot_data, savefig

XLABEL = r'E [GeV]'
YLABEL = r'E$^{3}$ I [GeV$^{2}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'

def plot_allparticles_dataonly():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[5e5, 2e10], ylim=[8e5, 9e6], xscale='log', yscale='log')
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data_statonly(ax, 'AUGER_all_energy.txt', 3., 1, 'o', 'r', 'Pierre Auger Obs', 19)
    plot_data_statonly(ax, 'TA_all_energy.txt', 3., 1, 'o', 'b', 'Telescope Array', 20)
    plot_data_statonly(ax, 'TALE_QGSJET-II-04_all_energy.txt', 3., 1, 'o', 'y', 'TALE', 15)
    plot_data_statonly(ax, 'TUNKA-133_QGSJET-01_all_energy.txt', 3., 1, 'o', 'tab:brown', 'TUNKA-133', 14)
    plot_data_statonly(ax, 'TIBET_SIBYLL+HD_all_energy.txt', 3., 1, 'o', 'g', 'Tibet-III', 18)
    plot_data_statonly(ax, 'ICECUBE_SIBYLL_21_all_energy.txt', 3., 1, 'o', 'c', 'IceCube+IceTop', 17)
    plot_data_statonly(ax, 'ICETOP_QGSJET-II-04_all_energy.txt', 3., 1., 'o', 'tab:orange', 'IceTop', 12)
    plot_data_statonly(ax, 'KASCADE_QGSJET-01_all_energy.txt', 3., 1, 'o', 'tab:gray', 'KASCADE SIBYLL 2005')
    plot_data_statonly(ax, 'GAMMA_SIBYLL_all_energy.txt', 3., 1, 'o', 'm', 'GAMMA', 16)
    plot_data_statonly(ax, 'LHAASO_QGSJET-II-04_all_energy.txt', 3., 1, 'o', 'navy', 'LHAASO', 22)

    # plot_data_statonly(ax, 'TIBET_QGSJET+HD_all_energy.txt', 3., 1, 'o', 'g', 'Tibet-III QGSJET 2008')
    # plot_data_statonly(ax, 'TIBET_QGSJET+PD_all_energy.txt', 3., 1, 'o', 'g', 'Tibet-III QGSJET 2008')
    # plot_data_statonly(ax, 'ICETOP_SIBYLL_21_all_energy.txt', 3., 1, 'o', 'tab:orange', 'IceTop SIBYLL')
    # plot_data_statonly(ax, 'KASCADE_SIBYLL_21_all_energy.txt', 3., 1, 'o', 'tab:red', 'KASCADE SIBYLL 2005')
    # plot_data_statonly(ax, 'KGRANDE_QGSJET-II-04_all_energy.txt', 3., 1, 'o', 'navy', 'KASCADE-Grande QGSJET 2013')
    # plot_data_statonly(ax, 'KGRANDE_SIBYLL_23_all_energy.txt', 3., 1, 'o', 'tab:olive', 'KASCADE-Grande QGSJET 2013')
    # plot_data_statonly(ax, 'HAWC_QGSJET-II-04_all_energy.txt', 3., 1, 'o', 'y', 'HAWC')
    # plot_data_statonly(ax, 'DAMPE_all_energy.txt', 3., 1, 'o', 'c', 'DAMPE', 20)
    # plot_data_statonly(ax, 'NUCLEON_all_energy.txt', 3., 1, 'o', 'tab:gray', 'NUCLEON', 3)
 
    x = 3e9
    y = np.logspace(6.56, 6.88, 10)
    fontsize = 15
    ax.text(x, y[0], 'GAMMA', fontsize=fontsize, color='m')
    ax.text(x, y[1], 'IceCube+IceTop', fontsize=fontsize, color='c')
    ax.text(x, y[2], 'IceTop', fontsize=fontsize, color='tab:orange')
    ax.text(x, y[3], 'KASCADE', fontsize=fontsize, color='tab:gray')
    ax.text(x, y[4], 'LHAASO', fontsize=fontsize, color='navy')
    ax.text(x, y[5], 'Pierre Auger Obs', fontsize=fontsize, color='r')
    ax.text(x, y[6], 'TALE', fontsize=fontsize, color='y')
    ax.text(x, y[7], 'Tibet-III', fontsize=fontsize, color='g')
    ax.text(x, y[8], 'Telescope Array', fontsize=fontsize, color='b')
    ax.text(x, y[9], 'TUNKA-133', fontsize=fontsize, color='tab:brown')

    ax.annotate('1st knee',xy=(4e6, 2.6e6),xytext=(4e6, 2.0e6),
                arrowprops=dict(color='r'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=15, color='r')

    ax.annotate('1st ankle',xy=(20e6, 0.9 * 2.6e6),xytext=(20e6, 0.9 * 2.0e6),
                arrowprops=dict(color='b'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=15, color='b')

    ax.annotate('2nd knee',xy=(100e6, 2.6e6),xytext=(100e6, 2.0e6),
                arrowprops=dict(color='r'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=15, color='r')

    ax.annotate('2nd ankle',xy=(5e9, 0.45 * 2.6e6),xytext=(5e9, 0.45 * 2.0e6),
                arrowprops=dict(color='b'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=15, color='b')
    
    savefig(fig, 'EVA_allparticles_dataonly.pdf')

def plot_lnA_dataonly():
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
    set_axes(ax, xlabel=XLABEL, ylabel=r'$\langle$ ln A $\rangle$', xscale='log', xlim=[5e5, 2e10], ylim=[0, np.log(56)])

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
    
    plot_data(ax, 'LHAASO_QGSJET-II-04_lnA_energy.txt', 0, 1, 'o', 'navy', 'LHAASO', 22)

    ax.annotate('1st knee',xy=(4e6, 0.5),xytext=(4e6, 0.2),
                arrowprops=dict(color='r'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=15, color='r')

    ax.annotate('1st ankle',xy=(20e6, 0.5),xytext=(20e6, 0.2),
                arrowprops=dict(color='b'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=15, color='b')

    ax.annotate('2nd knee',xy=(100e6, 0.5),xytext=(100e6, 0.2),
                arrowprops=dict(color='r'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=15, color='r')

    ax.annotate('2nd ankle',xy=(5e9, 0.5),xytext=(5e9, 0.2),
                arrowprops=dict(color='b'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=15, color='b')
    
    ax.fill_between([1e5, 1e11], np.log(1), np.log(4), color='tab:olive', alpha=0.10)
    ax.fill_between([1e5, 1e11], np.log(4), np.log(14), color='tab:orange', alpha=0.10)
    ax.fill_between([1e5, 1e11], np.log(14), np.log(24), color='tab:red', alpha=0.10)
    ax.fill_between([1e5, 1e11], np.log(24), np.log(56), color='tab:brown', alpha=0.10)

    x = 8e5
    y = np.linspace(2.25, 3.65, 9)
    fontsize = 15
    ax.text(x, y[0], 'Casa-Blanca', fontsize=fontsize, color='tab:brown')
    ax.text(x, y[1], 'Hi-Res', fontsize=fontsize, color='tab:gray')
    ax.text(x, y[2], 'Hi-Res MIA', fontsize=fontsize, color='tab:orange')
    ax.text(x, y[3], 'LHAASO', fontsize=fontsize, color='navy')
    ax.text(x, y[4], 'Pierre Auger Xmax', fontsize=fontsize, color='r')
    ax.text(x, y[5], 'Telescope Array', fontsize=fontsize, color='b')
    ax.text(x, y[6], 'TUNKA', fontsize=fontsize, color='tab:olive')
    ax.text(x, y[7], 'Yakutsk Small', fontsize=fontsize, color='tab:pink')
    ax.text(x, y[8], 'Yakutsk Big', fontsize=fontsize, color='tab:cyan')

    savefig(fig, 'EVA_lnA_dataonly.pdf')
    
if __name__== "__main__":
    plot_allparticles_dataonly()
    plot_lnA_dataonly()