import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, plot_data_statonly, savefig

XLABEL = r'E [GeV]'
YLABEL = r'E$^{2.75}$ I [GeV$^{1.75}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'

def plot_LHAASO():
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[2e5, 5e7], ylim=[4e4, 1.1e5], xscale='log', yscale='linear')
    ax.yaxis.label.set_color('tab:red')
    ax.tick_params(axis='y', colors='tab:red')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))

    plot_data(ax, 'LHAASO_QGSJET-II-04_all_energy.txt', 2.75, 1, 'o', 'tab:red', 'QGSJET-II-04', 8)
    plot_data(ax, 'LHAASO_EPOS-LHC_all_energy.txt', 2.75, 1, '*', 'tab:orange', 'EPOS-LHC', 7)
    plot_data(ax, 'LHAASO_SIBYLL-23_all_energy.txt', 2.75, 1, 's', 'tab:olive', 'SIBYLL-23', 6)
    
    #ax.legend(fontsize=16)

    ax2 = ax.twinx()
    ax2.yaxis.label.set_color('tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')

    ax2.set_xscale('log')
    ax2.set_xlim([2e5, 5e7])
    ax2.set_ylabel(r'$\ln(A)$')
    ax2.set_ylim([1, 3])

    plot_data(ax2, 'LHAASO_QGSJET-II-04_lnA_energy.txt', 0., 1, 'o', 'tab:blue', 'LHAASO', 8)
    plot_data(ax2, 'LHAASO_EPOS-LHC_lnA_energy.txt', 0., 1, 'o', 'tab:purple', 'LHAASO', 7)
    plot_data(ax2, 'LHAASO_SIBYLL-23_lnA_energy.txt', 0., 1, 'o', 'tab:cyan', 'LHAASO', 6)

    ax.text(0.05, 0.11, 'LHAASO Coll., PRL, 2024', fontsize=16, ha='left', va='top', transform=ax.transAxes)

    ax.vlines(3e6, 1e1, 1e6, ls='--', lw=5, color='tab:gray', alpha=0.4)

    savefig(fig, 'EVA_LHAASO_data.pdf')

if __name__== "__main__":
    plot_LHAASO()
