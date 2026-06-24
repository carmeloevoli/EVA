import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data_statonly, plot_data, savefig

XLABEL = r'E [GeV]'
YLABEL = r'E$^{2.7}$ I [GeV$^{2}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'

def plot_protons():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e2, 1e6], ylim=[1e3, 22e3], xscale='log', yscale='linear')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    #plot_data(ax, 'PAMELA_H_energy.txt', 2.7, 1., 'o', 'tab:olive', 'PAMELA', 1)
    #plot_data(ax, 'AMS-02_H_energy.txt', 2.7, 1., 'o', 'tab:green', 'AMS-02', 2)
    plot_data_statonly(ax, 'DAMPE_H_energy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)
    plot_data_statonly(ax, 'CALET_H_energy.txt', 2.7, 1., 'o', 'tab:orange', 'CALET', 4)
    #plot_data(ax, 'CREAM_H_energy.txt', 2.7, 1., 'o', 'tab:purple', 'CREAM', 5)
    #plot_data(ax, 'ISS-CREAM_H_energy.txt', 2.7, 1., 'o', 'tab:blue', 'ISS-CREAM', 6)
    #plot_data(ax, 'kiss_tables/NUCLEON_H_totalEnergy.txt', 2.7, 1., 'o', 'tab:gray', 'NUCLEON', 1)

    ax.annotate(r'$E = 0.5$ TeV',xy=(5e2, 12e3),xytext=(5e2, 13.5e3),
                arrowprops=dict(color='r'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=18, color='r')

    ax.annotate(r'$E = 1.4$ TeV',xy=(1.4e4, 10e3), xytext=(1.4e4, 7.65e3),
                arrowprops=dict(color='b'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=18, color='b')


#    ax.text(2e2, 20e3, 'stat/sys uncertainties', color='tab:gray', fontsize=20)

    ax.legend(fontsize=15, loc='best')
    savefig(fig, 'EVA_proton_spectrum.pdf')

def plot_he_spectrum():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e2, 1e6], ylim=[1e3, 22e3], xscale='log', yscale='linear')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'PAMELA_He_energy.txt', 2.7, 1., 'o', 'tab:olive', 'PAMELA', 1)
    plot_data(ax, 'AMS-02_He_energy.txt', 2.7, 1., 'o', 'tab:green', 'AMS-02', 2)
    plot_data(ax, 'DAMPE_He_energy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)
    plot_data(ax, 'CALET_He_energy.txt', 2.7, 1., 'o', 'tab:orange', 'CALET', 4)
    plot_data(ax, 'CREAM_He_energy.txt', 2.7, 1., 'o', 'tab:purple', 'CREAM', 5)
#    plot_data(ax, 'kiss_tables/ISS-CREAM_He_kineticEnergy.txt', 2.7, 1e-3, 'o', 'tab:blue', 'ISS-CREAM', 6)
    #plot_data(ax, 'kiss_tables/NUCLEON_H_totalEnergy.txt', 2.7, 'o', 'tab:gray', 'CREAM')

    ax.annotate(r'$E = 1$ TeV',xy=(2. * 5e2, 10e3),xytext=(2. * 5e2, 11.5e3),
                arrowprops=dict(color='r'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=18, color='r')

    ax.annotate(r'$E = 2.8$ TeV',xy=(2. * 1.4e4, 10e3), xytext=(2. * 1.4e4, 7.9e3),
                arrowprops=dict(color='b'), bbox=dict(pad=10, facecolor='none', edgecolor='none'),
                ha='center', va='bottom', fontsize=18, color='b')
    
    ax.legend(fontsize=15, loc='best')
    savefig(fig, 'EVA_He_spectrum.pdf')
 
if __name__== "__main__":
    plot_protons()
    #plot_he_spectrum()

    #plot_lnA_dataonly()