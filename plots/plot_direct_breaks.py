import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import plot_data, set_axes, savefig
from plot_galactic import XLABEL, YLABEL

def plot_PAMELA():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e2, 5e3], ylim=[8e3, 13e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'PAMELA_H_energy.txt', 2.7, 1, 'o', 'tab:blue', 'PAMELA', zorder=3)

    x, y, yerr = np.loadtxt('output/PAMELA_H_break.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    ax.legend(fontsize=26)
    savefig(fig, f'EVA_direct_breaks_PAMELA.pdf')

def plot_AMS02():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e2, 5e3], ylim=[8e3, 13e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'AMS-02_H_energy.txt', 2.7, 1, 'o', 'tab:blue', 'AMS-02', zorder=3)

    x, y, yerr = np.loadtxt('output/AMS-02_H_break.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    ax.legend(fontsize=26)
    savefig(fig, f'EVA_direct_breaks_AMS-02.pdf')

def plot_CALET():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e2, 5e4], ylim=[8e3, 16e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'CALET_H_energy.txt', 2.7, 1, 'o', 'tab:blue', 'CALET', zorder=3)

    x, y, yerr = np.loadtxt('output/CALET_H_break.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    ax.legend(fontsize=26)
    savefig(fig, f'EVA_direct_breaks_CALET.pdf')

def plot_DAMPE():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[1e2, 5e4], ylim=[8e3, 16e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'CALET_H_energy.txt', 2.7, 1, 'o', 'tab:blue', 'CALET', zorder=3)

    x, y, yerr = np.loadtxt('output/CALET_H_break.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    ax.legend(fontsize=26)
    savefig(fig, f'EVA_direct_breaks_DAMPE.pdf')

def plot_CREAM():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[5e2, 5e5], ylim=[4e3, 16e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'CREAM_H_energy.txt', 2.7, 1, 'o', 'tab:blue', 'CREAM', zorder=3)

    x, y, yerr = np.loadtxt('output/CREAM_H_break.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    ax.legend(fontsize=26)
    savefig(fig, f'EVA_direct_breaks_CREAM.pdf')

def plot_ISSCREAM():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[5e2, 5e5], ylim=[4e3, 16e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'ISS-CREAM_H_energy.txt', 2.7, 1, 'o', 'tab:blue', 'ISS-CREAM', zorder=3)

    x, y, yerr = np.loadtxt('output/ISS-CREAM_H_break.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    ax.legend(fontsize=26)
    savefig(fig, f'EVA_direct_breaks_ISS-CREAM.pdf')

def plot_NUCLEON():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[5e2, 5e5], ylim=[4e3, 16e3], xscale='log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    plot_data(ax, 'NUCLEON_H_energy.txt', 2.7, 1, 'o', 'tab:blue', 'NUCLEON', zorder=3)

    x, y, yerr = np.loadtxt('output/NUCLEON_H_break.txt', usecols=(0,1,2), unpack=True)

    ax.plot(x, y, color='tab:red', zorder=10)
    ax.fill_between(x, y - yerr, y + yerr, color='tab:gray', alpha=0.20, zorder=1)

    ax.legend(fontsize=26)
    savefig(fig, f'EVA_direct_breaks_NUCLEON.pdf')

if __name__== "__main__":
    plot_PAMELA()
    plot_AMS02()
    plot_CALET()
    plot_DAMPE()
    plot_CREAM()
    plot_NUCLEON()
    plot_ISSCREAM()
