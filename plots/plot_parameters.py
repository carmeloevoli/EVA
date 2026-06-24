import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, savefig

def add_slope(ax, x, y, err_y, color):
    fmt = 'o'
    label = ''

    ax.errorbar(x, y, yerr=err_y, fmt=fmt, markeredgecolor=color, color=color, 
                label=label, capsize=8.0, markersize=15., elinewidth=3.8, capthick=3.8)

    ax.fill_between([0, 5], y - err_y, y + err_y, color=color, alpha=0.1)

def plot_slopes():
    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel='', ylabel='slope - 2.7', xlim=[0.6, 3.4], ylim=[0.08, 0.24])
        
    xs = [1, 2, 3]
    labels = [r'H', r'He', r'$Z > 2$']
    ax.set_xticks(xs, labels)

    add_slope(ax, 1, 0.121, 0.015, 'tab:red')
    add_slope(ax, 2, 0.197, 0.020, 'tab:blue')
    add_slope(ax, 3, 0.168, 0.032, 'tab:green')

    savefig(fig, 'EVA_params_slopes.pdf')

if __name__== "__main__":
    plot_slopes()