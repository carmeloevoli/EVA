import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data_statonly, plot_data, savefig

def plot_data_point(ax, x, dx_lo, dx_hi, y, dy_lo, dy_hi, fmt, color, label, zorder=1):
    y = abs(y)
    x *= 1e3
    dx_lo *= 1e3
    dx_hi *= 1e3
    if label == 'none':
        ax.errorbar([x], [y], xerr=([dx_lo], [dx_hi]), yerr=([dy_lo], [dy_hi]),
                    fmt=fmt, markeredgecolor=color, color=color, capsize=4.5, markersize=7, elinewidth=2.4, capthick=2.4, zorder=zorder)
    else:
        ax.errorbar([x], [y], xerr=([dx_lo], [dx_hi]), yerr=([dy_lo], [dy_hi]),
                    fmt=fmt, markeredgecolor=color, color=color, capsize=4.5, markersize=7, elinewidth=2.4, capthick=2.4, label=label, zorder=zorder)

def plot_direct_breaks():
    def add_data_point(ax, y, dy, x, dx, color, zorder):
        fmt = 'o'
        ax.errorbar(x, y, xerr=dx, yerr=dy,
                    fmt=fmt, markeredgecolor=color, color=color, capsize=4.5, markersize=7, elinewidth=2.4, capthick=2.4, zorder=zorder)

    fig, ax = plt.subplots(figsize=(11.50, 8.5))
    set_axes(ax, xlabel=r'log E [GeV]', ylabel=r'$\Delta \alpha$', xscale='linear', xlim=[1.8, 5.8], ylim=[-1, 1])
    
    add_data_point(ax, 2.522e-01, 1.113e-01, 2.435e+00, 2.316e-01, 'tab:gray', 7) # PAMELA
    add_data_point(ax, 1.348e-01, 4.398e-02, 2.519e+00, 2.132e-01, 'tab:orange', 8) # AMS-02
    add_data_point(ax, -2.720e-01, 3.213e-01, 4.221e+00, 2.448e-01, 'tab:blue', 6) # CREAM
    add_data_point(ax, -2.726e-01, 4.284e-01, 3.932e+00, 2.343e-01, 'tab:cyan', 5) # ISS-CREAM
    add_data_point(ax, -1.737e-01, 3.627e-01, 4.532e+00, 8.392e-01, 'tab:olive', 4) # NUCLEON 
    add_data_point(ax, 2.444e-01, 7.415e-02, 2.790e+00, 2.286e-01, 'g', 9) # CALET
    add_data_point(ax, -3.000e-01, 3.801e-01, 3.954e+00, 1.627e-01, 'g', 9) # CALET 
    add_data_point(ax, 1.631e-01, 1.129e-01, 2.735e+00, 4.878e-01, 'r', 10) # DAMPE 
    add_data_point(ax, -2.624e-01, 1.114e-01, 4.113e+00, 3.509e-01, 'r', 10) # DAMPE


    x = 2.35
    y = np.linspace(-0.1, -0.4, 4)
    ax.text(x, y[0], 'PAMELA', fontsize=20, color='tab:gray')
    ax.text(x, y[1], 'AMS-02', fontsize=20, color='tab:orange')
    ax.text(x, y[2], 'CALET', fontsize=20, color='g')
    ax.text(x, y[3], 'DAMPE', fontsize=20, color='r')

    x = 4.05
    y = np.linspace(0.25, 0.65, 5)

    ax.text(x, y[0], 'DAMPE', fontsize=20, color='r')
    ax.text(x, y[1], 'CALET', fontsize=20, color='g')
    ax.text(x, y[2], 'CREAM', fontsize=20, color='tab:blue')
    ax.text(x, y[3], 'ISS-CREAM', fontsize=20, color='tab:cyan')
    ax.text(x, y[4], 'NUCLEON', fontsize=20, color='tab:olive')
    
    savefig(fig, f'EVA_direct_breaks.pdf')

#     plot_data_point(ax, 0.29, 0.19, 0.37, 2.84 - 2.66, 0.16 + 0.02, 0.10 + 0.03, 'v', 'tab:olive', 'PAMELA', 1)
#     plot_data_point(ax, 0.38, 0.15, 0.51, 2.82 - 2.64, 0.14 + 0.01, 0.04 + 0.07, '^', 'tab:green', 'AMS-02', 2)
#     plot_data_point(ax, 15.83, 6.45, 18.65, 2.84 - 2.58, 0.13 + 0.01, 0.16 + 0.01, 'o', 'tab:purple', 'CREAM', 3)

#     plot_data_point(ax, 0.62, 0.04, 0.05, 2.81 - 2.56, 0.02, 0.02, 'D', 'tab:orange', 'CALET', 4)
#     plot_data_point(ax, 9.72, 1.22, 1.56, 2.90 - 2.56, 0.06, 0.08, 'D', 'tab:orange', 'none', 4)

#     plot_data_point(ax, 0.54, 0.05, 0.05, 2.76 - 2.58, 0.02, 0.03, 's', 'tab:red', 'DAMPE', 5)
#     plot_data_point(ax, 12.74, 2.09, 3.76, 2.86 - 2.58, 0.05, 0.08, 's', 'tab:red', 'none', 5)

# #    plot_data_point(ax, 36.02, 12.06, 42.84, 2.69 - 2.53, 0.06, 0.23, 'o', 'tab:gray', 'NUCLEON', 1)

#     plot_data_point(ax, 9.89, 1.99, 2.20, 2.84 - 2.57, 0.08, 0.08, 'o', 'tab:blue', 'ISS-CREAM', 6)
#     plot_data_point(ax, 200., 0, 0, 2.84 - 2.58, 0, 0, 'o', 'tab:blue', 'none', 6)

#     #ax.legend(fontsize=14)
    
#     handles, labels = ax.get_legend_handles_labels()
#     handles = [h[0] for h in handles]
#     ax.legend(handles, labels, loc='best', fontsize=14, markerscale=1.3)

if __name__== "__main__":
    plot_direct_breaks()