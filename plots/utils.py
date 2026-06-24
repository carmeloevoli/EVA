import numpy as np
import matplotlib.pyplot as plt

def savefig(fig: plt.Figure, filename: str, dpi: int = 300, bbox_inches: str = 'tight', pad_inches: float = 0.1, transparent: bool = False) -> None:
    """
    Save the given matplotlib figure to a file with enhanced options for better quality.
    
    Parameters:
    - fig: The matplotlib Figure object to save.
    - filename: The file name or path where the figure will be saved.
    - dpi: The resolution in dots per inch (default is 300 for high-quality).
    - bbox_inches: Adjusts bounding box ('tight' will minimize excess whitespace).
    - pad_inches: Amount of padding around the figure when bbox_inches is 'tight' (default is 0.1).
    - transparent: Whether to save the plot with a transparent background (default is False).
    """
    
    try:
        fig.savefig(f'../figures/{filename}', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=transparent, format='pdf')
        print(f'Plot successfully saved to {filename} with dpi={dpi}, bbox_inches={bbox_inches}, pad_inches={pad_inches}, transparent={transparent}')
    except Exception as e:
        print(f"Error saving plot to {filename}: {e}")

def _calculate_errors(err_sta_lo: np.ndarray, err_sta_up: np.ndarray, err_sys_lo: np.ndarray, err_sys_up: np.ndarray) -> tuple:
    """Calculate combined statistical and systematic errors."""
    err_tot_lo = np.sqrt(err_sta_lo**2 + err_sys_lo**2)
    err_tot_up = np.sqrt(err_sta_up**2 + err_sys_up**2)
    return err_tot_lo, err_tot_up

def _normalize_data(x: np.ndarray, y: np.ndarray, err_tot_lo: np.ndarray, err_tot_up: np.ndarray, slope: float, norm: float) -> tuple:
    """Normalize data values by given slope and normalization factor."""
    x_norm = x / norm
    scaling = norm * np.power(x_norm, slope)
    y_norm = scaling * y
    y_err_lo_norm = scaling * err_tot_lo
    y_err_up_norm = scaling * err_tot_up
    return x_norm, y_norm, y_err_lo_norm, y_err_up_norm

def load_data(filename, slope, norm, min_energy, max_energy=1e20):
    from utils import _calculate_errors, _normalize_data

    path = f'data/{filename}'
    cols = (0, 1, 2, 3, 4, 5)
    x, y, err_sta_lo, err_sta_up, err_sys_lo, err_sys_up = np.loadtxt(path, usecols=cols, unpack=True)

    err_tot_lo, err_tot_up = _calculate_errors(err_sta_lo, err_sta_up, err_sys_lo, err_sys_up)

    x_norm, y_norm, y_err_lo_norm, y_err_up_norm = _normalize_data(x, y, err_tot_lo, err_tot_up, slope, norm)
    
    mask = (x_norm > min_energy) & (x_norm < max_energy)
    return x_norm[mask], y_norm[mask], y_err_lo_norm[mask], y_err_up_norm[mask]

def plot_data(ax: plt.Axes, filename: str, slope: float, norm: float, fmt: str, color: str, label: str, zorder: int = 1) -> None:
    """
    Load data from file, normalize, and plot with error bars.
    
    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - filename: The path to the data file.
    - slope: The power to which the data should be scaled.
    - norm: Normalization factor for the data.
    - fmt: Format string for the plot markers.
    - color: Color of the plot markers and lines.
    - label: Label for the plot legend.
    - zorder: Z-order for layering the plot.
    """
    try:
        x, y, err_sta_lo, err_sta_up, err_sys_lo, err_sys_up = np.loadtxt(f'./data/{filename}', usecols=(0, 1, 2, 3, 4, 5), unpack=True)
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return
        
    # Normalize data
    x_norm, y_norm, y_err_lo_norm, y_err_up_norm = _normalize_data(x, y, err_sta_lo, err_sta_up, slope, norm)
    
    # Plot the data with error bars
    ax.errorbar(x_norm, y_norm, yerr=[y_err_lo_norm, y_err_up_norm], fmt=fmt, markeredgecolor=color, color=color, 
                label=label, capsize=4.0, markersize=5.8, elinewidth=1.8, capthick=1.8, zorder=zorder)

    # Normalize data
    x_norm, y_norm, y_err_lo_norm, y_err_up_norm = _normalize_data(x, y, err_sys_lo, err_sys_up, slope, norm)
    
    # Plot the data with error bars
    ax.errorbar(x_norm, y_norm, yerr=[y_err_lo_norm, y_err_up_norm], fmt=fmt, markeredgecolor=color, color=color, 
                capsize=4.0, markersize=5.8, elinewidth=1.8, capthick=1.8, zorder=zorder)

def plot_data_statonly(ax: plt.Axes, filename: str, slope: float, norm: float, fmt: str, color: str, label: str, zorder: int = 1) -> None:
    """
    Load data from file, normalize, and plot with error bars.
    
    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - filename: The path to the data file.
    - slope: The power to which the data should be scaled.
    - norm: Normalization factor for the data.
    - fmt: Format string for the plot markers.
    - color: Color of the plot markers and lines.
    - label: Label for the plot legend.
    - zorder: Z-order for layering the plot.
    """
    try:
        x, y, err_sta_lo, err_sta_up, err_sys_lo, err_sys_up = np.loadtxt(f'./data/{filename}', usecols=(0, 1, 2, 3, 4, 5), unpack=True)
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return
    
    # Normalize data
    x_norm, y_norm, y_err_lo_norm, y_err_up_norm = _normalize_data(x, y, err_sta_lo, err_sta_up, slope, norm)
    
    # Plot the data with error bars
    ax.errorbar(x_norm, y_norm, yerr=[y_err_lo_norm, y_err_up_norm], fmt=fmt, markeredgecolor=color, color=color, 
                label=label, capsize=4.0, markersize=5.8, elinewidth=1.8, capthick=1.8, zorder=zorder)
    
def set_axes(ax: plt.Axes, xlabel: str, ylabel: str, xscale: str = 'linear', yscale: str = 'linear', xlim: tuple = None, ylim: tuple = None) -> None:
    """
    Set the properties for the axes of a plot.
    
    Parameters:
    - ax: Matplotlib Axes object.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - xscale: Scale of the x-axis ('linear' or 'log').
    - yscale: Scale of the y-axis ('linear' or 'log').
    - xlim: Limits for the x-axis (min, max).
    - ylim: Limits for the y-axis (min, max).
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Validate and set axis scale
    if xscale in ['linear', 'log']:
        ax.set_xscale(xscale)
    else:
        print(f"Invalid xscale '{xscale}', defaulting to 'linear'.")
        ax.set_xscale('linear')
    if yscale in ['linear', 'log']:
        ax.set_yscale(yscale)
    else:
        print(f"Invalid yscale '{yscale}', defaulting to 'linear'.")
        ax.set_yscale('linear')

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
