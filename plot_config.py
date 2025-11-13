"""
Global plot configuration and formatting utilities.
Ensures consistent, publication-ready plots across all modules.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional, List


# Global plot configuration
def configure_plot_defaults():
    """Set global matplotlib parameters for consistent styling."""
    plt.rcParams.update({
        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'svg',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Font
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 0,  # No titles
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,

        # Grid - DISABLED
        'axes.grid': False,
        'grid.alpha': 0.0,

        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,

        # Ticks
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Legend
        'legend.frameon': False,
        'legend.loc': 'best',
    })


def format_axis(
    ax,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,  # Will be ignored (no titles)
    legend: bool = True,
    legend_labels: Optional[List[str]] = None,
    grid: bool = False,  # Always False
    log_x: bool = False,
    log_y: bool = False,
):
    """
    Apply consistent formatting to a matplotlib axis.

    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label (short and concise)
        ylabel: Y-axis label (short and concise)
        title: Ignored (no titles in plots)
        legend: Whether to show legend
        legend_labels: Custom legend labels
        grid: Ignored (no grids)
        log_x: Use logarithmic x-axis
        log_y: Use logarithmic y-axis
    """
    # NO GRID
    ax.grid(False)

    # NO TITLE
    ax.set_title('')

    # Labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Log scales
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # Legend
    if legend:
        if legend_labels:
            ax.legend(legend_labels, frameon=False, loc='best')
        else:
            ax.legend(frameon=False, loc='best')
    else:
        legend_obj = ax.get_legend()
        if legend_obj:
            legend_obj.remove()

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Tight layout
    ax.figure.tight_layout()


def get_diffusion_colors():
    """
    Get consistent colors for diffusion types.

    Returns:
        Dictionary mapping diffusion type labels to colors
    """
    return {
        0: '#1f77b4',  # Normal - Blue
        1: '#ff7f0e',  # Subdiffusion - Orange
        2: '#2ca02c',  # Confined - Green
        3: '#d62728',  # Superdiffusion - Red
    }


def get_diffusion_labels():
    """
    Get consistent short English labels for diffusion types.

    Returns:
        Dictionary mapping diffusion type IDs to short labels
    """
    return {
        0: 'Normal',
        1: 'Subdiffusion',
        2: 'Confined',
        3: 'Superdiffusion'
    }


def save_figure(fig, filepath, dpi=300):
    """
    Save figure as high-quality SVG.

    Args:
        fig: Matplotlib figure
        filepath: Output path (will force .svg extension)
        dpi: Resolution (for rasterized elements)
    """
    # Ensure SVG format
    filepath = str(filepath)
    if not filepath.endswith('.svg'):
        filepath = filepath.rsplit('.', 1)[0] + '.svg'

    fig.savefig(filepath, format='svg', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def apply_publication_style(fig=None, ax=None):
    """
    Apply publication-ready styling to current or specified figure/axis.
    Removes grids, titles, and formats spines.

    Args:
        fig: Optional figure to format (default: current figure)
        ax: Optional axis to format (default: current axis or all axes in fig)
    """
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        axes = fig.get_axes()
    else:
        axes = [ax] if not isinstance(ax, list) else ax

    for ax in axes:
        # Remove grid
        ax.grid(False)

        # Remove title
        ax.set_title('')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set linewidth for remaining spines
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)

        # Format legend if present
        legend = ax.get_legend()
        if legend:
            legend.set_frame_on(False)

    fig.tight_layout()


# Initialize defaults when module is imported
configure_plot_defaults()
