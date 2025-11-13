"""
Advanced time series plotting functions with improved formatting.
All plots follow consistent style: no grid, no title, concise English legends, SVG output.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib import cm

try:
    from plot_config import format_axis, get_diffusion_colors, get_diffusion_labels, save_figure
    from logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Fallback if plot_config not available
    def format_axis(ax, **kwargs):
        ax.grid(False)
        ax.set_title('')

    def get_diffusion_colors():
        return {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}

    def get_diffusion_labels():
        return {0: 'Normal', 1: 'Subdiffusion', 2: 'Confined', 3: 'Superdiffusion'}

    def save_figure(fig, filepath, dpi=300):
        fig.savefig(filepath, format='svg', dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_diffusion_fractions_area(df: pd.DataFrame, save_path: Path,
                                    progress_callback=None):
    """
    Plot stacked area chart showing diffusion type distribution over time.

    Args:
        df: Combined dataframe with 'polymerization_time' and 'predicted_label' columns
        save_path: Output path for SVG
        progress_callback: Optional progress callback
    """
    if progress_callback:
        progress_callback("Creating diffusion fraction area plot...")

    # Group by time and label, count tracks
    grouped = df.groupby(['polymerization_time', 'predicted_label']).size().reset_index(name='count')

    # Pivot to get labels as columns
    pivot = grouped.pivot(index='polymerization_time', columns='predicted_label', values='count').fillna(0)

    # Calculate fractions
    totals = pivot.sum(axis=1)
    fractions = pivot.div(totals, axis=0) * 100  # Convert to percentage

    # Sort times
    fractions = fractions.sort_index()
    times = fractions.index.values

    # Get colors and labels
    colors = get_diffusion_colors()
    label_names = get_diffusion_labels()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Stacked area plot
    bottom = np.zeros(len(times))
    for label_id in sorted(fractions.columns):
        if label_id in fractions.columns:
            values = fractions[label_id].values
            color = colors.get(label_id, '#888888')
            label = label_names.get(label_id, f'Type {label_id}')

            ax.fill_between(times, bottom, bottom + values,
                           color=color, alpha=0.8, label=label, edgecolor='none')
            bottom += values

    # Format
    format_axis(ax,
                xlabel='Polymerization Time',
                ylabel='Fraction (%)',
                legend=True)

    ax.set_ylim(0, 100)

    # Save
    save_figure(fig, save_path)


def plot_D_boxplots_per_diffusion_type(df: pd.DataFrame, save_dir: Path,
                                         progress_callback=None):
    """
    Create boxplots for diffusion coefficient D over time, separated by diffusion type.

    Args:
        df: Combined dataframe with 'polymerization_time', 'predicted_label', and 'D' columns
        save_dir: Directory to save plots
        progress_callback: Optional progress callback
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    colors = get_diffusion_colors()
    label_names = get_diffusion_labels()

    times = sorted(df['polymerization_time'].unique())

    for label_id in sorted(df['predicted_label'].unique()):
        if progress_callback:
            progress_callback(f"  Creating D boxplot for {label_names.get(label_id, f'Type {label_id}')}...")

        # Filter data for this diffusion type
        df_label = df[df['predicted_label'] == label_id].copy()

        if df_label.empty or 'D' not in df_label.columns:
            continue

        # Prepare data for boxplot
        data_by_time = []
        valid_times = []

        for t in times:
            d_values = df_label[df_label['polymerization_time'] == t]['D'].dropna()
            if len(d_values) > 0:
                data_by_time.append(d_values.values)
                valid_times.append(t)

        if not data_by_time:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # Boxplot
        bp = ax.boxplot(data_by_time, positions=valid_times, widths=np.diff(valid_times).min() * 0.6 if len(valid_times) > 1 else 1,
                        patch_artist=True, showfliers=False)

        # Color boxes
        color = colors.get(label_id, '#888888')
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Format
        label_name = label_names.get(label_id, f'Type {label_id}')
        format_axis(ax,
                    xlabel='Polymerization Time',
                    ylabel=f'D (μm²/s)',
                    legend=False)

        # Save
        save_path = save_dir / f'D_boxplot_{label_name.lower().replace(" ", "_")}.svg'
        save_figure(fig, save_path)


def plot_D_log_linear_per_diffusion_type(df: pd.DataFrame, save_dir: Path,
                                          progress_callback=None):
    """
    Create log and linear plots for D over time, per diffusion type.

    Args:
        df: Combined dataframe with 'polymerization_time', 'predicted_label', and 'D'
        save_dir: Directory to save plots
        progress_callback: Optional progress callback
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    colors = get_diffusion_colors()
    label_names = get_diffusion_labels()

    for label_id in sorted(df['predicted_label'].unique()):
        label_name = label_names.get(label_id, f'Type {label_id}')

        if progress_callback:
            progress_callback(f"  Creating D plots (log/linear) for {label_name}...")

        # Filter data
        df_label = df[df['predicted_label'] == label_id].copy()

        if df_label.empty or 'D' not in df_label.columns:
            continue

        # Group by time and calculate statistics
        stats = df_label.groupby('polymerization_time')['D'].agg(['mean', 'std', 'count']).reset_index()
        stats = stats[stats['count'] > 0]

        if stats.empty:
            continue

        times = stats['polymerization_time'].values
        mean_D = stats['mean'].values
        std_D = stats['std'].fillna(0).values

        color = colors.get(label_id, '#888888')

        # Linear plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(times, mean_D, yerr=std_D, fmt='o-', color=color,
                   linewidth=2, markersize=6, capsize=4, label=label_name)

        format_axis(ax,
                    xlabel='Polymerization Time',
                    ylabel='D (μm²/s)',
                    legend=False)

        save_path = save_dir / f'D_linear_{label_name.lower().replace(" ", "_")}.svg'
        save_figure(fig, save_path)

        # Log plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(times, mean_D, yerr=std_D, fmt='o-', color=color,
                   linewidth=2, markersize=6, capsize=4, label=label_name)

        format_axis(ax,
                    xlabel='Polymerization Time',
                    ylabel='D (μm²/s)',
                    legend=False,
                    log_y=True)

        save_path = save_dir / f'D_log_{label_name.lower().replace(" ", "_")}.svg'
        save_figure(fig, save_path)


def plot_alpha_log_linear_per_diffusion_type(df: pd.DataFrame, save_dir: Path,
                                              progress_callback=None):
    """
    Create log and linear plots for alpha over time, per diffusion type.

    Args:
        df: Combined dataframe with 'polymerization_time', 'predicted_label', and 'alpha'
        save_dir: Directory to save plots
        progress_callback: Optional progress callback
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    colors = get_diffusion_colors()
    label_names = get_diffusion_labels()

    for label_id in sorted(df['predicted_label'].unique()):
        label_name = label_names.get(label_id, f'Type {label_id}')

        if progress_callback:
            progress_callback(f"  Creating alpha plots (log/linear) for {label_name}...")

        # Filter data
        df_label = df[df['predicted_label'] == label_id].copy()

        if df_label.empty or 'alpha' not in df_label.columns:
            continue

        # Group by time and calculate statistics
        stats = df_label.groupby('polymerization_time')['alpha'].agg(['mean', 'std', 'count']).reset_index()
        stats = stats[stats['count'] > 0]

        if stats.empty:
            continue

        times = stats['polymerization_time'].values
        mean_alpha = stats['mean'].values
        std_alpha = stats['std'].fillna(0).values

        color = colors.get(label_id, '#888888')

        # Linear plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(times, mean_alpha, yerr=std_alpha, fmt='o-', color=color,
                   linewidth=2, markersize=6, capsize=4, label=label_name)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        format_axis(ax,
                    xlabel='Polymerization Time',
                    ylabel='α (Anomalous Exponent)',
                    legend=False)

        save_path = save_dir / f'alpha_linear_{label_name.lower().replace(" ", "_")}.svg'
        save_figure(fig, save_path)

        # Log plot
        fig, ax = plt.subplots(figsize=(8, 5))
        # For log scale, we need positive values only
        mask = mean_alpha > 0
        if mask.sum() > 0:
            ax.errorbar(times[mask], mean_alpha[mask], yerr=std_alpha[mask],
                       fmt='o-', color=color, linewidth=2, markersize=6,
                       capsize=4, label=label_name)
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            format_axis(ax,
                        xlabel='Polymerization Time',
                        ylabel='α (Anomalous Exponent)',
                        legend=False,
                        log_y=True)

            save_path = save_dir / f'alpha_log_{label_name.lower().replace(" ", "_")}.svg'
            save_figure(fig, save_path)


def plot_z_distribution_over_time(df: pd.DataFrame, tracks_data: Dict[Tuple[Path, float], pd.DataFrame],
                                   save_path: Path, progress_callback=None):
    """
    Plot z-position distribution evolution over time.
    Shows how particles distribute in z-axis over polymerization time.

    Args:
        df: Combined RF results dataframe
        tracks_data: Dictionary mapping (folder, time) -> tracks dataframe
        save_path: Output path for plot
        progress_callback: Optional progress callback
    """
    if progress_callback:
        progress_callback("Creating z-distribution over time plot...")

    # Load track data and extract z positions
    z_data_by_time = {}

    for (folder, poly_time), tracks_df in tracks_data.items():
        if 'z [nm]' in tracks_df.columns:
            z_col = 'z [nm]'
        elif 'z' in tracks_df.columns:
            z_col = 'z'
        else:
            continue

        z_values = tracks_df[z_col].dropna().values
        if len(z_values) > 0:
            z_data_by_time[poly_time] = z_values

    if not z_data_by_time:
        if progress_callback:
            progress_callback("  Warning: No z-position data available")
        return

    # Sort by time
    times = sorted(z_data_by_time.keys())
    z_data_list = [z_data_by_time[t] for t in times]

    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 6))

    parts = ax.violinplot(z_data_list, positions=times, widths=np.diff(times).min() * 0.8 if len(times) > 1 else 1,
                          showmeans=True, showextrema=True)

    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    format_axis(ax,
                xlabel='Polymerization Time',
                ylabel='z-Position (nm)',
                legend=False)

    save_figure(fig, save_path)


def plot_z_distribution_heatmap(df: pd.DataFrame, tracks_data: Dict[Tuple[Path, float], pd.DataFrame],
                                 save_path: Path, progress_callback=None):
    """
    Plot 2D heatmap of z-position distribution over time.

    Args:
        df: Combined RF results dataframe
        tracks_data: Dictionary mapping (folder, time) -> tracks dataframe
        save_path: Output path for plot
        progress_callback: Optional progress callback
    """
    if progress_callback:
        progress_callback("Creating z-distribution heatmap...")

    # Extract z positions by time
    z_data_by_time = {}

    for (folder, poly_time), tracks_df in tracks_data.items():
        if 'z [nm]' in tracks_df.columns:
            z_col = 'z [nm]'
        elif 'z' in tracks_df.columns:
            z_col = 'z'
        else:
            continue

        z_values = tracks_df[z_col].dropna().values
        if len(z_values) > 0:
            z_data_by_time[poly_time] = z_values

    if not z_data_by_time:
        if progress_callback:
            progress_callback("  Warning: No z-position data available for heatmap")
        return

    # Create 2D histogram
    times = sorted(z_data_by_time.keys())

    # Determine z-range
    all_z = np.concatenate([z_data_by_time[t] for t in times])
    z_min, z_max = np.percentile(all_z, [1, 99])

    # Create bins
    n_z_bins = 50
    n_time_bins = len(times)

    z_bins = np.linspace(z_min, z_max, n_z_bins + 1)

    # Build 2D histogram
    hist_2d = np.zeros((n_z_bins, n_time_bins))

    for i, t in enumerate(times):
        z_values = z_data_by_time[t]
        hist, _ = np.histogram(z_values, bins=z_bins)
        hist_2d[:, i] = hist / hist.sum() if hist.sum() > 0 else hist

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    extent = [times[0], times[-1], z_min, z_max]
    im = ax.imshow(hist_2d, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', interpolation='bilinear')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', rotation=270, labelpad=15)
    cbar.outline.set_visible(False)

    format_axis(ax,
                xlabel='Polymerization Time',
                ylabel='z-Position (nm)',
                legend=False)

    save_figure(fig, save_path)
