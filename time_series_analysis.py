"""
TIME SERIES ANALYSIS FOR RF-CLASSIFIED TRACKS
Analyze how diffusion types evolve over polymerization time
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

# Import new plotting functions
try:
    from time_series_plots import (
        plot_diffusion_fractions_area,
        plot_D_boxplots_per_diffusion_type,
        plot_D_log_linear_per_diffusion_type,
        plot_alpha_log_linear_per_diffusion_type,
        plot_z_distribution_over_time,
        plot_z_distribution_heatmap
    )
    ADVANCED_PLOTS_AVAILABLE = True
except ImportError:
    ADVANCED_PLOTS_AVAILABLE = False
    logger.warning("Advanced plotting functions not available")


def load_rf_results_from_folder(folder: Path) -> Optional[pd.DataFrame]:
    """
    Load RF analysis results from a folder.

    Args:
        folder: Path to folder containing 3D_Tracking_Results/08_RF_Analysis/

    Returns:
        DataFrame with track summary or None if not found
    """
    rf_analysis_dir = folder / '3D_Tracking_Results' / '08_RF_Analysis'
    summary_file = rf_analysis_dir / 'track_summary.csv'

    if not summary_file.exists():
        return None

    try:
        df = pd.read_csv(summary_file)
        return df
    except Exception as e:
        print(f"Error loading {summary_file}: {e}")
        return None


def load_clustering_results_from_folder(folder: Path) -> Optional[pd.DataFrame]:
    """
    Load clustering analysis results from a folder.

    Args:
        folder: Path to folder containing 3D_Tracking_Results/10_Clustering_Analysis/

    Returns:
        DataFrame with clustering results or None if not found
    """
    cluster_analysis_dir = folder / '3D_Tracking_Results' / '10_Clustering_Analysis'
    summary_file = cluster_analysis_dir / 'track_features_clusters.csv'

    if not summary_file.exists():
        return None

    try:
        df = pd.read_csv(summary_file)
        return df
    except Exception as e:
        print(f"Error loading {summary_file}: {e}")
        return None


def load_tracks_from_folder(folder: Path) -> Optional[pd.DataFrame]:
    """
    Load tracked particles data from a folder (for z-position analysis).

    Args:
        folder: Path to folder containing tracking results

    Returns:
        DataFrame with tracks or None if not found
    """
    # Try multiple possible locations
    possible_paths = [
        folder / '3D_Tracking_Results' / '04_Tracks' / 'all_trajectories.csv',
        folder / '3D_Tracking_Results' / 'tracks.csv',
        folder / 'tracks.csv'
    ]

    for track_path in possible_paths:
        if track_path.exists():
            try:
                df = pd.read_csv(track_path)
                return df
            except Exception as e:
                logger.warning(f"Error loading {track_path}: {e}")
                continue

    return None


def aggregate_time_series_data(folders_with_times: List[Tuple[Path, float]],
                                progress_callback=None) -> pd.DataFrame:
    """
    Aggregate RF results from multiple folders with time points.

    Args:
        folders_with_times: List of (folder_path, polymerization_time) tuples
        progress_callback: Optional progress callback

    Returns:
        DataFrame with columns ['time', 'track_id', 'length', 'dominant_label', ...]
    """
    if progress_callback:
        progress_callback("Loading RF results from all folders...")

    all_data = []

    for folder, poly_time in folders_with_times:
        if progress_callback:
            progress_callback(f"  Loading {folder.name} (t={poly_time})")

        df = load_rf_results_from_folder(folder)

        if df is None:
            if progress_callback:
                progress_callback(f"    No RF results found - skipping")
            continue

        # Add time column
        df['polymerization_time'] = poly_time
        df['folder'] = str(folder.name)

        all_data.append(df)

    if not all_data:
        if progress_callback:
            progress_callback("ERROR: No RF data found in any folder!")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    if progress_callback:
        progress_callback(f"Loaded {len(combined)} tracks from {len(folders_with_times)} timepoints")

    return combined


def calculate_time_series_statistics(df: pd.DataFrame, label_names: Dict[int, str]) -> pd.DataFrame:
    """
    Calculate statistics per time point.

    Args:
        df: Combined DataFrame with 'polymerization_time' column
        label_names: Dict mapping label -> name

    Returns:
        DataFrame with time series statistics
    """
    stats = []

    time_points = sorted(df['polymerization_time'].unique())

    for t in time_points:
        t_data = df[df['polymerization_time'] == t]

        # Overall stats
        stat = {
            'time': float(t),
            'n_tracks': len(t_data),
            'mean_track_length': float(t_data['length'].mean()),
            'std_track_length': float(t_data['length'].std()),
            'mean_alpha_all': float(t_data['alpha'].mean()),
            'std_alpha_all': float(t_data['alpha'].std()),
            'mean_D_all': float(t_data['D'].mean()),
            'std_D_all': float(t_data['D'].std()),
        }

        # Per-label stats
        for label in sorted(df['dominant_label'].unique()):
            label_data = t_data[t_data['dominant_label'] == label]
            label_name = label_names.get(label, f'Type_{label}')

            stat[f'n_tracks_{label_name}'] = len(label_data)
            stat[f'fraction_{label_name}'] = len(label_data) / len(t_data) if len(t_data) > 0 else 0.0

            if len(label_data) > 0:
                stat[f'mean_alpha_{label_name}'] = float(label_data['alpha'].mean())
                stat[f'std_alpha_{label_name}'] = float(label_data['alpha'].std())
                stat[f'mean_D_{label_name}'] = float(label_data['D'].mean())
                stat[f'std_D_{label_name}'] = float(label_data['D'].std())
            else:
                stat[f'mean_alpha_{label_name}'] = np.nan
                stat[f'std_alpha_{label_name}'] = np.nan
                stat[f'mean_D_{label_name}'] = np.nan
                stat[f'std_D_{label_name}'] = np.nan

        stats.append(stat)

    return pd.DataFrame(stats)


def plot_time_series_track_counts(stats_df: pd.DataFrame, save_path: Path,
                                   label_names: Dict[int, str]):
    """
    Plot track counts over time (stacked area plot).

    Args:
        stats_df: Statistics DataFrame
        save_path: Path to save plot
        label_names: Dict mapping label -> name
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = stats_df['time'].values

    # Color scheme
    colors = {
        'Normal Diffusion': '#2E86AB',
        'Subdiffusion (fBm)': '#A23B72',
        'Confined Diffusion': '#F18F01',
        'Superdiffusion': '#C73E1D'
    }

    # Prepare data for stacking
    label_cols = [col for col in stats_df.columns if col.startswith('n_tracks_') and col != 'n_tracks']

    if not label_cols:
        return

    # Extract label names from column names
    plot_data = {}
    for col in label_cols:
        label_name = col.replace('n_tracks_', '')
        plot_data[label_name] = stats_df[col].values

    # Stack plot
    bottom = np.zeros(len(times))
    for label_name in sorted(plot_data.keys()):
        counts = plot_data[label_name]
        color = colors.get(label_name, '#888888')

        ax.fill_between(times, bottom, bottom + counts,
                         label=label_name, color=color, alpha=0.7)
        bottom += counts

    ax.set_xlabel('Polymerization Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Tracks', fontsize=12, fontweight='bold')
    ax.set_title('Track Count Evolution by Diffusion Type', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_time_series_fractions(stats_df: pd.DataFrame, save_path: Path,
                                label_names: Dict[int, str]):
    """
    Plot fraction of each diffusion type over time (100% stacked area).

    Args:
        stats_df: Statistics DataFrame
        save_path: Path to save plot
        label_names: Dict mapping label -> name
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = stats_df['time'].values

    # Color scheme
    colors = {
        'Normal Diffusion': '#2E86AB',
        'Subdiffusion (fBm)': '#A23B72',
        'Confined Diffusion': '#F18F01',
        'Superdiffusion': '#C73E1D'
    }

    # Prepare data for stacking
    fraction_cols = [col for col in stats_df.columns if col.startswith('fraction_')]

    if not fraction_cols:
        return

    # Extract fractions
    plot_data = {}
    for col in fraction_cols:
        label_name = col.replace('fraction_', '')
        plot_data[label_name] = stats_df[col].values

    # Stack plot (100%)
    bottom = np.zeros(len(times))
    for label_name in sorted(plot_data.keys()):
        fractions = plot_data[label_name] * 100  # Convert to percentage
        color = colors.get(label_name, '#888888')

        ax.fill_between(times, bottom, bottom + fractions,
                         label=label_name, color=color, alpha=0.7)
        bottom += fractions

    ax.set_xlabel('Polymerization Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Tracks (%)', fontsize=12, fontweight='bold')
    ax.set_title('Diffusion Type Distribution Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_time_series_alpha_evolution(stats_df: pd.DataFrame, save_path: Path,
                                      label_names: Dict[int, str]):
    """
    Plot alpha evolution over time per diffusion type.

    Args:
        stats_df: Statistics DataFrame
        save_path: Path to save plot
        label_names: Dict mapping label -> name
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = stats_df['time'].values

    # Color scheme
    colors = {
        'Normal Diffusion': '#2E86AB',
        'Subdiffusion (fBm)': '#A23B72',
        'Confined Diffusion': '#F18F01',
        'Superdiffusion': '#C73E1D'
    }

    # Plot mean alpha for each type
    alpha_cols = [col for col in stats_df.columns if col.startswith('mean_alpha_') and col != 'mean_alpha_all']

    for col in alpha_cols:
        label_name = col.replace('mean_alpha_', '')
        std_col = col.replace('mean_alpha_', 'std_alpha_')

        mean_alpha = stats_df[col].values
        std_alpha = stats_df[std_col].values if std_col in stats_df.columns else None

        color = colors.get(label_name, '#888888')

        # Plot with error band
        ax.plot(times, mean_alpha, label=label_name, color=color,
                linewidth=2.5, marker='o', markersize=6)

        if std_alpha is not None:
            ax.fill_between(times, mean_alpha - std_alpha, mean_alpha + std_alpha,
                             color=color, alpha=0.2)

    ax.set_xlabel('Polymerization Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha (Anomalous Exponent)', fontsize=12, fontweight='bold')
    ax.set_title('Alpha Evolution by Diffusion Type', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Normal (α=1)')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_time_series_D_evolution(stats_df: pd.DataFrame, save_path: Path,
                                  label_names: Dict[int, str]):
    """
    Plot diffusion coefficient evolution over time per diffusion type.

    Args:
        stats_df: Statistics DataFrame
        save_path: Path to save plot
        label_names: Dict mapping label -> name
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = stats_df['time'].values

    # Color scheme
    colors = {
        'Normal Diffusion': '#2E86AB',
        'Subdiffusion (fBm)': '#A23B72',
        'Confined Diffusion': '#F18F01',
        'Superdiffusion': '#C73E1D'
    }

    # Plot mean D for each type
    D_cols = [col for col in stats_df.columns if col.startswith('mean_D_') and col != 'mean_D_all']

    for col in D_cols:
        label_name = col.replace('mean_D_', '')
        std_col = col.replace('mean_D_', 'std_D_')

        mean_D = stats_df[col].values
        std_D = stats_df[std_col].values if std_col in stats_df.columns else None

        color = colors.get(label_name, '#888888')

        # Plot with error band
        ax.plot(times, mean_D, label=label_name, color=color,
                linewidth=2.5, marker='o', markersize=6)

        if std_D is not None:
            ax.fill_between(times, mean_D - std_D, mean_D + std_D,
                             color=color, alpha=0.2)

    ax.set_xlabel('Polymerization Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('D (Diffusion Coefficient) [μm²/s]', fontsize=12, fontweight='bold')
    ax.set_title('Diffusion Coefficient Evolution by Type', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    # Log scale if needed
    if stats_df[[col for col in D_cols]].max().max() / (stats_df[[col for col in D_cols]].min().min() + 1e-10) > 100:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_time_series_heatmap(df: pd.DataFrame, save_path: Path,
                              label_names: Dict[int, str]):
    """
    Create heatmap showing distribution of diffusion types over time.

    Args:
        df: Combined DataFrame with all tracks
        save_path: Path to save plot
        label_names: Dict mapping label -> name
    """
    # Create pivot table: time x diffusion type
    time_points = sorted(df['polymerization_time'].unique())
    labels = sorted(df['dominant_label'].unique())

    # Count matrix
    matrix = np.zeros((len(labels), len(time_points)))

    for i, label in enumerate(labels):
        for j, t in enumerate(time_points):
            count = len(df[(df['polymerization_time'] == t) & (df['dominant_label'] == label)])
            matrix[i, j] = count

    # Normalize to fractions
    totals = matrix.sum(axis=0)
    matrix_norm = matrix / (totals + 1e-10) * 100  # Percentage

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(time_points)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels([f'{t:.1f}' for t in time_points])
    ax.set_yticklabels([label_names.get(label, f'Type {label}') for label in labels])

    ax.set_xlabel('Polymerization Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Diffusion Type', fontsize=12, fontweight='bold')
    ax.set_title('Diffusion Type Distribution Heatmap (%)', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction (%)', fontsize=11)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(time_points)):
            text = ax.text(j, i, f'{matrix_norm[i, j]:.1f}',
                           ha="center", va="center", color="black" if matrix_norm[i, j] < 50 else "white",
                           fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_overall_time_trends(stats_df: pd.DataFrame, save_path: Path):
    """
    Plot overall trends (all diffusion types combined).

    Args:
        stats_df: Statistics DataFrame
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times = stats_df['time'].values

    # Track count
    ax = axes[0, 0]
    ax.plot(times, stats_df['n_tracks'].values, color='#2E86AB',
            linewidth=3, marker='o', markersize=8)
    ax.set_xlabel('Polymerization Time', fontweight='bold')
    ax.set_ylabel('Total Track Count', fontweight='bold')
    ax.set_title('Total Tracks Over Time', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Mean track length
    ax = axes[0, 1]
    mean_len = stats_df['mean_track_length'].values
    std_len = stats_df['std_track_length'].values
    ax.plot(times, mean_len, color='#F18F01', linewidth=3, marker='o', markersize=8)
    ax.fill_between(times, mean_len - std_len, mean_len + std_len,
                     color='#F18F01', alpha=0.2)
    ax.set_xlabel('Polymerization Time', fontweight='bold')
    ax.set_ylabel('Track Length (frames)', fontweight='bold')
    ax.set_title('Mean Track Length Over Time', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Mean alpha (all)
    ax = axes[1, 0]
    mean_alpha = stats_df['mean_alpha_all'].values
    std_alpha = stats_df['std_alpha_all'].values
    ax.plot(times, mean_alpha, color='#A23B72', linewidth=3, marker='o', markersize=8)
    ax.fill_between(times, mean_alpha - std_alpha, mean_alpha + std_alpha,
                     color='#A23B72', alpha=0.2)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Normal (α=1)')
    ax.set_xlabel('Polymerization Time', fontweight='bold')
    ax.set_ylabel('Alpha', fontweight='bold')
    ax.set_title('Mean Alpha Over Time (All Types)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Mean D (all)
    ax = axes[1, 1]
    mean_D = stats_df['mean_D_all'].values
    std_D = stats_df['std_D_all'].values
    ax.plot(times, mean_D, color='#C73E1D', linewidth=3, marker='o', markersize=8)
    ax.fill_between(times, mean_D - std_D, mean_D + std_D,
                     color='#C73E1D', alpha=0.2)
    ax.set_xlabel('Polymerization Time', fontweight='bold')
    ax.set_ylabel('D [μm²/s]', fontweight='bold')
    ax.set_title('Mean Diffusion Coefficient Over Time (All Types)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def export_clustering_time_series(folders_with_times: List[Tuple[Path, float]],
                                 ts_dir: Path, progress_callback=None):
    """
    Export clustering time series analysis.

    Args:
        folders_with_times: List of (folder_path, polymerization_time) tuples
        ts_dir: Time series output directory
        progress_callback: Optional progress callback
    """
    # Load clustering data
    all_cluster_data = []

    for folder, poly_time in folders_with_times:
        df = load_clustering_results_from_folder(folder)

        if df is None:
            continue

        df['polymerization_time'] = poly_time
        df['folder'] = str(folder.name)

        all_cluster_data.append(df)

    if not all_cluster_data:
        if progress_callback:
            progress_callback("  No clustering data found - skipping")
        return

    combined_cluster_df = pd.concat(all_cluster_data, ignore_index=True)

    # Statistics per time point
    cluster_stats = []

    time_points = sorted(combined_cluster_df['polymerization_time'].unique())

    for t in time_points:
        t_data = combined_cluster_df[combined_cluster_df['polymerization_time'] == t]

        # Overall stats
        stat = {
            'time': float(t),
            'n_tracks': len(t_data),
            'n_clusters': int(t_data['cluster'].nunique()),
            'mean_alpha': float(t_data['alpha_msd'].mean()),
            'std_alpha': float(t_data['alpha_msd'].std()),
            'mean_D': float(t_data['D'].mean()),
            'std_D': float(t_data['D'].std()),
        }

        # Per-cluster stats
        for cluster in sorted(t_data['cluster'].unique()):
            if cluster == -1:  # Skip noise
                continue

            cluster_data = t_data[t_data['cluster'] == cluster]

            stat[f'n_tracks_cluster_{cluster}'] = len(cluster_data)
            stat[f'fraction_cluster_{cluster}'] = len(cluster_data) / len(t_data)
            stat[f'mean_alpha_cluster_{cluster}'] = float(cluster_data['alpha_msd'].mean())
            stat[f'mean_D_cluster_{cluster}'] = float(cluster_data['D'].mean())

        cluster_stats.append(stat)

    cluster_stats_df = pd.DataFrame(cluster_stats)

    # Export CSV
    cluster_stats_df.to_csv(ts_dir / 'time_series_clustering_statistics.csv', index=False)
    combined_cluster_df.to_csv(ts_dir / 'time_series_all_tracks_clustering.csv', index=False)

    if progress_callback:
        progress_callback("  Saved clustering time series CSVs")

    # Create plots
    # 1. Cluster distribution heatmap over time
    try:
        from clustering_analysis import create_cluster_colormap

        fig, ax = plt.subplots(figsize=(12, 6))

        clusters_all = sorted([c for c in combined_cluster_df['cluster'].unique() if c != -1])
        times = cluster_stats_df['time'].values

        matrix = np.zeros((len(clusters_all), len(times)))

        for i, cluster in enumerate(clusters_all):
            for j, t in enumerate(times):
                t_data = combined_cluster_df[combined_cluster_df['polymerization_time'] == t]
                count = len(t_data[t_data['cluster'] == cluster])
                matrix[i, j] = count

        # Normalize to fractions
        totals = matrix.sum(axis=0)
        matrix_norm = matrix / (totals + 1e-10) * 100

        im = ax.imshow(matrix_norm, cmap='YlGnBu', aspect='auto', interpolation='nearest')

        ax.set_xticks(np.arange(len(times)))
        ax.set_yticks(np.arange(len(clusters_all)))

        ax.set_xticklabels([f'{t:.1f}' for t in times])
        ax.set_yticklabels([f'Cluster {c}' for c in clusters_all])

        ax.set_xlabel('Polymerization Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Distribution Heatmap Over Time (%)', fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fraction (%)', fontsize=11)

        # Annotate
        for i in range(len(clusters_all)):
            for j in range(len(times)):
                if matrix_norm[i, j] > 0.1:  # Only annotate non-negligible values
                    text = ax.text(j, i, f'{matrix_norm[i, j]:.1f}',
                                   ha="center", va="center",
                                   color="black" if matrix_norm[i, j] < 50 else "white",
                                   fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(ts_dir / 'clustering_heatmap_over_time.svg', format='svg', bbox_inches='tight', dpi=300)
        plt.close(fig)

        if progress_callback:
            progress_callback("  Created: clustering_heatmap_over_time.svg")

    except Exception as e:
        if progress_callback:
            progress_callback(f"  Warning: Could not create clustering heatmap: {e}")

    # 2. Cluster count evolution (stacked area)
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        color_map = create_cluster_colormap(max(clusters_all) + 1)

        bottom = np.zeros(len(times))
        for cluster in sorted(clusters_all):
            counts = []
            for t in times:
                t_data = combined_cluster_df[combined_cluster_df['polymerization_time'] == t]
                count = len(t_data[t_data['cluster'] == cluster])
                counts.append(count)

            counts = np.array(counts)
            color = color_map.get(cluster, '#888888')

            ax.fill_between(times, bottom, bottom + counts,
                             label=f'Cluster {cluster}', color=color, alpha=0.7)
            bottom += counts

        ax.set_xlabel('Polymerization Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Tracks', fontsize=12, fontweight='bold')
        ax.set_title('Track Count Evolution by Cluster', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(ts_dir / 'cluster_counts_over_time.svg', format='svg', bbox_inches='tight', dpi=300)
        plt.close(fig)

        if progress_callback:
            progress_callback("  Created: cluster_counts_over_time.svg")

    except Exception as e:
        if progress_callback:
            progress_callback(f"  Warning: Could not create cluster count plot: {e}")


def export_time_series_analysis(folders_with_times: List[Tuple[Path, float]],
                                output_dir: Path,
                                progress_callback=None):
    """
    Full time series analysis workflow.

    Args:
        folders_with_times: List of (folder_path, polymerization_time) tuples
        output_dir: Output directory for time series results
        progress_callback: Optional progress callback
    """
    ts_dir = output_dir / 'timeSeries'
    ts_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("=" * 70)
        progress_callback("STARTING TIME SERIES ANALYSIS")

    label_names = {
        0: 'Normal Diffusion',
        1: 'Subdiffusion (fBm)',
        2: 'Confined Diffusion',
        3: 'Superdiffusion'
    }

    # Load all data
    combined_df = aggregate_time_series_data(folders_with_times, progress_callback)

    if combined_df.empty:
        if progress_callback:
            progress_callback("ERROR: No data loaded - aborting time series analysis")
        return

    # Calculate statistics
    if progress_callback:
        progress_callback("Calculating time series statistics...")

    stats_df = calculate_time_series_statistics(combined_df, label_names)

    # Export statistics
    if progress_callback:
        progress_callback("Exporting statistics...")

    stats_df.to_csv(ts_dir / 'time_series_statistics.csv', index=False)
    combined_df.to_csv(ts_dir / 'time_series_all_tracks.csv', index=False)

    # Excel export
    with pd.ExcelWriter(ts_dir / 'time_series_analysis.xlsx', engine='openpyxl') as writer:
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        combined_df.to_excel(writer, sheet_name='All_Tracks', index=False)

    if progress_callback:
        progress_callback(f"  Saved: time_series_statistics.csv, time_series_all_tracks.csv, time_series_analysis.xlsx")

    # Create visualizations
    if progress_callback:
        progress_callback("Creating time series plots...")

    plot_time_series_track_counts(stats_df, ts_dir / 'track_counts_over_time.svg', label_names)
    if progress_callback:
        progress_callback("  Created: track_counts_over_time.svg")

    plot_time_series_fractions(stats_df, ts_dir / 'diffusion_fractions_over_time.svg', label_names)
    if progress_callback:
        progress_callback("  Created: diffusion_fractions_over_time.svg")

    plot_time_series_alpha_evolution(stats_df, ts_dir / 'alpha_evolution.svg', label_names)
    if progress_callback:
        progress_callback("  Created: alpha_evolution.svg")

    plot_time_series_D_evolution(stats_df, ts_dir / 'D_evolution.svg', label_names)
    if progress_callback:
        progress_callback("  Created: D_evolution.svg")

    plot_time_series_heatmap(combined_df, ts_dir / 'diffusion_heatmap.svg', label_names)
    if progress_callback:
        progress_callback("  Created: diffusion_heatmap.svg")

    plot_overall_time_trends(stats_df, ts_dir / 'overall_trends.svg')
    if progress_callback:
        progress_callback("  Created: overall_trends.svg")

    # Create advanced plots if available
    if ADVANCED_PLOTS_AVAILABLE:
        if progress_callback:
            progress_callback("Creating advanced plots...")

        # Stacked area chart for diffusion fractions
        plot_diffusion_fractions_area(combined_df, ts_dir / 'diffusion_fractions_area.svg', progress_callback)

        # Boxplots for D per diffusion type
        d_boxplot_dir = ts_dir / 'D_boxplots'
        plot_D_boxplots_per_diffusion_type(combined_df, d_boxplot_dir, progress_callback)

        # Log/linear plots for D
        d_plots_dir = ts_dir / 'D_plots'
        plot_D_log_linear_per_diffusion_type(combined_df, d_plots_dir, progress_callback)

        # Log/linear plots for alpha
        alpha_plots_dir = ts_dir / 'alpha_plots'
        plot_alpha_log_linear_per_diffusion_type(combined_df, alpha_plots_dir, progress_callback)

        # Load tracks for z-position analysis
        if progress_callback:
            progress_callback("Loading track data for z-position analysis...")

        tracks_data = {}
        for folder, poly_time in folders_with_times:
            tracks_df = load_tracks_from_folder(folder)
            if tracks_df is not None:
                tracks_data[(folder, poly_time)] = tracks_df

        if tracks_data:
            # z-distribution plots
            plot_z_distribution_over_time(combined_df, tracks_data,
                                          ts_dir / 'z_distribution_violin.svg', progress_callback)
            plot_z_distribution_heatmap(combined_df, tracks_data,
                                        ts_dir / 'z_distribution_heatmap.svg', progress_callback)
        else:
            if progress_callback:
                progress_callback("  Warning: No track data found for z-position analysis")

    # Check if clustering data is also available
    cluster_available = all(
        (folder / '3D_Tracking_Results' / '10_Clustering_Analysis' / 'track_features_clusters.csv').exists()
        for folder, _ in folders_with_times
    )

    if cluster_available and progress_callback:
        progress_callback("Clustering data detected - creating clustering time series...")

        export_clustering_time_series(folders_with_times, ts_dir, progress_callback)

    if progress_callback:
        progress_callback("=" * 70)
        progress_callback("TIME SERIES ANALYSIS COMPLETE!")
        progress_callback(f"Output directory: {ts_dir}")
        progress_callback(f"Time points analyzed: {len(folders_with_times)}")
        progress_callback(f"Total tracks: {len(combined_df)}")
        if cluster_available:
            progress_callback("Clustering time series included!")
        if ADVANCED_PLOTS_AVAILABLE:
            progress_callback("Advanced plots created (area, boxplots, log/linear, z-distribution)")
