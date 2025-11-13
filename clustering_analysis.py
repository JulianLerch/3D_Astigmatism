"""
CLUSTERING ANALYSIS FOR TRACKED PARTICLES
Feature-based clustering using same features as RF classifier
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

# Import feature extractor from rf_analysis
try:
    from rf_analysis import DiffusionFeatureExtractor, calculate_diffusion_coefficient
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False

# Scikit-learn imports
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def extract_features_from_tracks(tracks: pd.DataFrame, dt: float = 0.1,
                                 progress_callback=None) -> pd.DataFrame:
    """
    Extract all 18 features from tracks for clustering.

    Args:
        tracks: DataFrame with columns ['particle', 'frame', 'x', 'y', 'z']
        dt: Integration time in seconds
        progress_callback: Optional progress callback

    Returns:
        DataFrame with features per track
    """
    if not FEATURES_AVAILABLE:
        raise ImportError("rf_analysis module not available - cannot extract features")

    if progress_callback:
        progress_callback("Extracting features for clustering...")

    particle_ids = sorted(tracks['particle'].unique())
    feature_list = []

    for idx, pid in enumerate(particle_ids, 1):
        track = tracks[tracks['particle'] == pid].copy()

        if len(track) < 10:
            # Skip very short tracks
            continue

        try:
            # Convert nm to μm
            pos_cols = ['x', 'y'] + (['z'] if 'z' in track.columns else [])
            trajectory_um = track[pos_cols].values / 1000.0

            # Extract features
            extractor = DiffusionFeatureExtractor(trajectory_um, dt=dt)
            features = extractor.extract_all_features()

            # Add metadata
            features['particle_id'] = int(pid)
            features['track_length'] = len(track)

            # Calculate D and alpha
            D, alpha = calculate_diffusion_coefficient(trajectory_um, dt=dt)
            features['D'] = D
            features['alpha_msd'] = alpha  # From MSD fit (may differ from feature alpha)

            feature_list.append(features)

            if progress_callback and (idx % 10 == 0 or idx == len(particle_ids)):
                progress_callback(f"  Extracted features from {idx}/{len(particle_ids)} tracks")

        except Exception as e:
            if progress_callback:
                progress_callback(f"  Warning: Could not extract features from track {pid}: {e}")
            continue

    if not feature_list:
        if progress_callback:
            progress_callback("ERROR: No features extracted!")
        return pd.DataFrame()

    df = pd.DataFrame(feature_list)

    if progress_callback:
        progress_callback(f"Feature extraction complete: {len(df)} tracks with {len(df.columns)} features")

    return df


def determine_optimal_clusters(X_scaled: np.ndarray, max_k: int = 10,
                               progress_callback=None) -> int:
    """
    Determine optimal number of clusters using elbow method and silhouette score.

    Args:
        X_scaled: Scaled feature matrix
        max_k: Maximum number of clusters to test
        progress_callback: Optional progress callback

    Returns:
        Optimal number of clusters
    """
    if progress_callback:
        progress_callback("Determining optimal number of clusters...")

    max_k = min(max_k, len(X_scaled) // 2)

    if max_k < 2:
        return 2

    inertias = []
    silhouettes = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)

        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
            sil = silhouette_score(X_scaled, labels)
            silhouettes.append(sil)
        else:
            silhouettes.append(-1)

    # Find elbow (maximum curvature)
    if len(inertias) >= 3:
        # Simple elbow detection: find point with max distance to line
        x = np.arange(len(inertias))
        y = np.array(inertias)

        # Line from first to last point
        line_start = np.array([x[0], y[0]])
        line_end = np.array([x[-1], y[-1]])

        # Distance from each point to line
        distances = []
        for i in range(len(x)):
            point = np.array([x[i], y[i]])
            dist = np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
            distances.append(dist)

        elbow_idx = np.argmax(distances)
        k_elbow = elbow_idx + 2  # +2 because we start from k=2
    else:
        k_elbow = 3

    # Find best silhouette
    k_silhouette = np.argmax(silhouettes) + 2

    # Choose based on both metrics (weighted average)
    k_optimal = int(np.round((k_elbow + k_silhouette) / 2))

    if progress_callback:
        progress_callback(f"  Elbow method suggests: {k_elbow} clusters")
        progress_callback(f"  Silhouette score suggests: {k_silhouette} clusters")
        progress_callback(f"  Using: {k_optimal} clusters")

    return k_optimal


def perform_clustering(features_df: pd.DataFrame, method: str = 'kmeans',
                      n_clusters: int = None, progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform clustering on features.

    Args:
        features_df: DataFrame with features
        method: 'kmeans', 'hierarchical', or 'dbscan'
        n_clusters: Number of clusters (None for automatic)
        progress_callback: Optional progress callback

    Returns:
        (features_df with cluster labels, clustering_info dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not available - cannot perform clustering")

    if progress_callback:
        progress_callback(f"Starting clustering with method: {method}")

    # Select feature columns (exclude metadata)
    feature_cols = [
        'alpha', 'msd_ratio', 'hurst_exponent', 'vacf_lag1', 'vacf_min',
        'kurtosis', 'straightness', 'mean_cos_theta', 'persistence_length',
        'efficiency', 'rg_saturation', 'asphericity', 'fractal_dimension',
        'convex_hull_area', 'confinement_probability', 'msd_plateauness',
        'space_exploration_ratio', 'boundary_proximity_var'
    ]

    # Handle missing features
    available_features = [col for col in feature_cols if col in features_df.columns]

    if not available_features:
        raise ValueError("No features available for clustering")

    X = features_df[available_features].values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal clusters if not specified
    if n_clusters is None and method != 'dbscan':
        n_clusters = determine_optimal_clusters(X_scaled, max_k=10, progress_callback=progress_callback)

    # Perform clustering
    if method == 'kmeans':
        if progress_callback:
            progress_callback(f"Running K-Means with {n_clusters} clusters...")

        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        davies_bouldin = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else -1

        info = {
            'method': 'K-Means',
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'inertia': float(clusterer.inertia_)
        }

    elif method == 'hierarchical':
        if progress_callback:
            progress_callback(f"Running Hierarchical Clustering with {n_clusters} clusters...")

        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clusterer.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        davies_bouldin = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else -1

        info = {
            'method': 'Hierarchical (Ward)',
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin)
        }

    elif method == 'dbscan':
        if progress_callback:
            progress_callback("Running DBSCAN clustering...")

        # Estimate eps using k-distance
        from sklearn.neighbors import NearestNeighbors
        k = min(10, len(X_scaled) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        eps = np.percentile(distances[:, -1], 90)

        clusterer = DBSCAN(eps=eps, min_samples=5)
        labels = clusterer.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 and -1 not in labels else -1
        davies_bouldin = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 and -1 not in labels else -1

        info = {
            'method': 'DBSCAN',
            'n_clusters': n_clusters,
            'n_noise': int(np.sum(labels == -1)),
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'eps': float(eps)
        }

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Add cluster labels to dataframe
    result_df = features_df.copy()
    result_df['cluster'] = labels

    if progress_callback:
        progress_callback(f"Clustering complete: {n_clusters} clusters found")
        progress_callback(f"  Silhouette score: {info['silhouette_score']:.3f}")
        progress_callback(f"  Davies-Bouldin score: {info['davies_bouldin_score']:.3f}")

    return result_df, info


def plot_track_by_cluster(track_data: pd.DataFrame, cluster_label: int,
                          save_path: Path, color_map: Dict[int, str]):
    """
    Plot 3D track colored by cluster.

    Args:
        track_data: DataFrame with x, y, z in nm
        cluster_label: Cluster label for this track
        save_path: Path to save SVG
        color_map: Dict mapping cluster -> color
    """
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = track_data['x'].values
    y = track_data['y'].values
    z = track_data['z'].values if 'z' in track_data.columns else np.zeros_like(x)

    # Get cluster color
    color = color_map.get(cluster_label, '#888888')

    # Plot track
    ax.plot(x, y, z, color=color, linewidth=2, alpha=0.8)

    # Start/end markers
    ax.scatter(x[0], y[0], z[0], color='green', s=80, marker='o',
               edgecolors='white', linewidths=2, zorder=10, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='black', s=80, marker='o',
               edgecolors='white', linewidths=2, zorder=10, label='End')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=f'Cluster {cluster_label}'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='green', markersize=8, label='Start'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='black', markersize=8, label='End')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    ax.set_xlabel('x / nm')
    ax.set_ylabel('y / nm')
    ax.set_zlabel('z / nm')

    # Set equal axes
    mins = np.array([x.min(), y.min(), z.min()])
    maxs = np.array([x.max(), y.max(), z.max()])
    centers = (mins + maxs) / 2
    half = (maxs - mins).max() / 2
    if not np.isfinite(half) or half == 0:
        half = 1.0
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def create_cluster_colormap(n_clusters: int) -> Dict[int, str]:
    """
    Create colormap for clusters.

    Args:
        n_clusters: Number of clusters

    Returns:
        Dict mapping cluster_id -> color (hex)
    """
    # Use tab20 for up to 20 clusters, otherwise generate
    if n_clusters <= 20:
        cmap = plt.cm.tab20
        colors = [matplotlib.colors.rgb2hex(cmap(i / 20)) for i in range(n_clusters)]
    else:
        cmap = plt.cm.hsv
        colors = [matplotlib.colors.rgb2hex(cmap(i / n_clusters)) for i in range(n_clusters)]

    # Handle noise cluster (-1) for DBSCAN
    color_map = {i: colors[i] for i in range(n_clusters)}
    color_map[-1] = '#808080'  # Gray for noise

    return color_map


def export_clustering_visualizations(tracks: pd.DataFrame, features_df: pd.DataFrame,
                                     output_dir: Path, n_tracks_to_plot='all',
                                     progress_callback=None):
    """
    Export clustered tracks as colored 3D plots.

    Args:
        tracks: Original tracks DataFrame
        features_df: Features with cluster labels
        output_dir: Output directory
        n_tracks_to_plot: 'all' or integer
        progress_callback: Optional progress callback
    """
    cluster_dir = output_dir / '09_Clustering_Classification'
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Creating clustering visualization plots...")

    # Get cluster info
    n_clusters = features_df['cluster'].nunique()
    color_map = create_cluster_colormap(n_clusters)

    # Select tracks to plot
    if n_tracks_to_plot == 'all':
        particle_ids = features_df['particle_id'].values
    else:
        # Select longest tracks
        sorted_by_length = features_df.sort_values('track_length', ascending=False)
        particle_ids = sorted_by_length.head(n_tracks_to_plot)['particle_id'].values

    total = len(particle_ids)

    for idx, pid in enumerate(particle_ids, 1):
        track = tracks[tracks['particle'] == pid].copy().sort_values('frame')

        # Get cluster label
        cluster_label = features_df[features_df['particle_id'] == pid]['cluster'].values[0]

        save_path = cluster_dir / f'Track_{int(pid):04d}_Cluster.svg'

        plot_track_by_cluster(track, cluster_label, save_path, color_map)

        if progress_callback and (idx % 5 == 0 or idx == total):
            progress_callback(f"  {idx}/{total} cluster plots created")

    if progress_callback:
        progress_callback(f"Clustering visualization complete → {cluster_dir}")


def create_clustering_boxplots(features_df: pd.DataFrame, save_dir: Path):
    """
    Create boxplots for alpha and D grouped by cluster.

    Args:
        features_df: DataFrame with cluster labels
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid values
    valid = features_df[['cluster', 'alpha_msd', 'D']].copy()
    valid = valid[np.isfinite(valid['alpha_msd']) & np.isfinite(valid['D'])]

    if len(valid) == 0:
        return

    # Get unique clusters (excluding noise)
    clusters_present = sorted([c for c in valid['cluster'].unique() if c != -1])

    if len(clusters_present) == 0:
        return

    # Create color map
    color_map = create_cluster_colormap(max(clusters_present) + 1)

    # Alpha boxplot
    fig, ax = plt.subplots(figsize=(8, 6))

    data_alpha = [valid[valid['cluster'] == c]['alpha_msd'].values for c in clusters_present]
    labels_str = [f'Cluster {c}' for c in clusters_present]

    bp = ax.boxplot(data_alpha, labels=labels_str, patch_artist=True)

    for patch, cluster in zip(bp['boxes'], clusters_present):
        patch.set_facecolor(color_map.get(cluster, '#888888'))
        patch.set_alpha(0.7)

    ax.set_ylabel('Alpha (Anomalous Exponent)', fontsize=12)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_title('Alpha Distribution by Cluster', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'alpha_boxplot_clusters.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    # D boxplot
    fig, ax = plt.subplots(figsize=(8, 6))

    data_D = [valid[valid['cluster'] == c]['D'].values for c in clusters_present]

    bp = ax.boxplot(data_D, labels=labels_str, patch_artist=True)

    for patch, cluster in zip(bp['boxes'], clusters_present):
        patch.set_facecolor(color_map.get(cluster, '#888888'))
        patch.set_alpha(0.7)

    ax.set_ylabel('D (Diffusion Coefficient) [μm²/s]', fontsize=12)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_title('Diffusion Coefficient Distribution by Cluster', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Log scale if needed
    if valid['D'].max() / (valid['D'].min() + 1e-10) > 100:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_dir / 'D_boxplot_clusters.svg', format='svg', bbox_inches='tight')
    plt.close(fig)


def create_pca_plot(features_df: pd.DataFrame, save_path: Path):
    """
    Create PCA plot colored by clusters.

    Args:
        features_df: DataFrame with features and cluster labels
        save_path: Path to save plot
    """
    # Select feature columns
    feature_cols = [
        'alpha', 'msd_ratio', 'hurst_exponent', 'vacf_lag1', 'vacf_min',
        'kurtosis', 'straightness', 'mean_cos_theta', 'persistence_length',
        'efficiency', 'rg_saturation', 'asphericity', 'fractal_dimension',
        'convex_hull_area', 'confinement_probability', 'msd_plateauness',
        'space_exploration_ratio', 'boundary_proximity_var'
    ]

    available_features = [col for col in feature_cols if col in features_df.columns]

    if len(available_features) < 2:
        return

    X = features_df[available_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    clusters = features_df['cluster'].values
    unique_clusters = sorted(set(clusters))

    color_map = create_cluster_colormap(max(unique_clusters) + 1)

    for cluster in unique_clusters:
        mask = clusters == cluster
        label = f'Cluster {cluster}' if cluster != -1 else 'Noise'
        color = color_map.get(cluster, '#888888')

        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, label=label, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_title('PCA of Track Features (Colored by Cluster)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def export_clustering_analysis(tracks: pd.DataFrame, features_df: pd.DataFrame,
                               clustering_info: Dict, output_dir: Path,
                               progress_callback=None):
    """
    Export comprehensive clustering analysis.

    Args:
        tracks: Original tracks
        features_df: Features with cluster labels
        clustering_info: Clustering metadata
        output_dir: Output directory
        progress_callback: Optional progress callback
    """
    analysis_dir = output_dir / '10_Clustering_Analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Creating clustering analysis exports...")

    # Cluster statistics
    cluster_stats = []
    for cluster in sorted(features_df['cluster'].unique()):
        cluster_data = features_df[features_df['cluster'] == cluster]

        stats = {
            'cluster': int(cluster),
            'n_tracks': len(cluster_data),
            'fraction': len(cluster_data) / len(features_df),
            'mean_track_length': float(cluster_data['track_length'].mean()),
            'std_track_length': float(cluster_data['track_length'].std()),
            'mean_alpha': float(cluster_data['alpha_msd'].mean()),
            'std_alpha': float(cluster_data['alpha_msd'].std()),
            'mean_D': float(cluster_data['D'].mean()),
            'std_D': float(cluster_data['D'].std()),
        }

        cluster_stats.append(stats)

    cluster_stats_df = pd.DataFrame(cluster_stats)

    # Export to Excel
    if progress_callback:
        progress_callback("  Writing Excel file...")

    excel_path = analysis_dir / 'clustering_analysis_complete.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Features with clusters
        features_df.to_excel(writer, sheet_name='Track_Features_Clusters', index=False)

        # Cluster statistics
        cluster_stats_df.to_excel(writer, sheet_name='Cluster_Statistics', index=False)

        # Clustering info
        pd.DataFrame([clustering_info]).to_excel(writer, sheet_name='Clustering_Info', index=False)

    if progress_callback:
        progress_callback(f"  Excel saved → {excel_path.name}")

    # Export CSVs
    if progress_callback:
        progress_callback("  Writing CSV files...")

    features_df.to_csv(analysis_dir / 'track_features_clusters.csv', index=False)
    cluster_stats_df.to_csv(analysis_dir / 'cluster_statistics.csv', index=False)

    # Create plots
    if progress_callback:
        progress_callback("  Creating boxplots...")

    create_clustering_boxplots(features_df, analysis_dir)

    if progress_callback:
        progress_callback("  Creating PCA plot...")

    create_pca_plot(features_df, analysis_dir / 'pca_clusters.svg')

    # Create cluster distribution plot
    if progress_callback:
        progress_callback("  Creating cluster distribution plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    cluster_counts = features_df['cluster'].value_counts().sort_index()
    clusters_present = cluster_counts.index.tolist()
    counts = cluster_counts.values

    color_map = create_cluster_colormap(max(clusters_present) + 1)
    bar_colors = [color_map.get(c, '#888888') for c in clusters_present]

    x_labels = [f'Cluster {c}' if c != -1 else 'Noise' for c in clusters_present]

    ax.bar(x_labels, counts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Number of Tracks', fontsize=12)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_title('Track Distribution by Cluster', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for i, (label, count) in enumerate(zip(x_labels, counts)):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(analysis_dir / 'cluster_distribution.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    if progress_callback:
        progress_callback(f"Clustering analysis complete → {analysis_dir}")


def perform_clustering_workflow(tracks: pd.DataFrame, output_dir: Path,
                                method: str = 'kmeans', n_clusters: int = None,
                                n_tracks_to_plot='all', dt: float = 0.1,
                                progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
    """
    Full clustering workflow.

    Args:
        tracks: Tracks DataFrame
        output_dir: Output directory
        method: Clustering method
        n_clusters: Number of clusters (None for automatic)
        n_tracks_to_plot: Number of tracks to plot
        dt: Integration time in seconds
        progress_callback: Optional progress callback

    Returns:
        (features_df with clusters, clustering_info)
    """
    if progress_callback:
        progress_callback("=" * 70)
        progress_callback("STARTING CLUSTERING ANALYSIS")

    # Extract features
    features_df = extract_features_from_tracks(tracks, dt=dt, progress_callback=progress_callback)

    if features_df.empty:
        if progress_callback:
            progress_callback("ERROR: No features extracted - aborting")
        return pd.DataFrame(), {}

    # Perform clustering
    features_df, clustering_info = perform_clustering(
        features_df, method=method, n_clusters=n_clusters,
        progress_callback=progress_callback
    )

    # Export visualizations
    export_clustering_visualizations(
        tracks=tracks,
        features_df=features_df,
        output_dir=output_dir,
        n_tracks_to_plot=n_tracks_to_plot,
        progress_callback=progress_callback
    )

    # Export analysis
    export_clustering_analysis(
        tracks=tracks,
        features_df=features_df,
        clustering_info=clustering_info,
        output_dir=output_dir,
        progress_callback=progress_callback
    )

    if progress_callback:
        progress_callback("=" * 70)
        progress_callback("CLUSTERING ANALYSIS COMPLETE!")
        progress_callback("  09_Clustering_Classification/ (SVG plots by cluster)")
        progress_callback("  10_Clustering_Analysis/ (Excel, CSV, PCA, boxplots)")

    return features_df, clustering_info
