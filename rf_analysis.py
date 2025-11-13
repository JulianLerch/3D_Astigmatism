"""
RF DIFFUSION CLASSIFIER - INTEGRATION
Feature extraction, sliding window analysis, voting, visualization, and export
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis as scipy_kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# Configure warnings and logging
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


class DiffusionFeatureExtractor:
    """
    Extracts all 18 features required by the Random Forest Diffusion Classifier.
    Handles 2D and 3D trajectories.
    """

    def __init__(self, trajectory: np.ndarray, dt: float = 0.1):
        """
        Args:
            trajectory: numpy array of shape (N, 2) or (N, 3) in micrometers
            dt: integration time in seconds (default: 0.1s = 100ms)
        """
        self.trajectory = np.asarray(trajectory, dtype=float)
        self.dt = dt
        self.N = len(self.trajectory)
        self.dim = self.trajectory.shape[1] if len(self.trajectory.shape) > 1 else 1

        if self.N < 3:
            raise ValueError(f"Trajectory too short: {self.N} frames (minimum 3)")

    def _msd(self, max_lag: Optional[int] = None) -> np.ndarray:
        """Calculate mean squared displacement."""
        if max_lag is None:
            max_lag = min(self.N // 4, 100)

        msd = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            if lag >= self.N:
                msd[lag - 1] = np.nan
                continue
            displacements = self.trajectory[lag:] - self.trajectory[:-lag]
            msd[lag - 1] = np.mean(np.sum(displacements**2, axis=1))

        return msd

    def extract_alpha(self) -> float:
        """Anomalous exponent from MSD power-law fit."""
        try:
            msd = self._msd()
            valid = np.isfinite(msd) & (msd > 0)
            if valid.sum() < 3:
                return 1.0

            lag = np.arange(1, len(msd) + 1)[valid]
            msd_valid = msd[valid]

            # Log-log fit: log(MSD) = log(4D) + alpha * log(t)
            log_lag = np.log(lag * self.dt)
            log_msd = np.log(msd_valid)

            coeffs = np.polyfit(log_lag, log_msd, 1)
            alpha = float(coeffs[0])

            # Clip to reasonable range
            return np.clip(alpha, 0.1, 2.5)
        except Exception:
            return 1.0

    def extract_msd_ratio(self) -> float:
        """MSD ratio R(4,1) - plateau formation indicator for confinement."""
        try:
            msd = self._msd(max_lag=min(20, self.N // 2))
            if len(msd) < 4 or not np.isfinite(msd[:4]).all():
                return 4.0

            ratio = msd[3] / (msd[0] + 1e-10)
            return float(np.clip(ratio, 0.1, 20.0))
        except Exception:
            return 4.0

    def extract_hurst_exponent(self) -> float:
        """Hurst exponent H = alpha/2 for fBm identification."""
        alpha = self.extract_alpha()
        return float(alpha / 2.0)

    def extract_vacf_lag1(self) -> float:
        """Velocity autocorrelation at lag 1."""
        try:
            if self.N < 3:
                return 0.0

            velocities = np.diff(self.trajectory, axis=0)
            if len(velocities) < 2:
                return 0.0

            v1 = velocities[:-1]
            v2 = velocities[1:]

            # Dot product and normalize
            dots = np.sum(v1 * v2, axis=1)
            norms = np.sqrt(np.sum(v1**2, axis=1) * np.sum(v2**2, axis=1))

            valid = norms > 1e-10
            if not valid.any():
                return 0.0

            vacf = dots[valid] / norms[valid]
            return float(np.mean(vacf))
        except Exception:
            return 0.0

    def extract_vacf_min(self) -> float:
        """Minimum of VACF (negative values indicate subdiffusion)."""
        try:
            if self.N < 10:
                return 0.0

            velocities = np.diff(self.trajectory, axis=0)
            max_lag = min(len(velocities) // 2, 20)

            vacf = []
            for lag in range(1, max_lag):
                if lag >= len(velocities):
                    break
                v1 = velocities[:-lag]
                v2 = velocities[lag:]
                dots = np.sum(v1 * v2, axis=1)
                norms = np.sqrt(np.sum(v1**2, axis=1) * np.sum(v2**2, axis=1))
                valid = norms > 1e-10
                if valid.any():
                    vacf.append(np.mean(dots[valid] / norms[valid]))

            if not vacf:
                return 0.0

            return float(np.min(vacf))
        except Exception:
            return 0.0

    def extract_kurtosis(self) -> float:
        """Excess kurtosis - Gaussianity indicator."""
        try:
            if self.N < 5:
                return 0.0

            displacements = np.diff(self.trajectory, axis=0)
            step_sizes = np.sqrt(np.sum(displacements**2, axis=1))

            if len(step_sizes) < 4:
                return 0.0

            kurt = scipy_kurtosis(step_sizes, fisher=True, nan_policy='omit')

            if not np.isfinite(kurt):
                return 0.0

            return float(np.clip(kurt, -5, 10))
        except Exception:
            return 0.0

    def extract_straightness(self) -> float:
        """Straightness index: net displacement / path length."""
        try:
            if self.N < 2:
                return 0.0

            net_displacement = np.linalg.norm(self.trajectory[-1] - self.trajectory[0])

            displacements = np.diff(self.trajectory, axis=0)
            path_length = np.sum(np.sqrt(np.sum(displacements**2, axis=1)))

            if path_length < 1e-10:
                return 0.0

            straightness = net_displacement / path_length
            return float(np.clip(straightness, 0.0, 1.0))
        except Exception:
            return 0.0

    def extract_mean_cos_theta(self) -> float:
        """Mean cosine of turning angles."""
        try:
            if self.N < 3:
                return 0.0

            velocities = np.diff(self.trajectory, axis=0)
            if len(velocities) < 2:
                return 0.0

            v1 = velocities[:-1]
            v2 = velocities[1:]

            dots = np.sum(v1 * v2, axis=1)
            norms = np.sqrt(np.sum(v1**2, axis=1) * np.sum(v2**2, axis=1))

            valid = norms > 1e-10
            if not valid.any():
                return 0.0

            cos_theta = dots[valid] / norms[valid]
            return float(np.mean(cos_theta))
        except Exception:
            return 0.0

    def extract_persistence_length(self) -> float:
        """Persistence length from velocity autocorrelation decay."""
        try:
            if self.N < 10:
                return 1.0

            velocities = np.diff(self.trajectory, axis=0)
            max_lag = min(len(velocities) // 2, 50)

            vacf = []
            for lag in range(1, max_lag):
                if lag >= len(velocities):
                    break
                v1 = velocities[:-lag]
                v2 = velocities[lag:]
                dots = np.sum(v1 * v2, axis=1)
                norms = np.sqrt(np.sum(v1**2, axis=1) * np.sum(v2**2, axis=1))
                valid = norms > 1e-10
                if valid.any():
                    vacf.append(np.mean(dots[valid] / norms[valid]))

            if len(vacf) < 3:
                return 1.0

            # Find where VACF drops to 1/e
            vacf_arr = np.array(vacf)
            threshold = 1.0 / np.e

            below = np.where(vacf_arr < threshold)[0]
            if len(below) > 0:
                persistence = float(below[0] + 1)
            else:
                persistence = float(len(vacf))

            return persistence
        except Exception:
            return 1.0

    def extract_efficiency(self) -> float:
        """Net displacement efficiency."""
        try:
            if self.N < 2:
                return 0.0

            net_disp_sq = np.sum((self.trajectory[-1] - self.trajectory[0])**2)

            msd = self._msd(max_lag=min(self.N - 1, 100))
            mean_msd = np.nanmean(msd)

            if mean_msd < 1e-10:
                return 0.0

            efficiency = net_disp_sq / (mean_msd + 1e-10)
            return float(np.clip(efficiency, 0.0, 10.0))
        except Exception:
            return 0.0

    def extract_rg_saturation(self) -> float:
        """Radius of gyration saturation (plateau indicator)."""
        try:
            if self.N < 10:
                return 0.0

            # Calculate Rg over trajectory
            center = np.mean(self.trajectory, axis=0)
            rg_values = []

            window = max(5, self.N // 10)
            for i in range(0, self.N - window, max(1, window // 2)):
                segment = self.trajectory[i:i+window]
                seg_center = np.mean(segment, axis=0)
                rg = np.sqrt(np.mean(np.sum((segment - seg_center)**2, axis=1)))
                rg_values.append(rg)

            if len(rg_values) < 3:
                return 0.0

            # Check for plateau (small variance in later half)
            rg_arr = np.array(rg_values)
            second_half = rg_arr[len(rg_arr)//2:]

            if len(second_half) < 2:
                return 0.0

            variance = np.var(second_half)
            mean_rg = np.mean(rg_arr)

            if mean_rg < 1e-10:
                return 0.0

            saturation = 1.0 - np.clip(variance / (mean_rg**2 + 1e-10), 0, 1)
            return float(saturation)
        except Exception:
            return 0.0

    def extract_asphericity(self) -> float:
        """Spatial asymmetry (→1: linear, →0: isotropic)."""
        try:
            if self.N < 3:
                return 0.0

            # Gyration tensor
            center = np.mean(self.trajectory, axis=0)
            centered = self.trajectory - center

            gyration_tensor = np.dot(centered.T, centered) / self.N

            eigenvalues = np.linalg.eigvalsh(gyration_tensor)
            eigenvalues = np.sort(eigenvalues)[::-1]  # descending

            if self.dim == 2:
                if eigenvalues[0] + eigenvalues[1] < 1e-10:
                    return 0.0
                asphericity = (eigenvalues[0] - eigenvalues[1])**2 / (2 * (eigenvalues[0] + eigenvalues[1])**2)
            else:  # 3D
                lambda_sum = eigenvalues.sum()
                if lambda_sum < 1e-10:
                    return 0.0
                asphericity = ((eigenvalues[0] - eigenvalues[1])**2 +
                               (eigenvalues[1] - eigenvalues[2])**2 +
                               (eigenvalues[0] - eigenvalues[2])**2) / (2 * lambda_sum**2)

            return float(np.clip(asphericity, 0.0, 1.0))
        except Exception:
            return 0.0

    def extract_fractal_dimension(self) -> float:
        """Fractal dimension (≈2: space-filling)."""
        try:
            if self.N < 10:
                return 2.0

            # Box-counting approximation via path length scaling
            scales = [2, 4, 8, 16]
            scales = [s for s in scales if s < self.N]

            if len(scales) < 2:
                return 2.0

            lengths = []
            for scale in scales:
                # Downsample trajectory
                indices = np.arange(0, self.N, scale)
                downsampled = self.trajectory[indices]

                if len(downsampled) < 2:
                    continue

                displacements = np.diff(downsampled, axis=0)
                length = np.sum(np.sqrt(np.sum(displacements**2, axis=1)))
                lengths.append(length)

            if len(lengths) < 2:
                return 2.0

            # Fractal dimension from slope
            log_scales = np.log(scales[:len(lengths)])
            log_lengths = np.log(np.array(lengths) + 1e-10)

            coeffs = np.polyfit(log_scales, log_lengths, 1)
            fractal_dim = 1.0 - coeffs[0]  # Inverted relationship

            return float(np.clip(fractal_dim, 1.0, 3.0))
        except Exception:
            return 2.0

    def extract_convex_hull_area(self) -> float:
        """Convex hull area/volume."""
        try:
            if self.N < 4:
                return 0.0

            if self.dim == 2:
                try:
                    hull = ConvexHull(self.trajectory)
                    return float(hull.volume)  # In 2D, volume = area
                except Exception:
                    return 0.0
            else:  # 3D
                try:
                    hull = ConvexHull(self.trajectory)
                    return float(hull.volume)
                except Exception:
                    # Fallback: use 2D projection
                    try:
                        hull_2d = ConvexHull(self.trajectory[:, :2])
                        return float(hull_2d.volume)
                    except Exception:
                        return 0.0
        except Exception:
            return 0.0

    def extract_confinement_probability(self) -> float:
        """Probability of confinement based on return frequency."""
        try:
            if self.N < 10:
                return 0.0

            # Calculate distance from starting point
            distances = np.sqrt(np.sum((self.trajectory - self.trajectory[0])**2, axis=1))

            # Find median distance
            median_dist = np.median(distances)

            if median_dist < 1e-10:
                return 1.0

            # Count returns within median distance
            returns = np.sum(distances < median_dist)

            confinement_prob = returns / self.N
            return float(np.clip(confinement_prob, 0.0, 1.0))
        except Exception:
            return 0.0

    def extract_msd_plateauness(self) -> float:
        """MSD plateau indicator."""
        try:
            msd = self._msd(max_lag=min(50, self.N // 3))

            if len(msd) < 10 or not np.isfinite(msd).any():
                return 0.0

            # Check if MSD slope decreases (plateau)
            mid_point = len(msd) // 2

            first_half = msd[:mid_point]
            second_half = msd[mid_point:]

            if len(first_half) < 3 or len(second_half) < 3:
                return 0.0

            # Linear fits
            x1 = np.arange(len(first_half))
            x2 = np.arange(len(second_half))

            valid1 = np.isfinite(first_half)
            valid2 = np.isfinite(second_half)

            if valid1.sum() < 2 or valid2.sum() < 2:
                return 0.0

            slope1 = np.polyfit(x1[valid1], first_half[valid1], 1)[0]
            slope2 = np.polyfit(x2[valid2], second_half[valid2], 1)[0]

            if slope1 < 1e-10:
                return 1.0

            plateauness = 1.0 - np.clip(slope2 / (slope1 + 1e-10), 0, 1)
            return float(plateauness)
        except Exception:
            return 0.0

    def extract_space_exploration_ratio(self) -> float:
        """Ratio of explored space to expected diffusive space."""
        try:
            if self.N < 5:
                return 1.0

            # Actual explored volume (bounding box or convex hull)
            ranges = np.ptp(self.trajectory, axis=0)
            explored_volume = np.prod(ranges)

            # Expected diffusive spread
            msd = self._msd()
            mean_msd = np.nanmean(msd)

            if mean_msd < 1e-10:
                return 0.0

            expected_spread = np.sqrt(mean_msd)
            expected_volume = expected_spread ** self.dim

            if expected_volume < 1e-10:
                return 1.0

            ratio = explored_volume / expected_volume
            return float(np.clip(ratio, 0.0, 10.0))
        except Exception:
            return 1.0

    def extract_boundary_proximity_var(self) -> float:
        """Variance in proximity to trajectory boundary."""
        try:
            if self.N < 10:
                return 0.0

            # Find boundary (min/max in each dimension)
            mins = np.min(self.trajectory, axis=0)
            maxs = np.max(self.trajectory, axis=0)

            # Distance to nearest boundary for each point
            distances_to_boundary = []
            for point in self.trajectory:
                dist_to_min = point - mins
                dist_to_max = maxs - point
                min_dist = np.min(np.concatenate([dist_to_min, dist_to_max]))
                distances_to_boundary.append(min_dist)

            if len(distances_to_boundary) < 2:
                return 0.0

            variance = np.var(distances_to_boundary)
            return float(variance)
        except Exception:
            return 0.0

    def extract_all_features(self) -> Dict[str, float]:
        """
        Extract all 18 features in the correct order for the RF model.
        Returns dict with feature names as keys.
        """
        features = {
            'alpha': self.extract_alpha(),
            'msd_ratio': self.extract_msd_ratio(),
            'hurst_exponent': self.extract_hurst_exponent(),
            'vacf_lag1': self.extract_vacf_lag1(),
            'vacf_min': self.extract_vacf_min(),
            'kurtosis': self.extract_kurtosis(),
            'straightness': self.extract_straightness(),
            'mean_cos_theta': self.extract_mean_cos_theta(),
            'persistence_length': self.extract_persistence_length(),
            'efficiency': self.extract_efficiency(),
            'rg_saturation': self.extract_rg_saturation(),
            'asphericity': self.extract_asphericity(),
            'fractal_dimension': self.extract_fractal_dimension(),
            'convex_hull_area': self.extract_convex_hull_area(),
            'confinement_probability': self.extract_confinement_probability(),
            'msd_plateauness': self.extract_msd_plateauness(),
            'space_exploration_ratio': self.extract_space_exploration_ratio(),
            'boundary_proximity_var': self.extract_boundary_proximity_var(),
        }

        return features


def calculate_diffusion_coefficient(trajectory: np.ndarray, dt: float = 0.1,
                                     alpha: Optional[float] = None) -> Tuple[float, float]:
    """
    Calculate diffusion coefficient D and alpha from MSD.

    Args:
        trajectory: numpy array (N, 2) or (N, 3) in micrometers
        dt: integration time in seconds
        alpha: if provided, use this alpha; otherwise calculate

    Returns:
        (D, alpha) where MSD = 2*d*D*t^alpha (d = dimensionality)
    """
    try:
        N = len(trajectory)
        dim = trajectory.shape[1] if len(trajectory.shape) > 1 else 1

        if N < 5:
            return (np.nan, np.nan)

        # Calculate MSD
        max_lag = min(N // 4, 50)
        msd = []
        for lag in range(1, max_lag + 1):
            if lag >= N:
                break
            displacements = trajectory[lag:] - trajectory[:-lag]
            msd.append(np.mean(np.sum(displacements**2, axis=1)))

        if len(msd) < 3:
            return (np.nan, np.nan)

        msd = np.array(msd)
        valid = np.isfinite(msd) & (msd > 0)

        if valid.sum() < 3:
            return (np.nan, np.nan)

        lag = np.arange(1, len(msd) + 1)[valid]
        msd_valid = msd[valid]

        # Fit MSD = 2*d*D*t^alpha
        log_lag = np.log(lag * dt)
        log_msd = np.log(msd_valid)

        coeffs = np.polyfit(log_lag, log_msd, 1)

        if alpha is None:
            alpha_fit = float(coeffs[0])
        else:
            alpha_fit = alpha

        # D from intercept: log(MSD) = log(2*d*D) + alpha*log(t)
        log_2dD = coeffs[1]
        D = np.exp(log_2dD) / (2 * dim)

        return (float(D), float(np.clip(alpha_fit, 0.1, 2.5)))

    except Exception:
        return (np.nan, np.nan)


def generate_sliding_windows(track_length: int, window_sizes: List[int],
                             overlap: float = 0.75, min_seg_length: int = 10) -> Dict[int, List[Tuple[int, int]]]:
    """
    Generate sliding window segments for a track.

    Args:
        track_length: Total length of the track in frames
        window_sizes: List of window sizes to use
        overlap: Overlap fraction (0.75 = 75%)
        min_seg_length: Minimum segment length (for smoothing)

    Returns:
        Dict mapping window_size -> list of (start, end) tuples
    """
    windows = {}

    for win_size in window_sizes:
        if win_size > track_length:
            continue

        if win_size < min_seg_length:
            continue

        step = max(1, int(win_size * (1 - overlap)))

        segments = []
        start = 0
        while start + win_size <= track_length:
            segments.append((start, start + win_size))
            start += step

        # Add final segment if needed
        if segments and segments[-1][1] < track_length:
            if track_length - segments[-1][1] >= min_seg_length:
                segments.append((track_length - win_size, track_length))

        if segments:
            windows[win_size] = segments

    return windows


def classify_segments_with_voting(track: pd.DataFrame, model, scaler,
                                  window_sizes: List[int] = [10, 20, 30, 40, 50, 100, 150, 200],
                                  overlap: float = 0.75, min_seg_length: int = 10,
                                  dt: float = 0.1) -> pd.DataFrame:
    """
    Classify track segments using sliding windows and voting.

    Args:
        track: DataFrame with columns ['frame', 'x', 'y', 'z'] (x,y,z in nm)
        model: Trained RF classifier
        scaler: Feature scaler
        window_sizes: List of window sizes
        overlap: Overlap fraction
        min_seg_length: Minimum segment length
        dt: Integration time in seconds

    Returns:
        DataFrame with columns ['frame', 'label', 'confidence', 'votes']
    """
    track = track.sort_values('frame').reset_index(drop=True)
    N = len(track)

    # Convert nm to μm for feature extraction
    pos_cols = ['x', 'y'] + (['z'] if 'z' in track.columns else [])
    trajectory_um = track[pos_cols].values / 1000.0  # nm → μm

    # Initialize voting array: [frame_idx, class_0_votes, class_1_votes, class_2_votes, class_3_votes]
    votes = np.zeros((N, 4), dtype=int)

    # Generate windows
    windows = generate_sliding_windows(N, window_sizes, overlap, min_seg_length)

    # Classify each segment
    for win_size, segments in windows.items():
        for start, end in segments:
            segment = trajectory_um[start:end]

            if len(segment) < min_seg_length:
                continue

            try:
                # Extract features
                extractor = DiffusionFeatureExtractor(segment, dt=dt)
                features = extractor.extract_all_features()

                # Convert to DataFrame with correct order
                X = pd.DataFrame([features])

                # Scale and predict
                X_scaled = scaler.transform(X)
                pred_label = model.predict(X_scaled)[0]

                # Add votes for this segment
                votes[start:end, pred_label] += 1

            except Exception:
                # Skip failed segments
                continue

    # Determine final label per frame by majority vote
    labels = []
    confidences = []
    vote_counts = []

    for i in range(N):
        frame_votes = votes[i]
        total_votes = frame_votes.sum()

        if total_votes == 0:
            # No votes - use default (normal diffusion)
            labels.append(0)
            confidences.append(0.0)
            vote_counts.append(0)
        else:
            winning_label = int(np.argmax(frame_votes))
            confidence = frame_votes[winning_label] / total_votes

            labels.append(winning_label)
            confidences.append(float(confidence))
            vote_counts.append(int(total_votes))

    result = pd.DataFrame({
        'frame': track['frame'].values,
        'label': labels,
        'confidence': confidences,
        'votes': vote_counts
    })

    return result


def smooth_labels(labels: np.ndarray, min_segment_length: int = 10) -> np.ndarray:
    """
    Smooth label transitions by removing short segments.

    Args:
        labels: Array of labels
        min_segment_length: Minimum length for a segment to persist

    Returns:
        Smoothed label array
    """
    if len(labels) < min_segment_length:
        return labels

    smoothed = labels.copy()

    # Find consecutive segments
    changes = np.where(np.diff(labels) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [len(labels)]])

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment_length = end - start

        # If segment is too short, replace with neighbor labels
        if segment_length < min_segment_length:
            # Use label from previous segment (or next if at start)
            if i > 0:
                replacement_label = smoothed[boundaries[i-1]]
            elif i < len(boundaries) - 2:
                replacement_label = smoothed[boundaries[i+1]]
            else:
                continue

            smoothed[start:end] = replacement_label

    return smoothed


def plot_track_by_diffusion_type(track_data: pd.DataFrame, labels: np.ndarray,
                                 save_path: Path, label_names: Dict[int, str],
                                 view_mode: str = 'raw'):
    """
    Plot 3D track colored by diffusion type.

    Args:
        track_data: DataFrame with x, y, z in nm
        labels: Array of labels per frame
        save_path: Path to save SVG
        label_names: Dict mapping label -> name
        view_mode: 'raw', 'time', or 'snr' (determines styling)
    """
    # Color scheme for diffusion types
    colors = {
        0: '#2E86AB',  # Normal - blue
        1: '#A23B72',  # Subdiffusion - purple
        2: '#F18F01',  # Confined - orange
        3: '#C73E1D',  # Superdiffusion - red
    }

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = track_data['x'].values
    y = track_data['y'].values
    z = track_data['z'].values if 'z' in track_data.columns else np.zeros_like(x)

    # Plot segments by label
    for i in range(len(x) - 1):
        label = labels[i]
        color = colors.get(label, '#888888')
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=color, linewidth=2, alpha=0.8)

    # Start/end markers
    ax.scatter(x[0], y[0], z[0], color='green', s=80, marker='o',
               edgecolors='white', linewidths=2, zorder=10, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='black', s=80, marker='o',
               edgecolors='white', linewidths=2, zorder=10, label='End')

    # Create custom legend for diffusion types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=label_names.get(i, f'Type {i}'))
                       for i in sorted(set(labels)) if i in colors]
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='green', markersize=8, label='Start'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='black', markersize=8, label='End'))

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


def create_boxplots(analysis_df: pd.DataFrame, save_dir: Path, label_names: Dict[int, str]):
    """
    Create boxplots for alpha and D grouped by diffusion type.

    Args:
        analysis_df: DataFrame with columns ['track_id', 'label', 'alpha', 'D', ...]
        save_dir: Directory to save plots
        label_names: Dict mapping label -> name
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid values
    valid = analysis_df[['label', 'alpha', 'D']].copy()
    valid = valid[np.isfinite(valid['alpha']) & np.isfinite(valid['D'])]

    if len(valid) == 0:
        return

    # Alpha boxplot
    fig, ax = plt.subplots(figsize=(8, 6))

    labels_present = sorted(valid['label'].unique())
    data_alpha = [valid[valid['label'] == label]['alpha'].values for label in labels_present]
    labels_str = [label_names.get(label, f'Type {label}') for label in labels_present]

    bp = ax.boxplot(data_alpha, labels=labels_str, patch_artist=True)

    # Color boxes
    colors = {0: '#2E86AB', 1: '#A23B72', 2: '#F18F01', 3: '#C73E1D'}
    for patch, label in zip(bp['boxes'], labels_present):
        patch.set_facecolor(colors.get(label, '#888888'))
        patch.set_alpha(0.7)

    ax.set_ylabel('Alpha (Anomalous Exponent)', fontsize=12)
    ax.set_xlabel('Diffusion Type', fontsize=12)
    ax.set_title('Alpha Distribution by Diffusion Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'alpha_boxplot.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    # D boxplot
    fig, ax = plt.subplots(figsize=(8, 6))

    data_D = [valid[valid['label'] == label]['D'].values for label in labels_present]

    bp = ax.boxplot(data_D, labels=labels_str, patch_artist=True)

    for patch, label in zip(bp['boxes'], labels_present):
        patch.set_facecolor(colors.get(label, '#888888'))
        patch.set_alpha(0.7)

    ax.set_ylabel('D (Diffusion Coefficient) [μm²/s]', fontsize=12)
    ax.set_xlabel('Diffusion Type', fontsize=12)
    ax.set_title('Diffusion Coefficient Distribution by Diffusion Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Log scale if needed
    if valid['D'].max() / (valid['D'].min() + 1e-10) > 100:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_dir / 'D_boxplot.svg', format='svg', bbox_inches='tight')
    plt.close(fig)


def load_rf_model(base_dir: Path) -> Tuple[Optional[object], Optional[object], Optional[Dict]]:
    """
    Load RF model, scaler, and metadata from directory.

    Args:
        base_dir: Directory containing model files

    Returns:
        (model, scaler, metadata) or (None, None, None) if not found
    """
    import json

    base_dir = Path(base_dir)

    # Find latest model files
    model_files = list(base_dir.glob('rf_diffusion_classifier_*.pkl'))
    scaler_files = list(base_dir.glob('feature_scaler_*.pkl'))
    metadata_files = list(base_dir.glob('model_metadata_*.json'))

    if not model_files or not scaler_files or not metadata_files:
        return (None, None, None)

    # Use latest
    model_file = sorted(model_files)[-1]
    scaler_file = sorted(scaler_files)[-1]
    metadata_file = sorted(metadata_files)[-1]

    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return (model, scaler, metadata)

    except Exception as e:
        print(f"Error loading RF model: {e}")
        return (None, None, None)


def perform_rf_classification_on_tracks(tracks: pd.DataFrame, model, scaler, metadata: Dict,
                                       window_sizes: List[int] = [10, 20, 30, 40, 50, 100, 150, 200],
                                       overlap: float = 0.75, min_seg_length: int = 10,
                                       dt: float = 0.1,
                                       progress_callback=None) -> pd.DataFrame:
    """
    Perform RF classification on all tracks with sliding window approach.

    Args:
        tracks: DataFrame with columns ['particle', 'frame', 'x', 'y', 'z']
        model: Trained RF classifier
        scaler: Feature scaler
        metadata: Model metadata dict
        window_sizes: List of window sizes
        overlap: Overlap fraction
        min_seg_length: Minimum segment length for smoothing
        dt: Integration time in seconds
        progress_callback: Optional callback for progress updates

    Returns:
        DataFrame with columns ['particle', 'frame', 'x', 'y', 'z', 'rf_label', 'rf_confidence', 'rf_votes']
    """
    if progress_callback:
        progress_callback("Starting RF classification on all tracks...")

    particle_ids = sorted(tracks['particle'].unique())
    total_tracks = len(particle_ids)

    results = []

    for idx, pid in enumerate(particle_ids, 1):
        track = tracks[tracks['particle'] == pid].copy()

        if len(track) < min_seg_length:
            # Too short - skip
            track['rf_label'] = 0
            track['rf_confidence'] = 0.0
            track['rf_votes'] = 0
            results.append(track)
            continue

        try:
            # Classify segments with voting
            classification = classify_segments_with_voting(
                track, model, scaler,
                window_sizes=window_sizes,
                overlap=overlap,
                min_seg_length=min_seg_length,
                dt=dt
            )

            # Smooth labels
            smoothed_labels = smooth_labels(classification['label'].values, min_segment_length=min_seg_length)

            # Merge back
            track['rf_label'] = smoothed_labels
            track['rf_confidence'] = classification['confidence'].values
            track['rf_votes'] = classification['votes'].values

            results.append(track)

            if progress_callback and (idx % 10 == 0 or idx == total_tracks):
                progress_callback(f"  Classified {idx}/{total_tracks} tracks")

        except Exception as e:
            if progress_callback:
                progress_callback(f"  Error classifying track {pid}: {e}")

            # Use default labels
            track['rf_label'] = 0
            track['rf_confidence'] = 0.0
            track['rf_votes'] = 0
            results.append(track)

    if progress_callback:
        progress_callback("RF classification complete!")

    return pd.concat(results, ignore_index=True)


def export_rf_visualizations(tracks_classified: pd.DataFrame, output_dir: Path,
                             label_names: Dict[int, str], n_tracks_to_plot='all',
                             progress_callback=None):
    """
    Export RF-classified tracks as colored 3D plots.

    Args:
        tracks_classified: DataFrame with rf_label column
        output_dir: Output directory (will create 06_RF_Model_Classification/)
        label_names: Dict mapping label -> name
        n_tracks_to_plot: 'all' or integer
        progress_callback: Optional callback for progress
    """
    rf_dir = output_dir / '06_RF_Model_Classification'
    rf_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Creating RF classification plots...")

    particle_ids = sorted(tracks_classified['particle'].unique())

    if n_tracks_to_plot != 'all':
        # Select longest tracks
        track_lengths = tracks_classified.groupby('particle').size()
        longest = track_lengths.nlargest(n_tracks_to_plot).index
        particle_ids = [pid for pid in particle_ids if pid in longest.values]

    total = len(particle_ids)

    for idx, pid in enumerate(particle_ids, 1):
        track = tracks_classified[tracks_classified['particle'] == pid].copy().sort_values('frame')

        save_path = rf_dir / f'Track_{int(pid):04d}_RF.svg'

        plot_track_by_diffusion_type(
            track,
            track['rf_label'].values,
            save_path,
            label_names,
            view_mode='raw'
        )

        if progress_callback and (idx % 5 == 0 or idx == total):
            progress_callback(f"  {idx}/{total} RF plots created")

    if progress_callback:
        progress_callback(f"RF visualization complete → {rf_dir}")


def export_rf_analysis(tracks_classified: pd.DataFrame, output_dir: Path,
                      label_names: Dict[int, str], dt: float = 0.1,
                      progress_callback=None):
    """
    Export comprehensive RF analysis to Excel, CSV, and plots.

    Args:
        tracks_classified: DataFrame with rf_label column
        output_dir: Output directory (will create 07_RF_Analysis/)
        label_names: Dict mapping label -> name
        dt: Integration time in seconds
        progress_callback: Optional callback for progress
    """
    analysis_dir = output_dir / '07_RF_Analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Creating RF analysis exports...")

    # === 1. Per-track summary ===
    particle_ids = sorted(tracks_classified['particle'].unique())

    track_summaries = []

    for pid in particle_ids:
        track = tracks_classified[tracks_classified['particle'] == pid].copy()

        if len(track) == 0:
            continue

        # Dominant label (most common)
        label_counts = track['rf_label'].value_counts()
        dominant_label = int(label_counts.idxmax())
        dominant_fraction = label_counts.max() / len(track)

        # Calculate alpha and D
        pos_cols = ['x', 'y'] + (['z'] if 'z' in track.columns else [])
        trajectory_um = track[pos_cols].values / 1000.0  # nm → μm

        D, alpha = calculate_diffusion_coefficient(trajectory_um, dt=dt)

        # Extract full features for this track
        try:
            extractor = DiffusionFeatureExtractor(trajectory_um, dt=dt)
            features = extractor.extract_all_features()
        except Exception:
            features = {name: np.nan for name in [
                'alpha', 'msd_ratio', 'hurst_exponent', 'vacf_lag1', 'vacf_min',
                'kurtosis', 'straightness', 'mean_cos_theta', 'persistence_length',
                'efficiency', 'rg_saturation', 'asphericity', 'fractal_dimension',
                'convex_hull_area', 'confinement_probability', 'msd_plateauness',
                'space_exploration_ratio', 'boundary_proximity_var'
            ]}

        summary = {
            'track_id': int(pid),
            'length': len(track),
            'dominant_label': dominant_label,
            'dominant_label_name': label_names.get(dominant_label, f'Type {dominant_label}'),
            'dominant_fraction': float(dominant_fraction),
            'mean_confidence': float(track['rf_confidence'].mean()),
            'D': float(D),
            'alpha': float(alpha),
        }

        # Add all features
        for feature_name, feature_value in features.items():
            summary[f'feature_{feature_name}'] = float(feature_value)

        track_summaries.append(summary)

    summary_df = pd.DataFrame(track_summaries)

    # === 2. Per-frame detailed data ===
    detailed_df = tracks_classified.copy()
    detailed_df['label_name'] = detailed_df['rf_label'].map(label_names)

    # === 3. Label distribution summary ===
    label_stats = []
    for label in sorted(tracks_classified['rf_label'].unique()):
        label_data = summary_df[summary_df['dominant_label'] == label]

        stats = {
            'label': int(label),
            'label_name': label_names.get(label, f'Type {label}'),
            'n_tracks': len(label_data),
            'mean_alpha': float(label_data['alpha'].mean()) if len(label_data) > 0 else np.nan,
            'std_alpha': float(label_data['alpha'].std()) if len(label_data) > 0 else np.nan,
            'mean_D': float(label_data['D'].mean()) if len(label_data) > 0 else np.nan,
            'std_D': float(label_data['D'].std()) if len(label_data) > 0 else np.nan,
        }

        label_stats.append(stats)

    label_stats_df = pd.DataFrame(label_stats)

    # === 4. Export to Excel ===
    if progress_callback:
        progress_callback("  Writing Excel file...")

    excel_path = analysis_dir / 'rf_analysis_complete.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Summary per track
        summary_df.to_excel(writer, sheet_name='Track_Summary', index=False)

        # Label statistics
        label_stats_df.to_excel(writer, sheet_name='Label_Statistics', index=False)

        # Detailed per-frame (limit to first 100k rows to avoid Excel limits)
        if len(detailed_df) > 100000:
            detailed_df.head(100000).to_excel(writer, sheet_name='Detailed_Data_Sample', index=False)
        else:
            detailed_df.to_excel(writer, sheet_name='Detailed_Data', index=False)

    if progress_callback:
        progress_callback(f"  Excel saved → {excel_path.name}")

    # === 5. Export CSVs ===
    if progress_callback:
        progress_callback("  Writing CSV files...")

    summary_df.to_csv(analysis_dir / 'track_summary.csv', index=False)
    label_stats_df.to_csv(analysis_dir / 'label_statistics.csv', index=False)
    detailed_df.to_csv(analysis_dir / 'detailed_per_frame.csv', index=False)

    # === 6. Create boxplots ===
    if progress_callback:
        progress_callback("  Creating boxplots...")

    create_boxplots(summary_df, analysis_dir, label_names)

    # === 7. Create summary histogram ===
    if progress_callback:
        progress_callback("  Creating label distribution plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    label_counts = summary_df['dominant_label'].value_counts().sort_index()
    labels_present = label_counts.index.tolist()
    counts = label_counts.values

    colors_map = {0: '#2E86AB', 1: '#A23B72', 2: '#F18F01', 3: '#C73E1D'}
    bar_colors = [colors_map.get(label, '#888888') for label in labels_present]

    x_labels = [label_names.get(label, f'Type {label}') for label in labels_present]

    ax.bar(x_labels, counts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Number of Tracks', fontsize=12)
    ax.set_xlabel('Diffusion Type', fontsize=12)
    ax.set_title('Track Distribution by Diffusion Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for i, (label, count) in enumerate(zip(x_labels, counts)):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(analysis_dir / 'label_distribution.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    if progress_callback:
        progress_callback(f"RF analysis complete → {analysis_dir}")

    return summary_df, label_stats_df
