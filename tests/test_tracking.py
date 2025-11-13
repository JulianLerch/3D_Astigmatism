"""Tests for core tracking functionality."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracking_tool import (
    calculate_snr,
    apply_quality_prefilter,
    load_and_prepare_data,
    perform_3d_tracking
)


class TestSNRCalculation:
    """Test suite for SNR calculation."""

    def test_calculate_snr_basic(self):
        """Test basic SNR calculation."""
        df = pd.DataFrame({
            'intensity [photon]': [1000, 2000, 3000],
            'offset [photon]': [100, 200, 300],
            'bkgstd [photon]': [10, 20, 30]
        })

        snr = calculate_snr(df)

        # SNR = (intensity - offset) / bkgstd
        expected = [(1000-100)/10, (2000-200)/20, (3000-300)/30]
        np.testing.assert_array_almost_equal(snr.values, expected)

    def test_calculate_snr_missing_columns(self):
        """Test SNR calculation with missing columns."""
        df = pd.DataFrame({
            'x [nm]': [100, 200],
            'y [nm]': [150, 250]
        })

        snr = calculate_snr(df)

        # Should return NaN when required columns missing
        assert snr.isna().all(), "Should return NaN when columns missing"

    def test_calculate_snr_zero_background(self):
        """Test SNR calculation when background std is zero."""
        df = pd.DataFrame({
            'intensity [photon]': [1000, 2000],
            'offset [photon]': [100, 200],
            'bkgstd [photon]': [0, 10]  # First is zero
        })

        snr = calculate_snr(df)

        # First should be 0 (replaced), second should be calculated
        assert snr.iloc[0] == 0, "Zero background should give SNR of 0"
        assert snr.iloc[1] == (2000-200)/10, "Valid SNR should be calculated"


class TestQualityPrefilter:
    """Test suite for quality pre-filtering."""

    def test_prefilter_basic(self):
        """Test basic quality filtering."""
        df = pd.DataFrame({
            'x [nm]': np.random.rand(100) * 1000,
            'y [nm]': np.random.rand(100) * 1000,
            'uncertainty [nm]': np.random.rand(100) * 50,  # 0-50 nm
            'chi2': np.random.rand(100) * 2,
            'SNR': np.random.rand(100) * 100,
            'frame': range(100)
        })

        df_filtered, stats = apply_quality_prefilter(
            df,
            uncertainty_max=30.0,
            chi2_percentile=95.0,
            snr_min_percentile=10.0,
            snr_max_percentile=99.0
        )

        # Check that filtering occurred
        assert len(df_filtered) <= len(df), "Filtered should have <= rows"
        assert stats['n_start'] == 100, "Should record original count"
        assert stats['n_final'] == len(df_filtered), "Final count should match"

    def test_prefilter_statistics(self):
        """Test that filtering statistics are correct."""
        df = pd.DataFrame({
            'uncertainty [nm]': [10, 20, 30, 40, 50],  # Last 2 should be filtered
            'chi2': [1.0, 1.5, 2.0, 2.5, 3.0],
            'SNR': [50, 60, 70, 80, 90],
            'frame': range(5)
        })

        df_filtered, stats = apply_quality_prefilter(
            df,
            uncertainty_max=30.0,
            chi2_percentile=95.0,
            snr_min_percentile=10.0,
            snr_max_percentile=99.0
        )

        assert 'n_start' in stats
        assert 'n_final' in stats
        assert 'removed_total' in stats
        assert 'removed_percent' in stats

        assert stats['n_start'] == 5
        assert stats['removed_total'] == stats['n_start'] - stats['n_final']

    def test_prefilter_no_quality_columns(self):
        """Test filtering when quality columns are missing."""
        df = pd.DataFrame({
            'x [nm]': [100, 200, 300],
            'y [nm]': [150, 250, 350],
            'frame': [0, 1, 2]
        })

        df_filtered, stats = apply_quality_prefilter(df)

        # Should return all data when no quality columns present
        assert len(df_filtered) == len(df), \
            "Should return all data when no quality columns"


class TestTracking:
    """Test suite for 3D tracking functionality."""

    @pytest.fixture
    def sample_localizations(self):
        """Create sample localization data for tracking."""
        np.random.seed(42)

        # Create 2 linear tracks
        n_frames = 50
        track1_x = np.linspace(100, 200, n_frames) + np.random.randn(n_frames) * 2
        track1_y = np.linspace(100, 200, n_frames) + np.random.randn(n_frames) * 2
        track1_z = np.linspace(0, 100, n_frames) + np.random.randn(n_frames) * 5

        track2_x = np.linspace(500, 600, n_frames) + np.random.randn(n_frames) * 2
        track2_y = np.linspace(500, 400, n_frames) + np.random.randn(n_frames) * 2
        track2_z = np.linspace(50, 150, n_frames) + np.random.randn(n_frames) * 5

        df = pd.DataFrame({
            'x [nm]': np.concatenate([track1_x, track2_x]),
            'y [nm]': np.concatenate([track1_y, track2_y]),
            'z [nm]': np.concatenate([track1_z, track2_z]),
            'frame': np.concatenate([range(n_frames), range(n_frames)])
        })

        return df

    def test_perform_3d_tracking(self, sample_localizations):
        """Test 3D tracking on sample data."""
        tracks, stats = perform_3d_tracking(
            sample_localizations,
            search_range=50.0,
            memory=0,
            min_track_length=10
        )

        assert not tracks.empty, "Should find tracks"
        assert 'particle' in tracks.columns, "Should have particle ID column"
        assert stats['n_tracks'] > 0, "Should have positive track count"

    def test_tracking_statistics(self, sample_localizations):
        """Test that tracking statistics are calculated correctly."""
        tracks, stats = perform_3d_tracking(
            sample_localizations,
            search_range=50.0,
            memory=0,
            min_track_length=10
        )

        required_stats = [
            'n_tracks', 'n_localizations', 'min_length',
            'max_length', 'mean_length'
        ]

        for stat in required_stats:
            assert stat in stats, f"Statistics should include {stat}"

        assert stats['n_localizations'] <= len(sample_localizations), \
            "Tracked localizations should be <= input"

    def test_tracking_min_length_filter(self, sample_localizations):
        """Test that minimum length filter works."""
        # Track with high min length
        tracks_long, stats_long = perform_3d_tracking(
            sample_localizations,
            search_range=50.0,
            memory=0,
            min_track_length=40
        )

        # Track with low min length
        tracks_short, stats_short = perform_3d_tracking(
            sample_localizations,
            search_range=50.0,
            memory=0,
            min_track_length=5
        )

        # More lenient filter should give more (or equal) tracks
        assert stats_short['n_tracks'] >= stats_long['n_tracks'], \
            "Lower min_length should give more tracks"

    def test_tracking_preserves_z_coordinate(self, sample_localizations):
        """Test that z-coordinate is preserved during tracking."""
        tracks, stats = perform_3d_tracking(
            sample_localizations,
            search_range=50.0,
            memory=0,
            min_track_length=10
        )

        # Check that z column exists and has values
        assert 'z [nm]' in tracks.columns or 'z' in tracks.columns, \
            "Should preserve z coordinate"

        z_col = 'z [nm]' if 'z [nm]' in tracks.columns else 'z'
        assert tracks[z_col].notna().any(), "Should have z values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
