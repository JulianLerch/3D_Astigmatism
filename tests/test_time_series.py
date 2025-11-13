"""Tests for time series analysis functionality."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from time_series_analysis import (
        export_time_series_analysis,
        aggregate_time_series_data,
        calculate_time_series_statistics
    )
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False


@pytest.mark.skipif(not TIMESERIES_AVAILABLE, reason="time_series_analysis module not available")
class TestTimeSeriesAnalysis:
    """Test suite for time series analysis functionality."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_tracking_folders(self, temp_output_dir):
        """Create mock tracking result folders for testing."""
        folders_with_times = []

        for i, poly_time in enumerate([0.0, 10.0, 20.0, 30.0]):
            folder = temp_output_dir / f"experiment_{i}"
            folder.mkdir(parents=True)

            # Create mock RF analysis results
            rf_dir = folder / '3D_Tracking_Results' / '08_RF_Analysis'
            rf_dir.mkdir(parents=True, exist_ok=True)

            # Create mock track_summary.csv
            n_tracks = 10 + i * 5
            track_data = {
                'track_id': list(range(n_tracks)),
                'predicted_label': np.random.choice([0, 1, 2, 3], n_tracks),
                'alpha': np.random.uniform(0.5, 1.5, n_tracks),
                'D': np.random.uniform(0.01, 1.0, n_tracks),
                'n_frames': np.random.randint(50, 200, n_tracks)
            }
            df = pd.DataFrame(track_data)
            df.to_csv(rf_dir / 'track_summary.csv', index=False)

            folders_with_times.append((folder, poly_time))

        return folders_with_times

    def test_timeseries_folder_creation(self, temp_output_dir, mock_tracking_folders):
        """Test that timeSeries folder is created."""
        export_time_series_analysis(
            folders_with_times=mock_tracking_folders,
            output_dir=temp_output_dir,
            progress_callback=None
        )

        ts_dir = temp_output_dir / 'timeSeries'
        assert ts_dir.exists(), "timeSeries folder should be created"
        assert ts_dir.is_dir(), "timeSeries should be a directory"

    def test_timeseries_output_files(self, temp_output_dir, mock_tracking_folders):
        """Test that required output files are created."""
        export_time_series_analysis(
            folders_with_times=mock_tracking_folders,
            output_dir=temp_output_dir,
            progress_callback=None
        )

        ts_dir = temp_output_dir / 'timeSeries'

        # Check for required CSV files
        assert (ts_dir / 'time_series_statistics.csv').exists(), \
            "time_series_statistics.csv should be created"
        assert (ts_dir / 'time_series_all_tracks.csv').exists(), \
            "time_series_all_tracks.csv should be created"

        # Check for Excel file
        assert (ts_dir / 'time_series_analysis.xlsx').exists(), \
            "time_series_analysis.xlsx should be created"

    def test_timeseries_plots_created(self, temp_output_dir, mock_tracking_folders):
        """Test that time series plots are created."""
        export_time_series_analysis(
            folders_with_times=mock_tracking_folders,
            output_dir=temp_output_dir,
            progress_callback=None
        )

        ts_dir = temp_output_dir / 'timeSeries'

        # Check for required plots (SVG format)
        expected_plots = [
            'track_counts_over_time.svg',
            'diffusion_fractions_over_time.svg',  # CRITICAL: Distribution of diffusion types!
            'alpha_evolution.svg',
            'D_evolution.svg',
            'diffusion_heatmap.svg',
            'overall_trends.svg'
        ]

        for plot_name in expected_plots:
            plot_path = ts_dir / plot_name
            assert plot_path.exists(), f"{plot_name} should be created"
            assert plot_path.stat().st_size > 0, f"{plot_name} should not be empty"

    def test_diffusion_fractions_plot_exists(self, temp_output_dir, mock_tracking_folders):
        """Test that diffusion fractions plot (distribution over time) is created."""
        export_time_series_analysis(
            folders_with_times=mock_tracking_folders,
            output_dir=temp_output_dir,
            progress_callback=None
        )

        ts_dir = temp_output_dir / 'timeSeries'
        plot_path = ts_dir / 'diffusion_fractions_over_time.svg'

        assert plot_path.exists(), \
            "diffusion_fractions_over_time.svg (distribution of diffusion types) must exist"

    def test_aggregate_time_series_data(self, mock_tracking_folders):
        """Test data aggregation from multiple time points."""
        combined_df = aggregate_time_series_data(
            mock_tracking_folders,
            progress_callback=None
        )

        assert not combined_df.empty, "Combined dataframe should not be empty"
        assert 'polymerization_time' in combined_df.columns, \
            "Should have polymerization_time column"
        assert 'predicted_label' in combined_df.columns, \
            "Should have predicted_label column"

        # Check that all time points are present
        time_points = combined_df['polymerization_time'].unique()
        expected_times = [0.0, 10.0, 20.0, 30.0]
        for t in expected_times:
            assert t in time_points, f"Time point {t} should be in data"

    def test_calculate_time_series_statistics(self, mock_tracking_folders):
        """Test statistics calculation for time series."""
        combined_df = aggregate_time_series_data(
            mock_tracking_folders,
            progress_callback=None
        )

        label_names = {
            0: 'Normal Diffusion',
            1: 'Subdiffusion (fBm)',
            2: 'Confined Diffusion',
            3: 'Superdiffusion'
        }

        stats_df = calculate_time_series_statistics(combined_df, label_names)

        assert not stats_df.empty, "Statistics dataframe should not be empty"
        assert 'polymerization_time' in stats_df.columns, \
            "Should have polymerization_time column"

        # Check that diffusion type fractions are present
        for label_name in label_names.values():
            fraction_col = f'{label_name}_fraction'
            assert fraction_col in stats_df.columns, \
                f"Should have {fraction_col} column"

    def test_empty_folders_list(self, temp_output_dir):
        """Test behavior with empty folders list."""
        export_time_series_analysis(
            folders_with_times=[],
            output_dir=temp_output_dir,
            progress_callback=None
        )

        # Should still create directory but may have minimal output
        ts_dir = temp_output_dir / 'timeSeries'
        assert ts_dir.exists(), "timeSeries folder should be created even with empty input"

    def test_progress_callback(self, temp_output_dir, mock_tracking_folders):
        """Test that progress callback is called."""
        messages = []

        def callback(msg):
            messages.append(msg)

        export_time_series_analysis(
            folders_with_times=mock_tracking_folders,
            output_dir=temp_output_dir,
            progress_callback=callback
        )

        assert len(messages) > 0, "Progress callback should be called"
        assert any("TIME SERIES" in msg for msg in messages), \
            "Should contain time series progress messages"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
