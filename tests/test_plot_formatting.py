"""Tests for plot formatting and new plot functions."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tempfile
import shutil
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from plot_config import (
        configure_plot_defaults,
        format_axis,
        get_diffusion_colors,
        get_diffusion_labels,
        save_figure,
        apply_publication_style
    )
    PLOT_CONFIG_AVAILABLE = True
except ImportError:
    PLOT_CONFIG_AVAILABLE = False


try:
    from time_series_plots import (
        plot_diffusion_fractions_area,
        plot_D_boxplots_per_diffusion_type,
        plot_D_log_linear_per_diffusion_type,
        plot_alpha_log_linear_per_diffusion_type
    )
    TIME_SERIES_PLOTS_AVAILABLE = True
except ImportError:
    TIME_SERIES_PLOTS_AVAILABLE = False


@pytest.mark.skipif(not PLOT_CONFIG_AVAILABLE, reason="plot_config module not available")
class TestPlotConfig:
    """Test suite for plot configuration utilities."""

    def test_configure_defaults(self):
        """Test that plot defaults are configured."""
        configure_plot_defaults()

        # Check that matplotlib params are set
        assert plt.rcParams['axes.grid'] == False
        assert plt.rcParams['savefig.format'] == 'svg'
        assert plt.rcParams['axes.spines.top'] == False
        assert plt.rcParams['axes.spines.right'] == False

    def test_get_diffusion_colors(self):
        """Test diffusion color mapping."""
        colors = get_diffusion_colors()

        assert len(colors) == 4, "Should have 4 diffusion types"
        assert 0 in colors, "Should have Normal (0)"
        assert 1 in colors, "Should have Subdiffusion (1)"
        assert 2 in colors, "Should have Confined (2)"
        assert 3 in colors, "Should have Superdiffusion (3)"

        # Check that colors are valid hex codes
        for color in colors.values():
            assert color.startswith('#'), "Colors should be hex codes"
            assert len(color) == 7, "Hex codes should be 7 characters"

    def test_get_diffusion_labels(self):
        """Test diffusion label mapping."""
        labels = get_diffusion_labels()

        assert len(labels) == 4, "Should have 4 diffusion types"
        assert labels[0] == 'Normal'
        assert labels[1] == 'Subdiffusion'
        assert labels[2] == 'Confined'
        assert labels[3] == 'Superdiffusion'

    def test_format_axis_basic(self):
        """Test basic axis formatting."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label='Test')

        format_axis(ax, xlabel='X', ylabel='Y', legend=True)

        # Check that grid is disabled
        assert not ax.xaxis._gridOnMajor, "Grid should be disabled"

        # Check that title is empty
        assert ax.get_title() == '', "Title should be empty"

        # Check labels
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'

        plt.close(fig)

    def test_format_axis_log_scale(self):
        """Test log scale formatting."""
        fig, ax = plt.subplots()
        ax.plot([1, 10, 100], [1, 10, 100])

        format_axis(ax, xlabel='X', ylabel='Y', log_x=True, log_y=True)

        # Check log scales
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'

        plt.close(fig)

    def test_save_figure(self):
        """Test figure saving with SVG format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / 'test_plot.svg'

            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])

            save_figure(fig, temp_path)

            # Check that file was created
            assert temp_path.exists(), "SVG file should be created"
            assert temp_path.stat().st_size > 0, "SVG file should not be empty"

    def test_apply_publication_style(self):
        """Test publication style application."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label='Data')
        ax.set_title('Title')
        ax.grid(True)
        ax.legend()

        apply_publication_style(fig, ax)

        # Check that grid is removed
        assert not ax.xaxis._gridOnMajor

        # Check that title is removed
        assert ax.get_title() == ''

        # Check spines
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        assert ax.spines['left'].get_visible()
        assert ax.spines['bottom'].get_visible()

        plt.close(fig)


@pytest.mark.skipif(not TIME_SERIES_PLOTS_AVAILABLE, reason="time_series_plots module not available")
class TestTimeSeriesPlots:
    """Test suite for new time series plotting functions."""

    @pytest.fixture
    def mock_time_series_data(self):
        """Create mock time series data."""
        np.random.seed(42)

        n_tracks = 100
        times = [0.0, 10.0, 20.0, 30.0]

        data = []
        for t in times:
            for label_id in [0, 1, 2, 3]:
                n_tracks_label = np.random.randint(10, 30)
                for _ in range(n_tracks_label):
                    data.append({
                        'polymerization_time': t,
                        'predicted_label': label_id,
                        'alpha': np.random.uniform(0.5, 1.5),
                        'D': np.random.uniform(0.01, 1.0),
                        'track_id': np.random.randint(0, 1000)
                    })

        return pd.DataFrame(data)

    def test_plot_diffusion_fractions_area(self, mock_time_series_data):
        """Test stacked area chart creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'fractions_area.svg'

            plot_diffusion_fractions_area(
                mock_time_series_data,
                save_path,
                progress_callback=None
            )

            assert save_path.exists(), "Area chart should be created"
            assert save_path.stat().st_size > 0, "File should not be empty"

    def test_plot_D_boxplots_per_diffusion_type(self, mock_time_series_data):
        """Test D boxplot creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / 'D_boxplots'

            plot_D_boxplots_per_diffusion_type(
                mock_time_series_data,
                save_dir,
                progress_callback=None
            )

            assert save_dir.exists(), "Boxplot directory should be created"

            # Check that files were created for each diffusion type
            svg_files = list(save_dir.glob('*.svg'))
            assert len(svg_files) > 0, "Should create at least one boxplot"

    def test_plot_D_log_linear(self, mock_time_series_data):
        """Test D log/linear plots creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / 'D_plots'

            plot_D_log_linear_per_diffusion_type(
                mock_time_series_data,
                save_dir,
                progress_callback=None
            )

            assert save_dir.exists(), "D plots directory should be created"

            # Check for both log and linear plots
            svg_files = list(save_dir.glob('*.svg'))
            assert len(svg_files) > 0, "Should create D plots"

            # Check for log and linear variants
            log_files = [f for f in svg_files if 'log' in f.name]
            linear_files = [f for f in svg_files if 'linear' in f.name]

            assert len(log_files) > 0, "Should create log plots"
            assert len(linear_files) > 0, "Should create linear plots"

    def test_plot_alpha_log_linear(self, mock_time_series_data):
        """Test alpha log/linear plots creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / 'alpha_plots'

            plot_alpha_log_linear_per_diffusion_type(
                mock_time_series_data,
                save_dir,
                progress_callback=None
            )

            assert save_dir.exists(), "Alpha plots directory should be created"

            svg_files = list(save_dir.glob('*.svg'))
            assert len(svg_files) > 0, "Should create alpha plots"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
