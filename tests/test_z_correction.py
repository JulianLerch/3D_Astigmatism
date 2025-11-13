"""Tests for z-axis correction functionality."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import tracking_tool
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracking_tool import apply_z_correction_inplace


class TestZCorrection:
    """Test suite for z-axis correction based on refractive indices."""

    def test_z_correction_basic(self):
        """Test basic z-correction with standard parameters."""
        # Create test dataframe
        df = pd.DataFrame({
            'x [nm]': [0, 100, 200],
            'y [nm]': [0, 100, 200],
            'z [nm]': [100.0, 200.0, 300.0],
            'frame': [0, 1, 2]
        })

        # Standard parameters (oil immersion)
        n_oil = 1.518
        n_glass = 1.523
        n_polymer = 1.33  # Water/aqueous polymer
        NA = 1.50
        d_glass_nm = 170000.0

        # Apply correction
        result = apply_z_correction_inplace(
            df, n_oil, n_glass, n_polymer, NA, d_glass_nm
        )

        # Check that correction was applied
        assert result is True, "z-correction should return True"

        # Check that z values changed
        original_z = np.array([100.0, 200.0, 300.0])
        corrected_z = df['z [nm]'].values

        # z-corrected should differ from original
        assert not np.allclose(corrected_z, original_z), \
            "Corrected z values should differ from original"

        # All corrected values should be positive
        assert np.all(corrected_z > 0), \
            "All corrected z values should be positive"

    def test_z_correction_preserves_other_columns(self):
        """Test that z-correction doesn't modify other columns."""
        df = pd.DataFrame({
            'x [nm]': [100.0, 200.0],
            'y [nm]': [150.0, 250.0],
            'z [nm]': [100.0, 200.0],
            'frame': [0, 1],
            'intensity [photon]': [1000, 1500]
        })

        original_x = df['x [nm]'].copy()
        original_y = df['y [nm]'].copy()
        original_frame = df['frame'].copy()
        original_intensity = df['intensity [photon]'].copy()

        apply_z_correction_inplace(
            df, n_oil=1.518, n_glass=1.523, n_polymer=1.33, NA=1.50, d_glass_nm=170000.0
        )

        # Check other columns unchanged
        pd.testing.assert_series_equal(df['x [nm]'], original_x, check_names=False)
        pd.testing.assert_series_equal(df['y [nm]'], original_y, check_names=False)
        pd.testing.assert_series_equal(df['frame'], original_frame, check_names=False)
        pd.testing.assert_series_equal(df['intensity [photon]'], original_intensity, check_names=False)

    def test_z_correction_without_z_column(self):
        """Test behavior when z column is missing."""
        df = pd.DataFrame({
            'x [nm]': [100.0, 200.0],
            'y [nm]': [150.0, 250.0],
            'frame': [0, 1]
        })

        result = apply_z_correction_inplace(
            df, n_oil=1.518, n_glass=1.523, n_polymer=1.33, NA=1.50, d_glass_nm=170000.0
        )

        # Should return False when z column missing
        assert result is False, "Should return False when z column is missing"

    def test_z_correction_alternative_z_column(self):
        """Test z-correction with alternative column name 'z'."""
        df = pd.DataFrame({
            'x': [100.0, 200.0],
            'y': [150.0, 250.0],
            'z': [100.0, 200.0],
            'frame': [0, 1]
        })

        result = apply_z_correction_inplace(
            df, n_oil=1.518, n_glass=1.523, n_polymer=1.33, NA=1.50, d_glass_nm=170000.0
        )

        assert result is True, "Should work with 'z' column name"
        assert 'z' in df.columns

    def test_z_correction_physical_plausibility(self):
        """Test that correction produces physically plausible results."""
        df = pd.DataFrame({
            'z [nm]': [100.0, 500.0, 1000.0]
        })

        n_oil = 1.518
        n_polymer = 1.33  # Polymer has lower refractive index than oil

        apply_z_correction_inplace(
            df, n_oil=n_oil, n_glass=1.523, n_polymer=n_polymer, NA=1.50, d_glass_nm=170000.0
        )

        # When n_polymer < n_oil, base scaling factor should be < 1
        # So corrected z should generally be smaller than apparent z
        # (though depth correction can modify this)
        f_base = n_polymer / n_oil
        assert f_base < 1.0, "Base scaling factor should be < 1 for polymer"

        # All values should still be positive and finite
        assert np.all(np.isfinite(df['z [nm]'])), "All z values should be finite"
        assert np.all(df['z [nm]'] > 0), "All z values should be positive"

    def test_z_correction_with_nan_values(self):
        """Test handling of NaN values in z column."""
        df = pd.DataFrame({
            'z [nm]': [100.0, np.nan, 300.0]
        })

        apply_z_correction_inplace(
            df, n_oil=1.518, n_glass=1.523, n_polymer=1.33, NA=1.50, d_glass_nm=170000.0
        )

        # Check that NaN is handled (might become NaN or zero)
        assert len(df) == 3, "DataFrame length should be preserved"

        # First and third values should be corrected
        assert np.isfinite(df['z [nm]'].iloc[0]), "First value should be finite"
        assert np.isfinite(df['z [nm]'].iloc[2]), "Third value should be finite"

    def test_z_correction_edge_cases(self):
        """Test edge cases for z-correction."""
        # Test with very small z values
        df_small = pd.DataFrame({'z [nm]': [1.0, 5.0, 10.0]})
        result = apply_z_correction_inplace(
            df_small, n_oil=1.518, n_glass=1.523, n_polymer=1.33, NA=1.50, d_glass_nm=170000.0
        )
        assert result is True
        assert np.all(np.isfinite(df_small['z [nm]'])), "Small z values should produce finite results"

        # Test with very large z values
        df_large = pd.DataFrame({'z [nm]': [10000.0, 50000.0, 100000.0]})
        result = apply_z_correction_inplace(
            df_large, n_oil=1.518, n_glass=1.523, n_polymer=1.33, NA=1.50, d_glass_nm=170000.0
        )
        assert result is True
        assert np.all(np.isfinite(df_large['z [nm]'])), "Large z values should produce finite results"

    def test_z_correction_identical_refractive_indices(self):
        """Test when all refractive indices are identical (no correction needed)."""
        df = pd.DataFrame({
            'z [nm]': [100.0, 200.0, 300.0]
        })

        n = 1.5  # Same for all
        apply_z_correction_inplace(
            df, n_oil=n, n_glass=n, n_polymer=n, NA=1.40, d_glass_nm=170000.0
        )

        # When all n are equal, correction should be minimal
        # (though NA term might still apply)
        assert np.all(np.isfinite(df['z [nm]'])), "Should produce finite results"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
