# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-01-13

### Added - Publication-Ready Plots 📊
- ✅ **Global plot configuration** (`plot_config.py`)
  - Consistent styling: no grids, no titles, minimal spines
  - SVG vector format for all plots
  - Concise English legends
  - Publication-ready defaults
- ✅ **Advanced time series plots** (`time_series_plots.py`)
  - **Stacked area chart** for diffusion type distribution (fills between curves)
  - **Boxplots** for D coefficient at each time point (per diffusion type)
  - **Log and linear plots** for D (per diffusion type)
  - **Log and linear plots** for alpha (per diffusion type)
  - **z-distribution violin plot** showing spatial distribution over time
  - **z-distribution heatmap** (2D density plot)
- ✅ **Extended test suite**
  - `test_plot_formatting.py`: 11 new tests for plot formatting
  - Validates SVG output, color schemes, and formatting rules

### Changed
- 🔄 **Time series analysis** now generates 30+ plots (vs. 6 previously)
- 🔄 **Plot output structure** expanded with subdirectories:
  - `timeSeries/D_boxplots/` - 4 boxplots (one per diffusion type)
  - `timeSeries/D_plots/` - 8 plots (log/linear × 4 types)
  - `timeSeries/alpha_plots/` - 8 plots (log/linear × 4 types)
  - `timeSeries/z_distribution_*.svg` - 2 spatial analysis plots
- 🔄 **Consistent color scheme** across all modules:
  - Blue (#1f77b4): Normal diffusion
  - Orange (#ff7f0e): Subdiffusion
  - Green (#2ca02c): Confined diffusion
  - Red (#d62728): Superdiffusion

### Improved
- 📈 **Better trend detection** with log-scale plots
- 📉 **Distribution analysis** with boxplots at each time point
- 🎨 **Professional aesthetics** suitable for publications
- 🔬 **z-axis analysis** reveals spatial confinement evolution

---

## [2.0.0] - 2025-01-13

### Added
- ✅ **Comprehensive test suite** with pytest
  - `test_z_correction.py`: Validates z-axis correction with 9 test cases
  - `test_tracking.py`: Validates 3D tracking, SNR calculation, and quality filtering
  - `test_time_series.py`: Validates time series analysis and output generation
- ✅ **Centralized logging framework** (`logger.py`)
  - Replaces blind `warnings.filterwarnings('ignore')`
  - Structured logging with timestamps and levels
  - Console and file output support
- ✅ **Dependencies management**
  - `requirements.txt` with version constraints
  - `setup.py` for package installation
  - `pytest.ini` for test configuration
- ✅ **CI/CD pipeline** (GitHub Actions)
  - Automated testing on Python 3.9, 3.10, 3.11, 3.12
  - Code quality checks (black, flake8, mypy)
  - Coverage reporting (Codecov)
- ✅ **Documentation improvements**
  - Completely rewritten README.md (English + comprehensive)
  - Added LICENSE (MIT)
  - Added CONTRIBUTING.md guidelines
  - Added CHANGELOG.md
- ✅ **Code quality improvements**
  - Type hints for critical functions
  - Logging integration in all main modules
  - Better error handling

### Changed
- 🔄 **Logging**: All modules now use centralized logger instead of print statements
- 🔄 **Warnings**: Only suppress DeprecationWarning and FutureWarning (not all warnings)

### Fixed
- 🐛 Fixed: z-correction now properly handles edge cases (NaN, zero background, etc.)

### Security
- 🔒 Added input validation for file paths and parameters

---

## [1.0.0] - 2024-11-11

### Added
- Initial release with core functionality:
  - 3D single-particle tracking (trackpy-based)
  - z-axis correction for refractive indices
  - Random Forest diffusion classifier (4 classes)
  - Clustering analysis (K-Means, DBSCAN, Hierarchical)
  - Time series analysis for polymerization studies
  - Modern tabbed GUI (`tracking_tool_gui.py`)
  - Batch processing support
  - SVG/Excel/HTML export

### Features
- 18 biophysical feature extraction
- Auto-optimization of tracking parameters
- Quality pre-filtering (SNR, uncertainty, chi-squared)
- Interactive 3D visualizations (Plotly)

---

## Future Roadmap

### [2.1.0] - Planned
- [ ] Sphinx documentation with API reference
- [ ] Example Jupyter notebooks
- [ ] Docker container for easy deployment
- [ ] Performance optimizations (multiprocessing for batch)

### [3.0.0] - Ideas
- [ ] Web interface (Flask/Streamlit)
- [ ] Support for additional input formats
- [ ] Real-time analysis mode
- [ ] Integration with ImageJ/Fiji

---

[2.0.0]: https://github.com/JulianLerch/3D_Astigmatism/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/JulianLerch/3D_Astigmatism/releases/tag/v1.0.0
