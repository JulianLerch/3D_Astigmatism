# 3D Single-Particle Tracking & Diffusion Analysis Suite

[![Tests](https://github.com/JulianLerch/3D_Astigmatism/actions/workflows/tests.yml/badge.svg)](https://github.com/JulianLerch/3D_Astigmatism/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python-based analysis tool for **3D single-particle tracking** and **diffusion classification** from astigmatism-based microscopy data.

---

## 🔬 Features

### Core Functionality
- **3D Particle Tracking**: Automated tracking using astigmatism-based localization (ThunderSTORM CSV format)
- **z-Axis Correction**: Refractive index correction for accurate 3D measurements (oil, glass, polymer)
- **Auto-Optimization**: Automatic parameter tuning (`search_range`, `memory`) via quality metric scanning
- **Quality Pre-filtering**: SNR, uncertainty, and chi-squared filtering

### Advanced Analysis
- **Machine Learning Classification**: Random Forest classifier for 4 diffusion types
  - Normal Diffusion (α = 1.0)
  - Subdiffusion (fBm, α < 1)
  - Confined Diffusion
  - Superdiffusion (α > 1)
- **Clustering Analysis**: K-Means, Hierarchical, DBSCAN with automatic cluster number determination
- **Time Series Analysis**: Track diffusion type evolution over polymerization time
- **18 Biophysical Features**: Alpha exponent, Hurst exponent, fractal dimension, kurtosis, and more

### Visualization & Export
- **High-Quality Plots**: SVG format for publication-ready figures
- **Excel Export**: One sheet per track with comprehensive metadata
- **Interactive HTML**: 3D rotatable plots (Plotly)
- **Batch Processing**: Process multiple datasets with polymerization times

---

## 📦 Installation

### Requirements
- Python 3.9 or higher
- See `requirements.txt` for all dependencies

### Quick Install
```bash
# Clone repository
git clone https://github.com/JulianLerch/3D_Astigmatism.git
cd 3D_Astigmatism

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

---

## 🚀 Quick Start

### 1. Single File Analysis (GUI)
```bash
python tracking_tool_gui.py
```

**Workflow:**
1. Load CSV (ThunderSTORM format with columns: `x [nm]`, `y [nm]`, `z [nm]`, `frame`)
2. Enable **Pre-Filter** (recommended: removes low-quality localizations)
3. Configure **z-Correction** parameters:
   - `n_oil`: Refractive index of immersion oil (default: 1.518)
   - `n_glass`: Refractive index of coverslip (default: 1.523)
   - `n_polymer`: Refractive index of sample medium (default: 1.33)
   - `NA`: Numerical aperture (default: 1.50)
   - `d_glass`: Coverslip thickness in nm (default: 170000)
4. Keep **Auto Mode** active (recommended) or set parameters manually
5. Click **Start Tracking**
6. Export results (SVG plots + Excel)

### 2. Batch Processing with Time Series
```bash
python tracking_tool_gui.py
```
1. Go to **Batch** tab
2. Add folders with different polymerization times
3. Enable **RF Classification** and/or **Clustering**
4. Run **Batch Analysis**
5. Run **Time Series Analysis** to see diffusion type evolution

---

## 📊 Output Structure

```
3D_Tracking_Results/
├── 01_Raw_Tracks/              # Black & white 3D trajectories (SVG)
├── 02_Time_Resolved_Tracks/    # Color-coded by time (plasma colormap)
├── 03_SNR_Tracks/              # Color-coded by signal-to-noise ratio
├── 04_Tracks/                  # Excel files (one sheet per track)
├── 05_Histogramm/              # z-position distributions
├── 06_Interactive/             # HTML interactive 3D plots (top 5 tracks)
├── 07_RF_Classification/       # Diffusion-type colored tracks
├── 08_RF_Analysis/             # Boxplots, distributions, statistics
├── 09_Clustering/              # Cluster-colored tracks
└── 10_Clustering_Analysis/     # PCA, silhouette scores, cluster metrics

timeSeries/
├── time_series_statistics.csv              # Summary statistics per time point
├── time_series_all_tracks.csv              # Combined track data
├── time_series_analysis.xlsx               # Excel export
├── track_counts_over_time.svg              # Track count evolution
├── diffusion_fractions_over_time.svg       # Legacy diffusion distribution
├── diffusion_fractions_area.svg            # **Stacked area chart** (NEW)
├── alpha_evolution.svg                     # Alpha exponent evolution
├── D_evolution.svg                         # Diffusion coefficient evolution
├── diffusion_heatmap.svg                   # Heatmap of diffusion types
├── overall_trends.svg                      # Summary trend plot
├── z_distribution_violin.svg               # **z-position violin plot** (NEW)
├── z_distribution_heatmap.svg              # **z-position heatmap** (NEW)
├── D_boxplots/                             # **Boxplots per diffusion type** (NEW)
│   ├── D_boxplot_normal.svg
│   ├── D_boxplot_subdiffusion.svg
│   ├── D_boxplot_confined.svg
│   └── D_boxplot_superdiffusion.svg
├── D_plots/                                # **Log/linear D plots** (NEW)
│   ├── D_linear_normal.svg
│   ├── D_log_normal.svg
│   ├── D_linear_subdiffusion.svg
│   ├── D_log_subdiffusion.svg
│   ├── D_linear_confined.svg
│   ├── D_log_confined.svg
│   ├── D_linear_superdiffusion.svg
│   └── D_log_superdiffusion.svg
└── alpha_plots/                            # **Log/linear alpha plots** (NEW)
    ├── alpha_linear_normal.svg
    ├── alpha_log_normal.svg
    ├── alpha_linear_subdiffusion.svg
    ├── alpha_log_subdiffusion.svg
    ├── alpha_linear_confined.svg
    ├── alpha_log_confined.svg
    ├── alpha_linear_superdiffusion.svg
    └── alpha_log_superdiffusion.svg
```

---

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_z_correction.py -v
```

**Critical Tests:**
- ✅ `test_z_correction.py`: Validates z-axis refractive index correction
- ✅ `test_tracking.py`: Validates 3D tracking and SNR calculation
- ✅ `test_time_series.py`: Validates time series folder creation and plots
- ✅ `test_plot_formatting.py`: Validates plot formatting and new plot functions

---

## 📊 Plot Formatting

All plots follow publication-ready formatting standards:

**Style Guidelines:**
- ✅ **No grid lines** - Clean, uncluttered visualizations
- ✅ **No titles** - Captions should be in figure legends/text
- ✅ **SVG vector format** - Scalable, high-quality output
- ✅ **Concise English legends** - Short, descriptive labels
- ✅ **Minimalist spines** - Only left and bottom axes visible
- ✅ **Consistent colors** - Blue (Normal), Orange (Subdiffusion), Green (Confined), Red (Superdiffusion)

**New Advanced Plots:**
- **Stacked Area Chart**: Shows diffusion type distribution evolution (fills between lines)
- **Boxplots**: Distribution of D coefficient at each time point, per diffusion type
- **Log/Linear Plots**: Both log and linear scale for D and alpha, enabling detection of trends across orders of magnitude
- **z-Distribution Analysis**: Violin plots and heatmaps showing spatial distribution evolution

---

## 📐 z-Axis Correction Formula

The z-correction compensates for refractive index mismatch:

```
z_corrected = z_apparent × f_base × f_NA × f_depth

where:
  f_base  = n_polymer / n_oil
  f_NA    = √(n_oil² - NA²) / √(n_polymer² - NA²)
  f_depth = 1 + (d_glass / z_apparent) × (1 - n_glass / n_polymer)
```

**Physical Interpretation:**
- `f_base`: Base scaling due to refractive index mismatch
- `f_NA`: Numerical aperture correction
- `f_depth`: Depth-dependent correction from coverslip

---

## 🤖 Machine Learning Model

### Pre-trained Random Forest Classifier
- **Model**: `rf_diffusion_classifier_20251111_113232.pkl`
- **Performance**: OOB Score = 1.0000, F1 Macro = 1.0000
- **Features**: 18 biophysical features extracted from trajectories
- **Classes**: 4 diffusion types (Normal, Subdiffusion, Confined, Superdiffusion)

### Top 3 Important Features:
1. Fractal Dimension (12.3%)
2. Alpha Exponent (11.8%)
3. Hurst Exponent (11.6%)

---

## 🛠️ Development

### Code Quality
```bash
# Format code
black *.py tests/

# Lint code
flake8 *.py tests/ --max-line-length=120

# Type checking
mypy --ignore-missing-imports *.py
```

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

---

## 📚 Documentation

- **User Guide**: [USER_GUIDE_20251111_113232.md](USER_GUIDE_20251111_113232.md)
- **API Documentation**: Coming soon (Sphinx)
- **Example Notebooks**: Coming soon (Jupyter)

---

## 📖 Citation

If you use this tool in your research, please cite:

```bibtex
@software{3d_astigmatism,
  author = {Lerch, Julian},
  title = {3D Single-Particle Tracking \& Diffusion Analysis Suite},
  year = {2025},
  url = {https://github.com/JulianLerch/3D_Astigmatism}
}
```

---

## 🔧 Troubleshooting

### Common Issues

**Q: No tracks found**
A: Try adjusting `search_range` (increase for slower diffusion) or reducing `min_track_length`

**Q: z-correction gives unrealistic values**
A: Verify refractive indices match your experimental setup (oil, coverslip, sample medium)

**Q: Time Series fails with "No RF results"**
A: Run Batch Analysis with **RF Classification enabled** first

**Q: ImportError for sklearn/trackpy**
A: Install all requirements: `pip install -r requirements.txt`

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **trackpy**: Single-particle tracking framework
- **scikit-learn**: Machine learning library
- **ThunderSTORM**: Localization microscopy software (input format)

---

## 📧 Contact

For questions, issues, or contributions:
- **GitHub Issues**: [https://github.com/JulianLerch/3D_Astigmatism/issues](https://github.com/JulianLerch/3D_Astigmatism/issues)
- **Author**: Julian Lerch

---

**Version**: 2.0.0
**Last Updated**: 2025-01-13
