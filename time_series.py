#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Analysis Module - Enhanced Trajectory Analysis Pipeline V7.1

NEUE STRUKTUR:
- Before_Refit/ und After_Refit/ Hauptordner
- Alpha_Plots/, D_Plots/, Distributions/, Summary_Data/
- Flächendiagramme für Verteilung
- Trendlinien für Normal/Subdiffusion
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import logging
from config import *
from msd_analysis import _finite, _posfinite

logger = logging.getLogger(__name__)

# =====================================================
#          HELPER: BOXPLOT STATISTIKEN
# =====================================================

def _extract_boxplot_data(combined_df, times, class_name, parameter, class_col='Class'):
    """Extrahiert Boxplot-Daten für einen Parameter und eine Klasse.

    WICHTIG: Für D-Werte (Diffusionskoeffizient) erfolgt automatische
    Einheitenkonvertierung von µm²/s zu m²/s (Division durch 1e12).
    """
    x_indices = []
    data_to_plot = []
    time_labels = []

    for idx, time in enumerate(times):
        time_data = combined_df[combined_df['Polymerization_Time'] == time]
        class_data = time_data[time_data[class_col] == class_name]

        if not class_data.empty and parameter in class_data.columns:
            values = _finite(class_data[parameter])

            # Konvertiere D von µm²/s zu m²/s
            if parameter == 'D' and len(values) > 0:
                values = values / 1e12

            if len(values) > 0:
                x_indices.append(idx)
                data_to_plot.append(values)
                time_labels.append(f'{time:.0f}')

    return x_indices, data_to_plot, time_labels

def _add_boxplot_legend(ax):
    """
    Fügt eine Text-basierte Legende zum Boxplot hinzu.
    Erklärt alle Komponenten des Boxplots auf Englisch.
    Platziert oben rechts im Plot (kann per config.PLOT_SHOW_BOXPLOT_LEGEND deaktiviert werden).
    """
    if not PLOT_SHOW_BOXPLOT_LEGEND:
        return  # Legende abgeschaltet

    legend_text = (
        "Boxplot Components:\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "Box:       25th-75th percentile (IQR)\n"
        "Line:      Median (50th percentile)\n"
        "Whiskers:  1.5 × IQR range\n"
        "Red dots:  Outliers (max 5 top/bottom)"
    )

    # Textbox mit schöner Formatierung
    props = dict(boxstyle='round,pad=0.8', facecolor='wheat',
                 alpha=0.9, edgecolor='black', linewidth=1.5)

    # Oben rechts in der Ecke positionieren
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           horizontalalignment='right', bbox=props,
           family='monospace', linespacing=1.5)

def _calculate_outliers_limited(data, position, n_outliers=5):
    """
    Berechnet Ausreißer und limitiert auf die n extremsten oben und unten.

    Args:
        data: Array von Datenpunkten
        position: x-Position für den Plot
        n_outliers: Anzahl extremster Ausreißer pro Seite (default: 5)

    Returns:
        (outlier_positions, outlier_values): Arrays für Plotting
    """
    if len(data) < 4:
        return np.array([]), np.array([])

    # Berechne Quartile und IQR
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    iqr = q75 - q25

    # Whisker-Grenzen (1.5 * IQR)
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    # Finde Ausreißer
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    if len(outliers) == 0:
        return np.array([]), np.array([])

    # Teile in obere und untere Ausreißer
    upper_outliers = outliers[outliers > upper_bound]
    lower_outliers = outliers[outliers < lower_bound]

    # Sortiere und nehme die extremsten n
    selected_outliers = []

    if len(upper_outliers) > 0:
        # Sortiere absteigend und nimm die obersten n
        upper_sorted = np.sort(upper_outliers)[::-1]
        selected_outliers.extend(upper_sorted[:n_outliers])

    if len(lower_outliers) > 0:
        # Sortiere aufsteigend und nimm die untersten n
        lower_sorted = np.sort(lower_outliers)
        selected_outliers.extend(lower_sorted[:n_outliers])

    selected_outliers = np.array(selected_outliers)
    positions = np.full(len(selected_outliers), position)

    return positions, selected_outliers

def _fit_best_trend(x_data, y_data):
    """
    Probiert mehrere Funktionstypen für Trendlinien aus und wählt den besten Fit.

    Getestete Funktionen:
    1. Linear: y = a*x + b
    2. Exponentiell: y = a * exp(b*x)
    3. Exponentieller Abfall: y = a * exp(-b*x) + c
    4. Logarithmisch: y = a * log(x+1) + b
    5. Power-law: y = a * x^b
    6. Inverse: y = a / (x+1) + b
    7. Quadratisch: y = a*x^2 + b*x + c

    Args:
        x_data: X-Werte (z.B. Zeitpunkte)
        y_data: Y-Werte (z.B. Mediane)

    Returns:
        (best_fit_y, best_name, best_r2, best_params): Tupel mit Fit-Werten, Name, R² und Parametern
    """
    from scipy.optimize import curve_fit

    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    # Hilfsfunktion: R² berechnen
    def calculate_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    best_r2 = -np.inf
    best_fit_y = None
    best_name = "Linear"
    best_params = None

    # 1. LINEAR: y = a*x + b
    try:
        def linear(x, a, b):
            return a * x + b

        popt, _ = curve_fit(linear, x_data, y_data, maxfev=10000)
        y_fit = linear(x_data, *popt)
        r2 = calculate_r2(y_data, y_fit)

        if r2 > best_r2:
            best_r2 = r2
            best_fit_y = y_fit
            best_name = "Linear"
            best_params = popt
    except:
        pass

    # 2. EXPONENTIELL: y = a * exp(b*x)
    try:
        def exponential(x, a, b):
            return a * np.exp(b * x)

        # Initial guess basierend auf Daten
        p0 = [y_data[0] if y_data[0] > 0 else 1, 0.1]
        popt, _ = curve_fit(exponential, x_data, y_data, p0=p0, maxfev=10000)
        y_fit = exponential(x_data, *popt)
        r2 = calculate_r2(y_data, y_fit)

        if r2 > best_r2 and np.all(np.isfinite(y_fit)):
            best_r2 = r2
            best_fit_y = y_fit
            best_name = "Exponential"
            best_params = popt
    except:
        pass

    # 3. EXPONENTIELLER ABFALL: y = a * exp(-b*x) + c
    try:
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        y_min = np.min(y_data)
        y_max = np.max(y_data)
        p0 = [y_max - y_min, 0.1, y_min]
        popt, _ = curve_fit(exp_decay, x_data, y_data, p0=p0, maxfev=10000)
        y_fit = exp_decay(x_data, *popt)
        r2 = calculate_r2(y_data, y_fit)

        if r2 > best_r2 and np.all(np.isfinite(y_fit)):
            best_r2 = r2
            best_fit_y = y_fit
            best_name = "Exp. Decay"
            best_params = popt
    except:
        pass

    # 4. LOGARITHMISCH: y = a * log(x+1) + b
    try:
        def logarithmic(x, a, b):
            return a * np.log(x + 1) + b

        popt, _ = curve_fit(logarithmic, x_data, y_data, maxfev=10000)
        y_fit = logarithmic(x_data, *popt)
        r2 = calculate_r2(y_data, y_fit)

        if r2 > best_r2 and np.all(np.isfinite(y_fit)):
            best_r2 = r2
            best_fit_y = y_fit
            best_name = "Logarithmic"
            best_params = popt
    except:
        pass

    # 5. POWER-LAW: y = a * (x+1)^b
    try:
        def power_law(x, a, b):
            return a * np.power(x + 1, b)

        p0 = [y_data[0] if y_data[0] != 0 else 1, 0.5]
        popt, _ = curve_fit(power_law, x_data, y_data, p0=p0, maxfev=10000)
        y_fit = power_law(x_data, *popt)
        r2 = calculate_r2(y_data, y_fit)

        if r2 > best_r2 and np.all(np.isfinite(y_fit)):
            best_r2 = r2
            best_fit_y = y_fit
            best_name = "Power-law"
            best_params = popt
    except:
        pass

    # 6. INVERSE: y = a / (x+1) + b
    try:
        def inverse(x, a, b):
            return a / (x + 1) + b

        popt, _ = curve_fit(inverse, x_data, y_data, maxfev=10000)
        y_fit = inverse(x_data, *popt)
        r2 = calculate_r2(y_data, y_fit)

        if r2 > best_r2 and np.all(np.isfinite(y_fit)):
            best_r2 = r2
            best_fit_y = y_fit
            best_name = "Inverse"
            best_params = popt
    except:
        pass

    # 7. QUADRATISCH: y = a*x^2 + b*x + c
    try:
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        popt, _ = curve_fit(quadratic, x_data, y_data, maxfev=10000)
        y_fit = quadratic(x_data, *popt)
        r2 = calculate_r2(y_data, y_fit)

        if r2 > best_r2 and np.all(np.isfinite(y_fit)):
            best_r2 = r2
            best_fit_y = y_fit
            best_name = "Quadratic"
            best_params = popt
    except:
        pass

    # Fallback auf linearen Fit falls alle fehlschlagen
    if best_fit_y is None:
        slope, intercept = np.polyfit(x_data, y_data, 1)
        best_fit_y = slope * x_data + intercept
        best_name = "Linear (fallback)"
        best_r2 = calculate_r2(y_data, best_fit_y)
        best_params = [slope, intercept]

    return best_fit_y, best_name, best_r2, best_params

# =====================================================
#          ALPHA BOXPLOTS
# =====================================================

def plot_alpha_boxplot(combined_df, class_name, output_path, class_col='Class',
                       log_scale=False, add_trend=False):
    """
    Alpha Boxplot für eine Klasse mit optionaler Trendlinie.

    Args:
        combined_df: DataFrame mit Daten
        class_name: Klassenname
        output_path: Speicherpfad
        class_col: Spaltenname für Klasse
        log_scale: Log-Skala verwenden
        add_trend: Trendlinie hinzufügen (nur für Normal/Subdiffusion)
    """
    times = sorted(combined_df['Polymerization_Time'].unique())
    x_indices, data_to_plot, time_labels = _extract_boxplot_data(
        combined_df, times, class_name, 'Alpha', class_col
    )

    if not data_to_plot:
        logger.warning(f"Keine Alpha-Daten für {class_name}")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)

    # Platz für Legende rechts schaffen (Plot-Area auf 82% Breite reduzieren)
    plt.subplots_adjust(right=0.82)

    # Boxplot (ohne Standard-Fliers, wir plotten sie manuell)
    color = NEW_COLORS.get(class_name, 'gray')
    bp = ax.boxplot(data_to_plot, positions=x_indices, widths=0.6,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(linewidth=LINEWIDTH_FIT, color='black'),
                   whiskerprops=dict(linewidth=LINEWIDTH_TRACK),
                   capprops=dict(linewidth=LINEWIDTH_TRACK),
                   boxprops=dict(linewidth=LINEWIDTH_TRACK, facecolor=color, alpha=0.7))

    # Plotte limitierte Ausreißer manuell (max 5 oben und 5 unten pro Box)
    for idx, (pos, data) in enumerate(zip(x_indices, data_to_plot)):
        outlier_positions, outlier_values = _calculate_outliers_limited(np.array(data), pos, n_outliers=5)
        if len(outlier_values) > 0:
            ax.scatter(outlier_positions, outlier_values, color='red', s=30, alpha=0.6,
                      marker='o', zorder=3, edgecolors='darkred', linewidths=0.5)

    # Boxplot-Legende hinzufügen (oben rechts)
    _add_boxplot_legend(ax)

    # Trendlinie (nur für Normal/Subdiffusion) - nur als "Guide to the eye", ohne Label
    if add_trend and class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
        medians = [np.median(data) for data in data_to_plot]
        if len(x_indices) >= 2:
            # Verwende Multi-Funktions-Fitting für beste Trendlinie
            trend_line, fit_name, r2, params = _fit_best_trend(x_indices, medians)

            # Plotte ohne Label (nur visueller Guide)
            ax.plot(x_indices, trend_line, '--', color='darkred', linewidth=LINEWIDTH_FIT,
                   alpha=0.7, zorder=5)

    # Achsen
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel(r'$t_\mathrm{poly}$ / min', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$\alpha$ / [-]', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        title = f'{class_name} - Alpha vs Zeit'
        if log_scale:
            title += ' (log)'
        if add_trend:
            title += ' mit Trend'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(time_labels)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def create_all_alpha_plots(combined_df, output_folder, class_col='Class', classes=None):
    """Erstellt alle Alpha-Plots für alle Klassen."""
    os.makedirs(output_folder, exist_ok=True)

    if classes is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES

    for class_name in classes:
        safe_name = class_name.replace(' ', '_').replace('.', '')

        # Linear
        plot_alpha_boxplot(combined_df, class_name,
                          os.path.join(output_folder, f'alpha_linear_{safe_name}.svg'),
                          class_col, log_scale=False, add_trend=False)

        # Linear mit Trend (nur Normal/Subdiffusion)
        if class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
            plot_alpha_boxplot(combined_df, class_name,
                              os.path.join(output_folder, f'alpha_linear_{safe_name}_trend.svg'),
                              class_col, log_scale=False, add_trend=True)

        # Log
        plot_alpha_boxplot(combined_df, class_name,
                          os.path.join(output_folder, f'alpha_log_{safe_name}.svg'),
                          class_col, log_scale=True, add_trend=False)

        # Log mit Trend (nur Normal/Subdiffusion)
        if class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
            plot_alpha_boxplot(combined_df, class_name,
                              os.path.join(output_folder, f'alpha_log_{safe_name}_trend.svg'),
                              class_col, log_scale=True, add_trend=True)

    logger.info(f"✓ Alpha-Plots erstellt in {output_folder}")

# =====================================================
#          D BOXPLOTS
# =====================================================

def plot_d_boxplot(combined_df, class_name, output_path, class_col='Class',
                   log_scale=False, add_trend=False):
    """D Boxplot für eine Klasse mit optionaler Trendlinie."""
    times = sorted(combined_df['Polymerization_Time'].unique())
    x_indices, data_to_plot, time_labels = _extract_boxplot_data(
        combined_df, times, class_name, 'D', class_col
    )

    if not data_to_plot:
        logger.warning(f"Keine D-Daten für {class_name}")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)

    # Platz für Legende rechts schaffen (Plot-Area auf 82% Breite reduzieren)
    plt.subplots_adjust(right=0.82)

    # Boxplot (ohne Standard-Fliers, wir plotten sie manuell)
    color = NEW_COLORS.get(class_name, 'gray')
    bp = ax.boxplot(data_to_plot, positions=x_indices, widths=0.6,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(linewidth=LINEWIDTH_FIT, color='black'),
                   whiskerprops=dict(linewidth=LINEWIDTH_TRACK),
                   capprops=dict(linewidth=LINEWIDTH_TRACK),
                   boxprops=dict(linewidth=LINEWIDTH_TRACK, facecolor=color, alpha=0.7))

    # Plotte limitierte Ausreißer manuell (max 5 oben und 5 unten pro Box)
    for idx, (pos, data) in enumerate(zip(x_indices, data_to_plot)):
        outlier_positions, outlier_values = _calculate_outliers_limited(np.array(data), pos, n_outliers=5)
        if len(outlier_values) > 0:
            ax.scatter(outlier_positions, outlier_values, color='red', s=30, alpha=0.6,
                      marker='o', zorder=3, edgecolors='darkred', linewidths=0.5)

    # Boxplot-Legende hinzufügen (oben rechts)
    _add_boxplot_legend(ax)

    # Trendlinie (nur für Normal/Subdiffusion) - nur als "Guide to the eye", ohne Label
    if add_trend and class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
        medians = [np.median(data) for data in data_to_plot]
        if len(x_indices) >= 2:
            # Verwende Multi-Funktions-Fitting für beste Trendlinie
            trend_line, fit_name, r2, params = _fit_best_trend(x_indices, medians)

            # Plotte ohne Label (nur visueller Guide)
            ax.plot(x_indices, trend_line, '--', color='darkred', linewidth=LINEWIDTH_FIT,
                   alpha=0.7, zorder=5)

    # Achsen
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel(r'$t_\mathrm{poly}$ / min', fontsize=FONTSIZE_LABEL)

    # Y-Label: Für SUBDIFFUSION mit Alpha-Exponent, sonst Standard
    if class_name == 'SUBDIFFUSION':
        ax.set_ylabel(r'$D$ / (m$^2$ / s$^{\alpha}$)', fontsize=FONTSIZE_LABEL)
    else:
        ax.set_ylabel(r'$D$ / (m$^2$ / s)', fontsize=FONTSIZE_LABEL)

    if PLOT_SHOW_TITLE:
        title = f'{class_name} - D vs Zeit'
        if log_scale:
            title += ' (log)'
        if add_trend:
            title += ' mit Trend'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(time_labels)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def create_all_d_plots(combined_df, output_folder, class_col='Class', classes=None):
    """Erstellt alle D-Plots für alle Klassen."""
    os.makedirs(output_folder, exist_ok=True)

    if classes is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES

    for class_name in classes:
        safe_name = class_name.replace(' ', '_').replace('.', '')

        # Linear
        plot_d_boxplot(combined_df, class_name,
                      os.path.join(output_folder, f'd_linear_{safe_name}.svg'),
                      class_col, log_scale=False, add_trend=False)

        # Linear mit Trend (nur Normal/Subdiffusion)
        if class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
            plot_d_boxplot(combined_df, class_name,
                          os.path.join(output_folder, f'd_linear_{safe_name}_trend.svg'),
                          class_col, log_scale=False, add_trend=True)

        # Log
        plot_d_boxplot(combined_df, class_name,
                      os.path.join(output_folder, f'd_log_{safe_name}.svg'),
                      class_col, log_scale=True, add_trend=False)

        # Log mit Trend (nur Normal/Subdiffusion)
        if class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
            plot_d_boxplot(combined_df, class_name,
                          os.path.join(output_folder, f'd_log_{safe_name}_trend.svg'),
                          class_col, log_scale=True, add_trend=True)

    logger.info(f"✓ D-Plots erstellt in {output_folder}")

# =====================================================
#          FEATURE BOXPLOTS (18 Features)
# =====================================================

# Feature-Namen für Clustering/RF (aus unsupervised_clustering.py)
FEATURE_NAMES = [
    'Alpha', 'D', 'Hurst_Exponent', 'MSD_Ratio', 'MSD_Plateauness',
    'Convex_Hull_Area', 'Space_Exploration_Ratio', 'Mean_Cos_Theta',
    'Efficiency', 'Straightness', 'VACF_Lag1', 'VACF_Min',
    'Persistence_Length', 'Fractal_Dimension', 'Asphericity',
    'Kurtosis', 'RG_Saturation', 'Boundary_Proximity_Var',
    'Confinement_Probability'
]

# Schöne Namen und Einheiten für Plots
FEATURE_DISPLAY_NAMES = {
    'Alpha': (r'$\alpha$ / [-]', False),
    'D': (r'$D$ / (m$^2$ / s)', True),  # Konvertierung zu m²/s
    'Hurst_Exponent': (r'Hurst Exponent $H$ / [-]', False),
    'MSD_Ratio': (r'MSD Ratio $R(4,1)$ / [-]', False),
    'MSD_Plateauness': (r'MSD Plateauness / [-]', False),
    'Convex_Hull_Area': (r'Convex Hull Area / µm$^2$', False),
    'Space_Exploration_Ratio': (r'Space Exploration Ratio / [-]', False),
    'Mean_Cos_Theta': (r'Mean $\cos(\theta)$ / [-]', False),
    'Efficiency': (r'Efficiency / [-]', False),
    'Straightness': (r'Straightness / [-]', False),
    'VACF_Lag1': (r'VACF (lag=1) / (µm$^2$/s$^2$)', False),
    'VACF_Min': (r'VACF Min / (µm$^2$/s$^2$)', False),
    'Persistence_Length': (r'Persistence Length / µm', False),
    'Fractal_Dimension': (r'Fractal Dimension / [-]', False),
    'Asphericity': (r'Asphericity / [-]', False),
    'Kurtosis': (r'Kurtosis / [-]', False),
    'RG_Saturation': (r'$R_g$ Saturation / [-]', False),
    'Boundary_Proximity_Var': (r'Boundary Proximity Var / µm$^2$', False),
    'Confinement_Probability': (r'Confinement Probability / [-]', False)
}

def plot_feature_boxplot(combined_df, class_name, feature_name, output_path,
                         class_col='Class', log_scale=False, colors=None):
    """
    Feature Boxplot für eine Klasse.

    Args:
        combined_df: DataFrame mit Daten
        class_name: Klassenname
        feature_name: Feature-Name (z.B. 'Hurst_Exponent')
        output_path: Speicherpfad
        class_col: Spaltenname für Klasse
        log_scale: Log-Skala verwenden
        colors: Dict mit Farben
    """
    if feature_name not in combined_df.columns:
        logger.debug(f"Feature {feature_name} nicht in DataFrame vorhanden")
        return

    times = sorted(combined_df['Polymerization_Time'].unique())

    # Extrahiere Daten (wie bei Alpha/D)
    x_indices = []
    data_to_plot = []
    time_labels = []

    for idx, time in enumerate(times):
        time_data = combined_df[combined_df['Polymerization_Time'] == time]
        class_data = time_data[time_data[class_col] == class_name]

        if not class_data.empty and feature_name in class_data.columns:
            values = _finite(class_data[feature_name])

            # Konvertiere D von µm²/s zu m²/s wenn nötig
            ylabel, convert_d = FEATURE_DISPLAY_NAMES.get(feature_name, (feature_name, False))
            if convert_d and len(values) > 0:
                values = values / 1e12

            if len(values) > 0:
                x_indices.append(idx)
                data_to_plot.append(values)
                time_labels.append(f'{time:.0f}')

    if not data_to_plot:
        logger.debug(f"Keine Daten für {class_name} - {feature_name}")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)

    # Platz für Legende rechts schaffen
    plt.subplots_adjust(right=0.82)

    # Farbe bestimmen
    if colors is None:
        colors = NEW_COLORS
    color = colors.get(class_name, 'gray')

    # Boxplot
    bp = ax.boxplot(data_to_plot, positions=x_indices, widths=0.6,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(linewidth=LINEWIDTH_FIT, color='black'),
                   whiskerprops=dict(linewidth=LINEWIDTH_TRACK),
                   capprops=dict(linewidth=LINEWIDTH_TRACK),
                   boxprops=dict(linewidth=LINEWIDTH_TRACK, facecolor=color, alpha=0.7))

    # Limitierte Ausreißer
    for idx, (pos, data) in enumerate(zip(x_indices, data_to_plot)):
        outlier_positions, outlier_values = _calculate_outliers_limited(np.array(data), pos, n_outliers=5)
        if len(outlier_values) > 0:
            ax.scatter(outlier_positions, outlier_values, color='red', s=30, alpha=0.6,
                      marker='o', zorder=3, edgecolors='darkred', linewidths=0.5)

    # Boxplot-Legende
    _add_boxplot_legend(ax)

    # Achsen
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel(r'$t_\mathrm{poly}$ / min', fontsize=FONTSIZE_LABEL)

    # Y-Label mit Einheit
    ylabel, _ = FEATURE_DISPLAY_NAMES.get(feature_name, (feature_name, False))
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)

    if PLOT_SHOW_TITLE:
        title = f'{class_name} - {feature_name} vs Zeit'
        if log_scale:
            title += ' (log)'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(time_labels)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def create_all_feature_boxplots(combined_df, output_folder, class_col='Class',
                                classes=None, colors=None):
    """
    Erstellt Feature-Boxplots für alle Features und alle Klassen.

    Args:
        combined_df: DataFrame mit Daten
        output_folder: Output-Ordner
        class_col: Spaltenname für Klasse
        classes: Liste von Klassen
        colors: Dict mit Farben
    """
    os.makedirs(output_folder, exist_ok=True)

    if classes is None:
        classes = NEW_CLASSES

    if colors is None:
        colors = NEW_COLORS

    logger.info(f"Erstelle Feature-Boxplots für {len(FEATURE_NAMES)} Features...")

    # Für jedes Feature
    for feature_name in FEATURE_NAMES:
        if feature_name not in combined_df.columns:
            logger.debug(f"  Feature {feature_name} nicht verfügbar, überspringe")
            continue

        feature_folder = os.path.join(output_folder, f'{feature_name}_Plots')
        os.makedirs(feature_folder, exist_ok=True)

        # Für jede Klasse
        for class_name in classes:
            safe_class = class_name.replace(' ', '_').replace('.', '')
            safe_feature = feature_name.replace(' ', '_').replace('.', '')

            # Linear
            plot_feature_boxplot(combined_df, class_name, feature_name,
                               os.path.join(feature_folder, f'{safe_feature}_linear_{safe_class}.svg'),
                               class_col, log_scale=False, colors=colors)

            # Log (nur für positive Features)
            plot_feature_boxplot(combined_df, class_name, feature_name,
                               os.path.join(feature_folder, f'{safe_feature}_log_{safe_class}.svg'),
                               class_col, log_scale=True, colors=colors)

    logger.info(f"✓ Feature-Boxplots erstellt in {output_folder}")

# =====================================================
#          DISTRIBUTIONEN
# =====================================================

def plot_distribution_bars(combined_df, output_path, class_col='Class',
                           classes=None, colors=None, xml_track_counts=None):
    """
    Gestapeltes Balkendiagramm für Klassenverteilung mit Track-Anzahl.

    Args:
        combined_df: DataFrame mit Daten
        output_path: Speicherpfad
        class_col: Spaltenname für Klasse ('Class' oder 'Final_Class')
        classes: Liste von Klassen (default: OLD_CLASSES oder NEW_CLASSES)
        colors: Dict mit Farben (default: ORIGINAL_COLORS oder NEW_COLORS)
        xml_track_counts: Dict {time: n_tracks} für Annotation
    """
    if classes is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES
    if colors is None:
        colors = NEW_COLORS if class_col == 'Final_Class' else ORIGINAL_COLORS

    times = sorted(combined_df['Polymerization_Time'].unique())
    n_times = len(times)

    # Prozentuale Anteile berechnen
    class_percentages = {cls: [] for cls in classes}
    totals_per_time = []

    for time in times:
        time_data = combined_df[combined_df['Polymerization_Time'] == time]
        total = len(time_data)
        totals_per_time.append(total)

        for cls in classes:
            count = len(time_data[time_data[class_col] == cls])
            percentage = (count / total * 100) if total > 0 else 0
            class_percentages[cls].append(percentage)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 7))

    x_indices = np.arange(n_times)
    bar_width = 0.6
    bottoms = np.zeros(n_times)

    # Gestapelte Balken (jetzt in Prozent)
    for cls in classes:
        vals = np.array(class_percentages[cls])
        color = colors.get(cls, 'gray')
        bars = ax.bar(x_indices, vals, bar_width, bottom=bottoms,
                     label=cls, color=color, edgecolor='black',
                     linewidth=LINEWIDTH_SEGMENT, alpha=0.8)

        # Prozentangaben in Balken (wenn > 5%)
        for idx, (val, bottom) in enumerate(zip(vals, bottoms)):
            if val > 5:
                ax.text(idx, bottom + val/2, f'{val:.0f}%',
                       ha='center', va='center', fontweight='bold',
                       fontsize=FONTSIZE_TICK, color='white')

        bottoms += vals

    # Track-Anzahl über Balken (nur als Info, nicht Teil der y-Skala!)
    if xml_track_counts:
        for idx, time in enumerate(times):
            n_tracks = xml_track_counts.get(time)
            if n_tracks is not None:
                # Position bei 100% + kleiner Offset
                y_pos = 100 + 3
                ax.text(idx, y_pos, f'$n$={n_tracks}',
                       ha='center', va='bottom', fontsize=FONTSIZE_TICK,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.8))

    # Achsen (jetzt in Prozent!)
    ax.set_xlabel(r'$t_\mathrm{poly}$ / min', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Percentage / %', fontsize=FONTSIZE_LABEL)
    ax.set_ylim(0, 110)  # 0-100% plus Platz für n-Annotation
    if PLOT_SHOW_TITLE:
        title = 'Klassenverteilung über Zeit'
        if class_col == 'Final_Class':
            title += ' (After Refit)'
        else:
            title += ' (Before Refit)'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{t:.0f}' for t in times])
    ax.tick_params(labelsize=FONTSIZE_TICK)
    # Legende außerhalb rechts
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def plot_distribution_area(combined_df, output_path, class_col='Class',
                           classes=None, colors=None):
    """
    Flächendiagramm für Klassenverteilung über Zeit.

    Args:
        combined_df: DataFrame mit Daten
        output_path: Speicherpfad
        class_col: Spaltenname für Klasse
        classes: Liste von Klassen
        colors: Dict mit Farben
    """
    if classes is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES
    if colors is None:
        colors = NEW_COLORS if class_col == 'Final_Class' else ORIGINAL_COLORS

    times = sorted(combined_df['Polymerization_Time'].unique())

    # Prozentuale Anteile berechnen
    percentages = {cls: [] for cls in classes}
    for time in times:
        time_data = combined_df[combined_df['Polymerization_Time'] == time]
        total = len(time_data)

        for cls in classes:
            count = len(time_data[time_data[class_col] == cls])
            pct = (count / total * 100) if total > 0 else 0
            percentages[cls].append(pct)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 7))

    x_indices = np.arange(len(times))

    # Gestapelte Flächen
    bottoms = np.zeros(len(times))
    for cls in classes:
        vals = np.array(percentages[cls])
        color = colors.get(cls, 'gray')
        ax.fill_between(x_indices, bottoms, bottoms + vals,
                        label=cls, color=color, alpha=0.7,
                        edgecolor='black', linewidth=LINEWIDTH_TRACK)
        bottoms += vals

    # Achsen
    ax.set_xlabel(r'$t_\mathrm{poly}$ / min', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Distribution / %', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        title = 'Klassenverteilung über Zeit (Fläche)'
        if class_col == 'Final_Class':
            title += ' (After Refit)'
        else:
            title += ' (Before Refit)'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{t:.0f}' for t in times])
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    # Legende außerhalb rechts
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def create_all_distribution_plots(combined_df, output_folder, class_col='Class',
                                  xml_track_counts=None, classes=None, colors=None):
    """Erstellt alle Distribution-Plots."""
    os.makedirs(output_folder, exist_ok=True)

    # Verwende übergebene Klassen/Farben, sonst Defaults
    if classes is None or colors is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES
        colors = NEW_COLORS if class_col == 'Final_Class' else ORIGINAL_COLORS

    # Balkendiagramm (colorblind)
    plot_distribution_bars(combined_df,
                          os.path.join(output_folder, 'distribution_colorblind.svg'),
                          class_col, classes, colors, xml_track_counts)

    # Flächendiagramm
    plot_distribution_area(combined_df,
                          os.path.join(output_folder, 'distribution_area.svg'),
                          class_col, classes, colors)

    logger.info(f"✓ Distribution-Plots erstellt in {output_folder}")

# =====================================================
#          HAUPTFUNKTION
# =====================================================

def create_time_series_analysis(all_fit_results, time_assignments, output_folder,
                                xml_track_counts=None, clustering_results=None,
                                rf_results=None, all_trajectories=None):
    """
    Erstellt vollständige Zeitreihen-Analyse mit Before/After/Clustering/RandomForest Struktur.

    Args:
        all_fit_results: dict {folder: fit_results_df}
        time_assignments: dict {folder: time}
        output_folder: Hauptordner für Output
        xml_track_counts: dict {folder: n_tracks} (optional)
        clustering_results: dict {folder: clustering_dict} (optional)
        rf_results: dict {folder: rf_dict} (optional)
        all_trajectories: dict {folder: trajectories} (optional, für Clustering/RF Alpha/D)
    """
    logger.info("Erstelle Zeitreihen-Analyse...")

    # Kombiniere alle Daten
    all_dfs = []
    for folder, fit_df in all_fit_results.items():
        if fit_df.empty:
            continue
        time = time_assignments[folder]
        df_copy = fit_df.copy()
        df_copy['Polymerization_Time'] = time
        df_copy['Folder'] = os.path.basename(folder)
        all_dfs.append(df_copy)

    if not all_dfs:
        logger.warning("Keine Daten für Zeitreihen-Analyse")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Track counts nach Zeit umrechnen
    xml_counts_by_time = None
    if xml_track_counts:
        xml_counts_by_time = {}
        for folder, n_tracks in xml_track_counts.items():
            time = time_assignments.get(folder)
            if time is not None:
                xml_counts_by_time[time] = n_tracks

    # BEFORE REFIT Ordner - Verwende Original D und Alpha Werte!
    before_folder = os.path.join(output_folder, 'Before_Refit')
    os.makedirs(before_folder, exist_ok=True)

    logger.info("  → Before Refit Analyse...")

    # Kopie mit Original-Werten erstellen (Before Refit nutzt D_original und Alpha_original)
    before_df = combined_df.copy()
    if 'D_original' in before_df.columns:
        before_df['D'] = before_df['D_original']
    if 'Alpha_original' in before_df.columns:
        before_df['Alpha'] = before_df['Alpha_original']

    # Alpha Plots (Before - mit Original-Werten)
    alpha_before = os.path.join(before_folder, 'Alpha_Plots')
    create_all_alpha_plots(before_df, alpha_before, class_col='Original_Class')

    # D Plots (Before - mit Original-Werten)
    d_before = os.path.join(before_folder, 'D_Plots')
    create_all_d_plots(before_df, d_before, class_col='Original_Class')

    # Distributions (Before - mit Original-Werten)
    dist_before = os.path.join(before_folder, 'Distributions')
    create_all_distribution_plots(before_df, dist_before, class_col='Original_Class',
                                  xml_track_counts=xml_counts_by_time)

    # Summary (Before - mit Original-Werten)
    summary_before = os.path.join(before_folder, 'Summary_Data')
    os.makedirs(summary_before, exist_ok=True)
    before_df.to_csv(os.path.join(summary_before, 'time_series_before.csv'), index=False)

    # AFTER REFIT Ordner
    after_folder = os.path.join(output_folder, 'After_Refit')
    os.makedirs(after_folder, exist_ok=True)

    logger.info("  → After Refit Analyse...")

    # Alpha Plots (After)
    alpha_after = os.path.join(after_folder, 'Alpha_Plots')
    create_all_alpha_plots(combined_df, alpha_after, class_col='Final_Class')

    # D Plots (After)
    d_after = os.path.join(after_folder, 'D_Plots')
    create_all_d_plots(combined_df, d_after, class_col='Final_Class')

    # Distributions (After)
    dist_after = os.path.join(after_folder, 'Distributions')
    create_all_distribution_plots(combined_df, dist_after, class_col='Final_Class',
                                  xml_track_counts=xml_counts_by_time)

    # Summary (After)
    summary_after = os.path.join(after_folder, 'Summary_Data')
    os.makedirs(summary_after, exist_ok=True)
    combined_df.to_csv(os.path.join(summary_after, 'time_series_after.csv'), index=False)

    # CLUSTERING Ordner (wenn vorhanden)
    if clustering_results:
        logger.info("  → Clustering Analyse...")
        clustering_folder = os.path.join(output_folder, 'Clustering')
        os.makedirs(clustering_folder, exist_ok=True)

        # Clustering-Daten mit Alpha und D konvertieren
        from unsupervised_clustering import clustering_results_to_dataframe, CLUSTERING_CLASSES, CLUSTERING_COLORS

        clustering_dfs = []
        for folder, clust_results in clustering_results.items():
            if not clust_results:
                continue

            time = time_assignments.get(folder)
            if time is None:
                continue

            # Benötige Trajektorien für Alpha/D Berechnung
            trajectories = all_trajectories.get(folder) if all_trajectories else None
            if trajectories is None:
                logger.warning(f"Keine Trajektorien für Clustering Alpha/D in {folder}")
                continue

            # Konvertiere Clustering-Ergebnisse zu DataFrame (mit Alpha und D!)
            clustering_df = clustering_results_to_dataframe(clust_results, trajectories, DEFAULT_INT_TIME)
            if not clustering_df.empty:
                clustering_df['Polymerization_Time'] = time
                clustering_df['Folder'] = os.path.basename(folder)
                clustering_dfs.append(clustering_df)

        if clustering_dfs:
            # Kombiniere alle Clustering-Daten
            combined_clustering_df = pd.concat(clustering_dfs, ignore_index=True)

            # Summary Data
            clustering_summary_folder = os.path.join(clustering_folder, 'Summary_Data')
            os.makedirs(clustering_summary_folder, exist_ok=True)
            combined_clustering_df.to_csv(os.path.join(clustering_summary_folder, 'clustering_time_series.csv'),
                                         index=False)

            # Alpha Plots (Clustering mit CLUSTERING_CLASSES)
            alpha_clustering = os.path.join(clustering_folder, 'Alpha_Plots')
            create_all_alpha_plots(combined_clustering_df, alpha_clustering, class_col='Class', classes=CLUSTERING_CLASSES)

            # D Plots (Clustering mit CLUSTERING_CLASSES)
            d_clustering = os.path.join(clustering_folder, 'D_Plots')
            create_all_d_plots(combined_clustering_df, d_clustering, class_col='Class', classes=CLUSTERING_CLASSES)

            # Feature Boxplots (alle 18 Features)
            features_clustering = os.path.join(clustering_folder, 'Feature_Boxplots')
            create_all_feature_boxplots(combined_clustering_df, features_clustering, class_col='Class',
                                       classes=CLUSTERING_CLASSES, colors=CLUSTERING_COLORS)

            # Distributions (mit Clustering-Farben!)
            dist_clustering = os.path.join(clustering_folder, 'Distributions')
            create_all_distribution_plots(combined_clustering_df, dist_clustering, class_col='Class',
                                         xml_track_counts=xml_counts_by_time,
                                         classes=CLUSTERING_CLASSES, colors=CLUSTERING_COLORS)

            logger.info(f"  Clustering: {clustering_folder}")

    # RANDOM FOREST Ordner (wenn vorhanden)
    if rf_results:
        logger.info("  → Random Forest Analyse...")
        rf_folder = os.path.join(output_folder, 'RandomForest')
        os.makedirs(rf_folder, exist_ok=True)

        # RF-Daten mit Alpha und D konvertieren
        from random_forest_classification import rf_results_to_dataframe, RF_CLASSES, RF_COLORS

        rf_dfs = []
        for folder, rf_result_dict in rf_results.items():
            if not rf_result_dict:
                continue

            time = time_assignments.get(folder)
            if time is None:
                continue

            # Benötige Trajektorien für Alpha/D Berechnung
            trajectories = all_trajectories.get(folder) if all_trajectories else None
            if trajectories is None:
                logger.warning(f"Keine Trajektorien für RF Alpha/D in {folder}")
                continue

            # Konvertiere RF-Ergebnisse zu DataFrame (mit Alpha und D!)
            rf_df = rf_results_to_dataframe(rf_result_dict, trajectories, DEFAULT_INT_TIME)
            if not rf_df.empty:
                rf_df['Polymerization_Time'] = time
                rf_df['Folder'] = os.path.basename(folder)
                rf_dfs.append(rf_df)

        if rf_dfs:
            # Kombiniere alle RF-Daten
            combined_rf_df = pd.concat(rf_dfs, ignore_index=True)

            # Summary Data
            rf_summary_folder = os.path.join(rf_folder, 'Summary_Data')
            os.makedirs(rf_summary_folder, exist_ok=True)
            combined_rf_df.to_csv(os.path.join(rf_summary_folder, 'rf_time_series.csv'),
                                 index=False)

            # Alpha Plots (RF mit RF_CLASSES)
            alpha_rf = os.path.join(rf_folder, 'Alpha_Plots')
            create_all_alpha_plots(combined_rf_df, alpha_rf, class_col='Class', classes=RF_CLASSES)

            # D Plots (RF mit RF_CLASSES)
            d_rf = os.path.join(rf_folder, 'D_Plots')
            create_all_d_plots(combined_rf_df, d_rf, class_col='Class', classes=RF_CLASSES)

            # Feature Boxplots (alle 18 Features)
            features_rf = os.path.join(rf_folder, 'Feature_Boxplots')
            create_all_feature_boxplots(combined_rf_df, features_rf, class_col='Class',
                                       classes=RF_CLASSES, colors=RF_COLORS)

            # Distributions (mit RF-Farben!)
            dist_rf = os.path.join(rf_folder, 'Distributions')
            create_all_distribution_plots(combined_rf_df, dist_rf, class_col='Class',
                                         xml_track_counts=xml_counts_by_time,
                                         classes=RF_CLASSES, colors=RF_COLORS)

            logger.info(f"  Random Forest: {rf_folder}")

    logger.info("✓ Zeitreihen-Analyse abgeschlossen")
    logger.info(f"  Before Refit: {before_folder}")
    logger.info(f"  After Refit: {after_folder}")
    if clustering_results:
        logger.info(f"  Clustering: {os.path.join(output_folder, 'Clustering')}")
    if rf_results:
        logger.info(f"  Random Forest: {os.path.join(output_folder, 'RandomForest')}")

# =====================================================
#          GENERALISIERTE COMPARISON ANALYSE
# =====================================================

def create_comparison_analysis(all_fit_results, comparison_assignments, output_folder,
                                comparison_type='time_series', xml_track_counts=None,
                                clustering_results=None, rf_results=None, all_trajectories=None):
    """
    Generalisierte Funktion für Vergleichsanalysen (Time Series oder Dye Comparison).

    Args:
        all_fit_results: dict {folder: fit_results_df}
        comparison_assignments: dict {folder: value} - value ist entweder Zeit (float) oder Farbstoff-Name (str)
        output_folder: Hauptordner für Output
        comparison_type: 'time_series' oder 'dye_comparison'
        xml_track_counts: dict {folder: n_tracks} (optional)
        clustering_results: dict {folder: clustering_dict} (optional)
        rf_results: dict {folder: rf_dict} (optional)
        all_trajectories: dict {folder: trajectories} (optional)
    """
    if comparison_type == 'time_series':
        # Original Time Series Analyse
        return create_time_series_analysis(
            all_fit_results=all_fit_results,
            time_assignments=comparison_assignments,
            output_folder=output_folder,
            xml_track_counts=xml_track_counts,
            clustering_results=clustering_results,
            rf_results=rf_results,
            all_trajectories=all_trajectories
        )
    elif comparison_type == 'dye_comparison':
        # Dye Comparison Analyse (neue Logik)
        return create_dye_comparison_analysis(
            all_fit_results=all_fit_results,
            dye_assignments=comparison_assignments,
            output_folder=output_folder,
            xml_track_counts=xml_track_counts,
            clustering_results=clustering_results,
            rf_results=rf_results,
            all_trajectories=all_trajectories
        )
    else:
        raise ValueError(f"Unbekannter comparison_type: {comparison_type}")

def create_dye_comparison_analysis(all_fit_results, dye_assignments, output_folder,
                                    xml_track_counts=None, clustering_results=None,
                                    rf_results=None, all_trajectories=None):
    """
    Erstellt Farbstoff-Vergleichsanalyse (identisch zu Time Series, aber mit Farbstoff-Namen auf X-Achse).

    Args:
        all_fit_results: dict {folder: fit_results_df}
        dye_assignments: dict {folder: dye_name}
        output_folder: Hauptordner für Output
        xml_track_counts: dict {folder: n_tracks} (optional)
        clustering_results: dict {folder: clustering_dict} (optional)
        rf_results: dict {folder: rf_dict} (optional)
        all_trajectories: dict {folder: trajectories} (optional)
    """
    logger.info("Erstelle Farbstoff-Vergleichsanalyse...")

    # Kombiniere alle Daten (identisch zu Time Series, aber mit 'Dye_Name' statt 'Polymerization_Time')
    all_dfs = []
    for folder, fit_df in all_fit_results.items():
        if fit_df.empty:
            continue
        dye_name = dye_assignments[folder]
        df_copy = fit_df.copy()
        df_copy['Dye_Name'] = dye_name  # Statt 'Polymerization_Time'
        df_copy['Folder'] = os.path.basename(folder)
        all_dfs.append(df_copy)

    if not all_dfs:
        logger.warning("Keine Daten für Farbstoff-Vergleich")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Track counts nach Farbstoff umrechnen
    xml_counts_by_dye = None
    if xml_track_counts:
        xml_counts_by_dye = {}
        for folder, n_tracks in xml_track_counts.items():
            dye_name = dye_assignments.get(folder)
            if dye_name is not None:
                # Falls mehrere Ordner den gleichen Farbstoff haben, summiere die Tracks
                if dye_name in xml_counts_by_dye:
                    xml_counts_by_dye[dye_name] += n_tracks
                else:
                    xml_counts_by_dye[dye_name] = n_tracks

    # BEFORE REFIT
    before_folder = os.path.join(output_folder, 'Before_Refit')
    os.makedirs(before_folder, exist_ok=True)

    logger.info("  → Before Refit Analyse...")

    before_df = combined_df.copy()
    if 'D_original' in before_df.columns:
        before_df['D'] = before_df['D_original']
    if 'Alpha_original' in before_df.columns:
        before_df['Alpha'] = before_df['Alpha_original']

    # Alpha Plots - mit modifizierten Funktionen für Dye Namen
    alpha_before = os.path.join(before_folder, 'Alpha_Plots')
    create_all_alpha_plots_dye(before_df, alpha_before, class_col='Original_Class')

    # D Plots
    d_before = os.path.join(before_folder, 'D_Plots')
    create_all_d_plots_dye(before_df, d_before, class_col='Original_Class')

    # Distributions
    dist_before = os.path.join(before_folder, 'Distributions')
    create_all_distribution_plots_dye(before_df, dist_before, class_col='Original_Class',
                                      xml_track_counts=xml_counts_by_dye)

    # Summary
    summary_before = os.path.join(before_folder, 'Summary_Data')
    os.makedirs(summary_before, exist_ok=True)
    before_df.to_csv(os.path.join(summary_before, 'dye_comparison_before.csv'), index=False)

    # AFTER REFIT
    after_folder = os.path.join(output_folder, 'After_Refit')
    os.makedirs(after_folder, exist_ok=True)

    logger.info("  → After Refit Analyse...")

    alpha_after = os.path.join(after_folder, 'Alpha_Plots')
    create_all_alpha_plots_dye(combined_df, alpha_after, class_col='Final_Class')

    d_after = os.path.join(after_folder, 'D_Plots')
    create_all_d_plots_dye(combined_df, d_after, class_col='Final_Class')

    dist_after = os.path.join(after_folder, 'Distributions')
    create_all_distribution_plots_dye(combined_df, dist_after, class_col='Final_Class',
                                      xml_track_counts=xml_counts_by_dye)

    summary_after = os.path.join(after_folder, 'Summary_Data')
    os.makedirs(summary_after, exist_ok=True)
    combined_df.to_csv(os.path.join(summary_after, 'dye_comparison_after.csv'), index=False)

    # CLUSTERING (falls vorhanden)
    if clustering_results:
        logger.info("  → Clustering Analyse...")
        clustering_folder = os.path.join(output_folder, 'Clustering')
        os.makedirs(clustering_folder, exist_ok=True)

        from unsupervised_clustering import clustering_results_to_dataframe, CLUSTERING_CLASSES, CLUSTERING_COLORS

        clustering_dfs = []
        for folder, clust_results in clustering_results.items():
            if not clust_results:
                continue

            dye_name = dye_assignments.get(folder)
            if dye_name is None:
                continue

            trajectories = all_trajectories.get(folder) if all_trajectories else None
            if trajectories is None:
                logger.warning(f"Keine Trajektorien für Clustering in {folder}")
                continue

            clustering_df = clustering_results_to_dataframe(clust_results, trajectories, DEFAULT_INT_TIME)
            if not clustering_df.empty:
                clustering_df['Dye_Name'] = dye_name
                clustering_df['Folder'] = os.path.basename(folder)
                clustering_dfs.append(clustering_df)

        if clustering_dfs:
            combined_clustering_df = pd.concat(clustering_dfs, ignore_index=True)

            clustering_summary_folder = os.path.join(clustering_folder, 'Summary_Data')
            os.makedirs(clustering_summary_folder, exist_ok=True)
            combined_clustering_df.to_csv(os.path.join(clustering_summary_folder, 'clustering_dye_comparison.csv'),
                                         index=False)

            alpha_clustering = os.path.join(clustering_folder, 'Alpha_Plots')
            create_all_alpha_plots_dye(combined_clustering_df, alpha_clustering, class_col='Class', classes=CLUSTERING_CLASSES)

            d_clustering = os.path.join(clustering_folder, 'D_Plots')
            create_all_d_plots_dye(combined_clustering_df, d_clustering, class_col='Class', classes=CLUSTERING_CLASSES)

            dist_clustering = os.path.join(clustering_folder, 'Distributions')
            create_all_distribution_plots_dye(combined_clustering_df, dist_clustering, class_col='Class',
                                             xml_track_counts=xml_counts_by_dye,
                                             classes=CLUSTERING_CLASSES, colors=CLUSTERING_COLORS)

            logger.info(f"  Clustering: {clustering_folder}")

    # RANDOM FOREST (falls vorhanden)
    if rf_results:
        logger.info("  → Random Forest Analyse...")
        rf_folder = os.path.join(output_folder, 'RandomForest')
        os.makedirs(rf_folder, exist_ok=True)

        from random_forest_classification import rf_results_to_dataframe, RF_CLASSES, RF_COLORS

        rf_dfs = []
        for folder, rf_result_dict in rf_results.items():
            if not rf_result_dict:
                continue

            dye_name = dye_assignments.get(folder)
            if dye_name is None:
                continue

            trajectories = all_trajectories.get(folder) if all_trajectories else None
            if trajectories is None:
                logger.warning(f"Keine Trajektorien für RF in {folder}")
                continue

            rf_df = rf_results_to_dataframe(rf_result_dict, trajectories, DEFAULT_INT_TIME)
            if not rf_df.empty:
                rf_df['Dye_Name'] = dye_name
                rf_df['Folder'] = os.path.basename(folder)
                rf_dfs.append(rf_df)

        if rf_dfs:
            combined_rf_df = pd.concat(rf_dfs, ignore_index=True)

            rf_summary_folder = os.path.join(rf_folder, 'Summary_Data')
            os.makedirs(rf_summary_folder, exist_ok=True)
            combined_rf_df.to_csv(os.path.join(rf_summary_folder, 'rf_dye_comparison.csv'),
                                 index=False)

            alpha_rf = os.path.join(rf_folder, 'Alpha_Plots')
            create_all_alpha_plots_dye(combined_rf_df, alpha_rf, class_col='Class', classes=RF_CLASSES)

            d_rf = os.path.join(rf_folder, 'D_Plots')
            create_all_d_plots_dye(combined_rf_df, d_rf, class_col='Class', classes=RF_CLASSES)

            dist_rf = os.path.join(rf_folder, 'Distributions')
            create_all_distribution_plots_dye(combined_rf_df, dist_rf, class_col='Class',
                                             xml_track_counts=xml_counts_by_dye,
                                             classes=RF_CLASSES, colors=RF_COLORS)

            logger.info(f"  Random Forest: {rf_folder}")

    logger.info("✓ Farbstoff-Vergleichsanalyse abgeschlossen")
    logger.info(f"  Before Refit: {before_folder}")
    logger.info(f"  After Refit: {after_folder}")
    if clustering_results:
        logger.info(f"  Clustering: {os.path.join(output_folder, 'Clustering')}")
    if rf_results:
        logger.info(f"  Random Forest: {os.path.join(output_folder, 'RandomForest')}")

# Helper-Funktionen für Dye Comparison (verwenden 'Dye_Name' statt 'Polymerization_Time')

def _extract_boxplot_data_dye(combined_df, dye_names, class_name, parameter, class_col='Class'):
    """Extrahiert Boxplot-Daten für einen Parameter und eine Klasse (Dye Comparison).

    WICHTIG: Für D-Werte (Diffusionskoeffizient) erfolgt automatische
    Einheitenkonvertierung von µm²/s zu m²/s (Division durch 1e12).
    """
    x_indices = []
    data_to_plot = []
    dye_labels = []

    for idx, dye_name in enumerate(dye_names):
        dye_data = combined_df[combined_df['Dye_Name'] == dye_name]
        class_data = dye_data[dye_data[class_col] == class_name]

        if not class_data.empty and parameter in class_data.columns:
            values = _finite(class_data[parameter])

            # Konvertiere D von µm²/s zu m²/s
            if parameter == 'D' and len(values) > 0:
                values = values / 1e12

            if len(values) > 0:
                x_indices.append(idx)
                data_to_plot.append(values)
                dye_labels.append(dye_name)

    return x_indices, data_to_plot, dye_labels

def plot_alpha_boxplot_dye(combined_df, class_name, output_path, class_col='Class',
                            log_scale=False, add_trend=False):
    """Alpha Boxplot für Dye Comparison - X-Achse zeigt Farbstoff-Namen."""
    dye_names = sorted(combined_df['Dye_Name'].unique())
    x_indices, data_to_plot, dye_labels = _extract_boxplot_data_dye(
        combined_df, dye_names, class_name, 'Alpha', class_col
    )

    if not data_to_plot:
        logger.warning(f"Keine Alpha-Daten für {class_name} (Dye Comparison)")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)

    # Platz für Legende rechts schaffen (Plot-Area auf 82% Breite reduzieren)
    plt.subplots_adjust(right=0.82)

    # Boxplot
    color = NEW_COLORS.get(class_name, 'gray')
    bp = ax.boxplot(data_to_plot, positions=x_indices, widths=0.6,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(linewidth=LINEWIDTH_FIT, color='black'),
                   whiskerprops=dict(linewidth=LINEWIDTH_TRACK),
                   capprops=dict(linewidth=LINEWIDTH_TRACK),
                   boxprops=dict(linewidth=LINEWIDTH_TRACK, facecolor=color, alpha=0.7))

    # Plotte limitierte Ausreißer manuell
    for idx, (pos, data) in enumerate(zip(x_indices, data_to_plot)):
        outlier_positions, outlier_values = _calculate_outliers_limited(np.array(data), pos, n_outliers=5)
        if len(outlier_values) > 0:
            ax.scatter(outlier_positions, outlier_values, color='red', s=30, alpha=0.6,
                      marker='o', zorder=3, edgecolors='darkred', linewidths=0.5)

    # Boxplot-Legende hinzufügen
    _add_boxplot_legend(ax)

    # Trendlinie (nur für Normal/Subdiffusion) - nur als Guide to the eye
    if add_trend and class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
        medians = [np.median(data) for data in data_to_plot]
        if len(x_indices) >= 2:
            trend_line, fit_name, r2, params = _fit_best_trend(x_indices, medians)
            ax.plot(x_indices, trend_line, '--', color='darkred', linewidth=LINEWIDTH_FIT,
                   alpha=0.7, zorder=5)

    # Achsen
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Dye', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$\alpha$ / [-]', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        title = f'{class_name} - Alpha vs Dye'
        if log_scale:
            title += ' (log)'
        if add_trend:
            title += ' with trend'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(dye_labels, rotation=45, ha='right')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def plot_d_boxplot_dye(combined_df, class_name, output_path, class_col='Class',
                        log_scale=False, add_trend=False):
    """D Boxplot für Dye Comparison - X-Achse zeigt Farbstoff-Namen."""
    dye_names = sorted(combined_df['Dye_Name'].unique())
    x_indices, data_to_plot, dye_labels = _extract_boxplot_data_dye(
        combined_df, dye_names, class_name, 'D', class_col
    )

    if not data_to_plot:
        logger.warning(f"Keine D-Daten für {class_name} (Dye Comparison)")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)

    # Platz für Legende rechts schaffen (Plot-Area auf 82% Breite reduzieren)
    plt.subplots_adjust(right=0.82)

    # Boxplot
    color = NEW_COLORS.get(class_name, 'gray')
    bp = ax.boxplot(data_to_plot, positions=x_indices, widths=0.6,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(linewidth=LINEWIDTH_FIT, color='black'),
                   whiskerprops=dict(linewidth=LINEWIDTH_TRACK),
                   capprops=dict(linewidth=LINEWIDTH_TRACK),
                   boxprops=dict(linewidth=LINEWIDTH_TRACK, facecolor=color, alpha=0.7))

    # Plotte limitierte Ausreißer manuell
    for idx, (pos, data) in enumerate(zip(x_indices, data_to_plot)):
        outlier_positions, outlier_values = _calculate_outliers_limited(np.array(data), pos, n_outliers=5)
        if len(outlier_values) > 0:
            ax.scatter(outlier_positions, outlier_values, color='red', s=30, alpha=0.6,
                      marker='o', zorder=3, edgecolors='darkred', linewidths=0.5)

    # Boxplot-Legende hinzufügen
    _add_boxplot_legend(ax)

    # Trendlinie (nur für Normal/Subdiffusion) - nur als Guide to the eye
    if add_trend and class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
        medians = [np.median(data) for data in data_to_plot]
        if len(x_indices) >= 2:
            trend_line, fit_name, r2, params = _fit_best_trend(x_indices, medians)
            ax.plot(x_indices, trend_line, '--', color='darkred', linewidth=LINEWIDTH_FIT,
                   alpha=0.7, zorder=5)

    # Achsen
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Dye', fontsize=FONTSIZE_LABEL)

    # Y-Label: Für SUBDIFFUSION mit Alpha-Exponent, sonst Standard
    if class_name == 'SUBDIFFUSION':
        ax.set_ylabel(r'$D$ / (m$^2$ / s$^{\alpha}$)', fontsize=FONTSIZE_LABEL)
    else:
        ax.set_ylabel(r'$D$ / (m$^2$ / s)', fontsize=FONTSIZE_LABEL)

    if PLOT_SHOW_TITLE:
        title = f'{class_name} - D vs Dye'
        if log_scale:
            title += ' (log)'
        if add_trend:
            title += ' with trend'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(dye_labels, rotation=45, ha='right')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def create_all_alpha_plots_dye(combined_df, output_folder, class_col='Class', classes=None):
    """Alpha Plots für Dye Comparison - X-Achse zeigt Farbstoff-Namen"""
    os.makedirs(output_folder, exist_ok=True)

    if classes is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES

    for class_name in classes:
        safe_name = class_name.replace(' ', '_').replace('.', '')

        # Linear
        plot_alpha_boxplot_dye(combined_df, class_name,
                               os.path.join(output_folder, f'alpha_linear_{safe_name}.svg'),
                               class_col, log_scale=False, add_trend=False)

        # Linear mit Trend (nur Normal/Subdiffusion)
        if class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
            plot_alpha_boxplot_dye(combined_df, class_name,
                                   os.path.join(output_folder, f'alpha_linear_{safe_name}_trend.svg'),
                                   class_col, log_scale=False, add_trend=True)

        # Log
        plot_alpha_boxplot_dye(combined_df, class_name,
                               os.path.join(output_folder, f'alpha_log_{safe_name}.svg'),
                               class_col, log_scale=True, add_trend=False)

        # Log mit Trend (nur Normal/Subdiffusion)
        if class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
            plot_alpha_boxplot_dye(combined_df, class_name,
                                   os.path.join(output_folder, f'alpha_log_{safe_name}_trend.svg'),
                                   class_col, log_scale=True, add_trend=True)

    logger.info(f"✓ Alpha-Plots für Farbstoff-Vergleich erstellt in {output_folder}")

def create_all_d_plots_dye(combined_df, output_folder, class_col='Class', classes=None):
    """D Plots für Dye Comparison - X-Achse zeigt Farbstoff-Namen"""
    os.makedirs(output_folder, exist_ok=True)

    if classes is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES

    for class_name in classes:
        safe_name = class_name.replace(' ', '_').replace('.', '')

        # Linear
        plot_d_boxplot_dye(combined_df, class_name,
                           os.path.join(output_folder, f'd_linear_{safe_name}.svg'),
                           class_col, log_scale=False, add_trend=False)

        # Linear mit Trend (nur Normal/Subdiffusion)
        if class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
            plot_d_boxplot_dye(combined_df, class_name,
                               os.path.join(output_folder, f'd_linear_{safe_name}_trend.svg'),
                               class_col, log_scale=False, add_trend=True)

        # Log
        plot_d_boxplot_dye(combined_df, class_name,
                           os.path.join(output_folder, f'd_log_{safe_name}.svg'),
                           class_col, log_scale=True, add_trend=False)

        # Log mit Trend (nur Normal/Subdiffusion)
        if class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION']:
            plot_d_boxplot_dye(combined_df, class_name,
                               os.path.join(output_folder, f'd_log_{safe_name}_trend.svg'),
                               class_col, log_scale=True, add_trend=True)

    logger.info(f"✓ D-Plots für Farbstoff-Vergleich erstellt in {output_folder}")

def plot_distribution_bars_dye(combined_df, output_path, class_col='Class',
                                classes=None, colors=None, xml_track_counts=None):
    """Gestapeltes Balkendiagramm für Dye Comparison - X-Achse zeigt Farbstoff-Namen."""
    if classes is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES
    if colors is None:
        colors = NEW_COLORS if class_col == 'Final_Class' else ORIGINAL_COLORS

    dye_names = sorted(combined_df['Dye_Name'].unique())
    n_dyes = len(dye_names)

    # Prozentuale Anteile berechnen
    class_percentages = {cls: [] for cls in classes}
    totals_per_dye = []

    for dye_name in dye_names:
        dye_data = combined_df[combined_df['Dye_Name'] == dye_name]
        total = len(dye_data)
        totals_per_dye.append(total)

        for cls in classes:
            count = len(dye_data[dye_data[class_col] == cls])
            percentage = (count / total * 100) if total > 0 else 0
            class_percentages[cls].append(percentage)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 7))

    x_indices = np.arange(n_dyes)
    bar_width = 0.6
    bottoms = np.zeros(n_dyes)

    # Gestapelte Balken
    for cls in classes:
        vals = np.array(class_percentages[cls])
        color = colors.get(cls, 'gray')
        bars = ax.bar(x_indices, vals, bar_width, bottom=bottoms,
                     label=cls, color=color, edgecolor='black',
                     linewidth=LINEWIDTH_SEGMENT, alpha=0.8)

        # Prozentangaben in Balken (wenn > 5%)
        for idx, (val, bottom) in enumerate(zip(vals, bottoms)):
            if val > 5:
                ax.text(idx, bottom + val/2, f'{val:.0f}%',
                       ha='center', va='center', fontweight='bold',
                       fontsize=FONTSIZE_TICK, color='white')

        bottoms += vals

    # Track-Anzahl über Balken
    if xml_track_counts:
        for idx, dye_name in enumerate(dye_names):
            n_tracks = xml_track_counts.get(dye_name)
            if n_tracks is not None:
                y_pos = 100 + 3
                ax.text(idx, y_pos, f'$n$={n_tracks}',
                       ha='center', va='bottom', fontsize=FONTSIZE_TICK,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.8))

    # Achsen
    ax.set_xlabel('Dye', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Percentage / %', fontsize=FONTSIZE_LABEL)
    ax.set_ylim(0, 110)
    if PLOT_SHOW_TITLE:
        title = 'Class Distribution by Dye'
        if class_col == 'Final_Class':
            title += ' (After Refit)'
        else:
            title += ' (Before Refit)'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(dye_names, rotation=45, ha='right')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def plot_distribution_area_dye(combined_df, output_path, class_col='Class',
                                classes=None, colors=None):
    """Flächendiagramm für Dye Comparison - X-Achse zeigt Farbstoff-Namen."""
    if classes is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES
    if colors is None:
        colors = NEW_COLORS if class_col == 'Final_Class' else ORIGINAL_COLORS

    dye_names = sorted(combined_df['Dye_Name'].unique())

    # Prozentuale Anteile berechnen
    percentages = {cls: [] for cls in classes}
    for dye_name in dye_names:
        dye_data = combined_df[combined_df['Dye_Name'] == dye_name]
        total = len(dye_data)

        for cls in classes:
            count = len(dye_data[dye_data[class_col] == cls])
            pct = (count / total * 100) if total > 0 else 0
            percentages[cls].append(pct)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 7))

    x_indices = np.arange(len(dye_names))

    # Gestapelte Flächen
    bottoms = np.zeros(len(dye_names))
    for cls in classes:
        vals = np.array(percentages[cls])
        color = colors.get(cls, 'gray')
        ax.fill_between(x_indices, bottoms, bottoms + vals,
                        label=cls, color=color, alpha=0.7,
                        edgecolor='black', linewidth=LINEWIDTH_TRACK)
        bottoms += vals

    # Achsen
    ax.set_xlabel('Dye', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Distribution / %', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        title = 'Class Distribution by Dye (Area)'
        if class_col == 'Final_Class':
            title += ' (After Refit)'
        else:
            title += ' (Before Refit)'
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(dye_names, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def create_all_distribution_plots_dye(combined_df, output_folder, class_col='Class',
                                      xml_track_counts=None, classes=None, colors=None):
    """Distribution Plots für Dye Comparison - X-Achse zeigt Farbstoff-Namen"""
    os.makedirs(output_folder, exist_ok=True)

    if classes is None or colors is None:
        classes = NEW_CLASSES if class_col == 'Final_Class' else OLD_CLASSES
        colors = NEW_COLORS if class_col == 'Final_Class' else ORIGINAL_COLORS

    # Balkendiagramm
    plot_distribution_bars_dye(combined_df,
                               os.path.join(output_folder, 'distribution_colorblind.svg'),
                               class_col, classes, colors, xml_track_counts)

    # Flächendiagramm
    plot_distribution_area_dye(combined_df,
                               os.path.join(output_folder, 'distribution_area.svg'),
                               class_col, classes, colors)

    logger.info(f"✓ Distribution-Plots für Farbstoff-Vergleich erstellt in {output_folder}")
