"""
3D SINGLE-PARTICLE TRACKING TOOL (V2 Lite)
Funktionen: z‑Korrektur (Brechungsindizes) + Tracking + Raw/Time/SNR (SVG) + Excel + z‑Histogramm (SVG) + Interaktive HTML‑Tracks (Top 5)
Run: python V2/tracking_tool.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import trackpy as tp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

warnings.filterwarnings('ignore')

# RF Classification module
try:
    from rf_analysis import (
        load_rf_model,
        perform_rf_classification_on_tracks,
        export_rf_visualizations,
        export_rf_analysis
    )
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False

# Time Series Analysis module
try:
    from time_series_analysis import export_time_series_analysis
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False

# Core: I/O + preprocessing
def calculate_snr(df: pd.DataFrame) -> pd.Series:
    if not {'intensity [photon]', 'offset [photon]', 'bkgstd [photon]'}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    numerator = (pd.to_numeric(df['intensity [photon]'], errors='coerce') -
                 pd.to_numeric(df['offset [photon]'], errors='coerce'))
    denominator = pd.to_numeric(df['bkgstd [photon]'], errors='coerce').replace(0, np.nan)
    snr = numerator / denominator
    snr = snr.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    return snr


def apply_quality_prefilter(
    df: pd.DataFrame,
    uncertainty_max: float = 30.0,
    chi2_percentile: float = 95.0,
    snr_min_percentile: float = 10.0,
    snr_max_percentile: float = 99.0,
) -> tuple[pd.DataFrame, dict]:
    n_start = int(len(df))
    mask_all = pd.Series(True, index=df.index)
    if 'uncertainty [nm]' in df.columns:
        mask_unc = pd.to_numeric(df['uncertainty [nm]'], errors='coerce') <= float(uncertainty_max)
    else:
        mask_unc = mask_all
    if 'chi2' in df.columns and df['chi2'].notna().any():
        chi2_thr = pd.to_numeric(df.loc[mask_unc, 'chi2'], errors='coerce').quantile(chi2_percentile / 100.0)
        mask_chi2 = pd.to_numeric(df['chi2'], errors='coerce') <= chi2_thr
    else:
        mask_chi2 = mask_all
    if 'SNR' in df.columns and df['SNR'].notna().any():
        snr_min = df.loc[mask_unc & mask_chi2, 'SNR'].quantile(snr_min_percentile / 100.0)
        snr_max = df.loc[mask_unc & mask_chi2, 'SNR'].quantile(snr_max_percentile / 100.0)
        mask_snr = (df['SNR'] >= snr_min) & (df['SNR'] <= snr_max)
    else:
        mask_snr = mask_all
    mask_final = (mask_unc & mask_chi2 & mask_snr).fillna(False)
    df_filtered = df.loc[mask_final].copy()
    stats = {
        'n_start': n_start,
        'n_after_uncertainty': int(mask_unc.sum()),
        'n_after_chi2': int((mask_unc & mask_chi2).sum()),
        'n_final': int(mask_final.sum()),
        'removed_total': int(n_start - mask_final.sum()),
        'removed_percent': float(100 * (n_start - mask_final.sum()) / max(n_start, 1)),
    }
    return df_filtered, stats


def apply_z_correction_inplace(
    df: pd.DataFrame,
    n_oil: float,
    n_glass: float,
    n_polymer: float,
    NA: float = 1.50,
    d_glass_nm: float = 170000.0,
) -> bool:
    # Find z column
    z_col = None
    if 'z [nm]' in df.columns:
        z_col = 'z [nm]'
    elif 'z' in df.columns:
        z_col = 'z'
    if z_col is None:
        return False
    z_apparent = pd.to_numeric(df[z_col], errors='coerce').to_numpy()
    # Base scaling and NA term
    f_base = 1.0
    f_na = 1.0
    try:
        f_base = float(n_polymer) / float(n_oil)
        num = np.sqrt(max(0.0, float(n_oil)**2 - float(NA)**2))
        den_arg = float(n_polymer)**2 - float(NA)**2
        den = np.sqrt(max(0.0, den_arg))
        f_na = (num / den) if den > 0 else 1.0
        if not np.isfinite(f_na) or f_na <= 0:
            f_na = 1.0
    except Exception:
        f_base, f_na = 1.0, 1.0
    # Depth term with small epsilon to avoid div by zero
    eps = 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        f_depth = 1.0 + (float(d_glass_nm) / (z_apparent + eps)) * (1.0 - float(n_glass) / float(n_polymer))
    z_corrected = z_apparent * f_base * f_na * f_depth
    df[z_col] = z_corrected
    return True


def load_and_prepare_data(
    filepath: str | Path,
    apply_prefilter: bool = True,
    uncertainty_max: float = 30,
    chi2_percentile: float = 95,
    snr_min_percentile: float = 10,
    snr_max_percentile: float = 99,
    apply_zcorr: bool = True,
    n_oil: float = 1.518,
    n_glass: float = 1.523,
    n_polymer: float = 1.47,
    NA: float = 1.50,
    d_glass_nm: float = 170000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(filepath, low_memory=False)
    for c in ['intensity [photon]', 'offset [photon]', 'bkgstd [photon]', 'x [nm]', 'y [nm]', 'z [nm]']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # Apply z-correction before any mapping/filtering
    if apply_zcorr:
        apply_z_correction_inplace(df, n_oil=n_oil, n_glass=n_glass, n_polymer=n_polymer, NA=NA, d_glass_nm=d_glass_nm)
    df['SNR'] = calculate_snr(df)
    mapping = {'x [nm]': 'x', 'y [nm]': 'y', 'z [nm]': 'z', 'frame': 'frame'}
    df_track = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}).copy()
    for c in ['x', 'y', 'frame']:
        if c not in df_track.columns:
            raise ValueError("CSV requires at least 'x [nm]', 'y [nm]', 'frame'.")
    df_track['frame'] = pd.to_numeric(df_track['frame'], errors='coerce').round().astype('Int64')
    df_track = df_track.dropna(subset=['frame']).copy()
    df_track['frame'] = df_track['frame'].astype(int)
    req_cols = ['x', 'y', 'frame'] + (['z'] if 'z' in df_track.columns else [])
    df_track = df_track.dropna(subset=req_cols)
    stats = {}
    if apply_prefilter:
        df_pref, stats = apply_quality_prefilter(
            df,
            uncertainty_max=uncertainty_max,
            chi2_percentile=chi2_percentile,
            snr_min_percentile=snr_min_percentile,
            snr_max_percentile=snr_max_percentile,
        )
        df = df_pref
        df_track = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}).copy()
        df_track['frame'] = pd.to_numeric(df_track['frame'], errors='coerce').round().astype('Int64')
        df_track = df_track.dropna(subset=['frame']).copy()
        df_track['frame'] = df_track['frame'].astype(int)
        req_cols = ['x', 'y', 'frame'] + (['z'] if 'z' in df_track.columns else [])
        df_track = df_track.dropna(subset=req_cols)
    return df, df_track, stats


def find_valid_csv_in_directory(dir_path: str | Path) -> Path | None:
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    candidates = list(dir_path.glob('*.csv'))
    if not candidates:
        return None
    valid: list[tuple[Path, float]] = []
    for f in candidates:
        try:
            df_head = pd.read_csv(f, nrows=1, low_memory=False)
            if {'x [nm]', 'y [nm]', 'frame'}.issubset(df_head.columns):
                valid.append((f, f.stat().st_mtime))
        except Exception:
            continue
    if not valid:
        return None
    # choose the most recently modified valid CSV
    valid.sort(key=lambda t: t[1], reverse=True)
    return valid[0][0]


# Tracking + Auto-Mode helpers
def perform_3d_tracking(df_track: pd.DataFrame, search_range: float, memory: int, min_track_length: int):
    pos_cols = ['x', 'y', 'z'] if 'z' in df_track.columns else ['x', 'y']
    df_track = df_track.dropna(subset=pos_cols + ['frame'])
    tracks = tp.link_df(
        df_track,
        search_range=search_range,
        memory=memory,
        t_column='frame',
        pos_columns=pos_cols
    ).reset_index(drop=True)
    tracks = tp.filter_stubs(tracks, min_track_length).reset_index(drop=True)
    if tracks.empty:
        return tracks, dict(n_tracks=0, n_localizations=0, min_length=0, max_length=0, mean_length=0.0, median_length=0.0)
    lengths = tracks.groupby('particle').size().reset_index(name='count')
    stats = {
        'n_tracks': int(tracks['particle'].nunique()),
        'n_localizations': int(len(tracks)),
        'min_length': int(lengths['count'].min()),
        'max_length': int(lengths['count'].max()),
        'mean_length': float(lengths['count'].mean()),
        'median_length': float(lengths['count'].median()),
    }
    return tracks, stats


def link_tracks_no_filter(df_track: pd.DataFrame, search_range: float, memory: int) -> pd.DataFrame:
    pos_cols = ['x', 'y', 'z'] if 'z' in df_track.columns else ['x', 'y']
    df_track = df_track.dropna(subset=pos_cols + ['frame'])
    tracks = tp.link_df(
        df_track,
        search_range=search_range,
        memory=memory,
        t_column='frame',
        pos_columns=pos_cols
    ).reset_index(drop=True)
    return tracks


def step_stats_along_tracks(tracks: pd.DataFrame) -> dict:
    if tracks is None or tracks.empty:
        return {'n_steps': 0, 'median_step': np.nan, 'p90_step': np.nan, 'p95_step': np.nan, 'max_step': np.nan, 'steps': pd.Series(dtype=float)}
    dims = ['x', 'y'] + (['z'] if 'z' in tracks.columns else [])
    steps = []
    for _, g in tracks.groupby('particle'):
        g = g.sort_values('frame')
        if len(g) < 2:
            continue
        diffsq = None
        for c in dims:
            d = np.diff(g[c].values)
            diffsq = (d**2) if diffsq is None else diffsq + d**2
        steps.extend(np.sqrt(diffsq))
    steps = pd.Series(steps, dtype=float)
    if steps.empty:
        return {'n_steps': 0, 'median_step': np.nan, 'p90_step': np.nan, 'p95_step': np.nan, 'max_step': np.nan, 'steps': steps}
    return {'n_steps': int(steps.size), 'median_step': float(steps.median()), 'p90_step': float(steps.quantile(0.90)), 'p95_step': float(steps.quantile(0.95)), 'max_step': float(steps.max()), 'steps': steps}


def estimate_interframe_steps(df_track: pd.DataFrame, max_frames_eval: int = 200, max_points_per_frame: int = 200, random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    frames = np.sort(df_track['frame'].unique())
    if frames.size < 2:
        return np.array([], dtype=float)
    if frames.size > max_frames_eval:
        idxs = np.linspace(0, frames.size - 2, max_frames_eval, dtype=int)
        frames_pairs = [(int(frames[i]), int(frames[i] + 1)) for i in idxs]
    else:
        frames_pairs = [(int(f), int(f + 1)) for f in frames[:-1]]
    dims = ['x', 'y'] + (['z'] if 'z' in df_track.columns else [])
    steps = []
    for f0, f1 in frames_pairs:
        a = df_track[df_track['frame'] == f0][dims].to_numpy()
        b = df_track[df_track['frame'] == f1][dims].to_numpy()
        if a.size == 0 or b.size == 0:
            continue
        if a.shape[0] > max_points_per_frame:
            sel = rng.choice(a.shape[0], size=max_points_per_frame, replace=False)
            a = a[sel]
        diff = a[:, None, :] - b[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        dmin = np.sqrt(np.min(d2, axis=1))
        steps.extend(dmin.tolist())
    return np.asarray(steps, dtype=float)


def propose_auto_parameters(df_track: pd.DataFrame):
    steps = estimate_interframe_steps(df_track)
    if steps.size == 0 or not np.isfinite(steps).any():
        base_sr = 500.0
    else:
        s_med = np.nanmedian(steps)
        s_p90 = np.nanquantile(steps, 0.90)
        base_sr = float(max(3.0 * s_med, s_p90))
        base_sr = float(np.clip(base_sr, 50.0, 2000.0))
    factors = [0.7, 1.0, 1.3, 1.6]
    search_ranges = sorted(set([int(np.clip(base_sr * f, 50, 2000)) for f in factors]))
    memory_list = [0, 1, 2, 3]
    n_frames = int(df_track['frame'].nunique())
    min_length = int(np.clip(max(6, 0.02 * n_frames), 8, 20))
    return search_ranges, memory_list, min_length, base_sr


def scan_parameters_and_select(df_track: pd.DataFrame, min_track_length: int, progress=None):
    sr_list, mem_list, _, base_sr = propose_auto_parameters(df_track)
    best = None
    for sr in sr_list:
        for mem in mem_list:
            try:
                tr_raw = link_tracks_no_filter(df_track, sr, mem)
                tr_filt = tp.filter_stubs(tr_raw, max(3, min_track_length // 2)).reset_index(drop=True)
                if tr_filt.empty:
                    score = (-1, -1, -1)
                else:
                    lengths = tr_filt.groupby('particle').size().values
                    med_len = float(np.median(lengths)) if lengths.size else 0.0
                    n_tracks = int(len(lengths))
                    st = step_stats_along_tracks(tr_filt)
                    bad_ratio = float(np.mean(st['steps'] > (2.0 * base_sr))) if st['n_steps'] > 0 else 1.0
                    score = (med_len, -bad_ratio, n_tracks)
                if progress:
                    ml = f"{score[0]:.1f}" if score[0] >= 0 else "0.0"
                    nt = f"{score[2]}" if score[2] >= 0 else "0"
                    progress(f"Scan sr={sr} nm, mem={mem}: med_len={ml}, n={nt}")
                if best is None or score > best[0]:
                    best = (score, sr, mem)
            except Exception as e:
                if progress:
                    progress(f"Scan sr={sr} nm, mem={mem}: ERROR {e}")
    if best is None:
        return 500, 2
    return best[1], best[2]


# Output helpers: folders, plotting, excel
def create_output_structure(base_path: str | Path, include_rf: bool = False) -> dict[str, Path]:
    base_path = Path(base_path)
    folders = {
        'main': base_path / '3D_Tracking_Results',
        'raw': base_path / '3D_Tracking_Results' / '01_Raw_Tracks',
        'time': base_path / '3D_Tracking_Results' / '02_Time_Resolved_Tracks',
        'snr': base_path / '3D_Tracking_Results' / '03_SNR_Tracks',
        'excel': base_path / '3D_Tracking_Results' / '04_Tracks',
        'hist': base_path / '3D_Tracking_Results' / '05_Histogramm',
        'interactive': base_path / '3D_Tracking_Results' / '06_Interactive',
    }

    if include_rf:
        folders['rf_class'] = base_path / '3D_Tracking_Results' / '07_RF_Model_Classification'
        folders['rf_analysis'] = base_path / '3D_Tracking_Results' / '08_RF_Analysis'

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def set_axes_equal_from_data(ax, x, y, z):
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    mins = np.array([x.min(), y.min(), z.min()])
    maxs = np.array([x.max(), y.max(), z.max()])
    centers = (mins + maxs) / 2
    half = (maxs - mins).max() / 2
    if not np.isfinite(half) or half == 0:
        half = 1.0
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def plot_single_track_raw(track_data: pd.DataFrame, save_path: Path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = track_data['x'].values
    y = track_data['y'].values
    z = track_data['z'].values if 'z' in track_data.columns else np.zeros_like(x)
    ax.plot(x, y, z, color='black', linewidth=1.5, alpha=0.7)
    ax.scatter(x[0], y[0], z[0], color='green', s=50, marker='o', edgecolors='black', linewidths=1, zorder=10)
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, marker='o', edgecolors='black', linewidths=1, zorder=10)
    ax.set_xlabel('x / nm'); ax.set_ylabel('y / nm'); ax.set_zlabel('z / nm')
    set_axes_equal_from_data(ax, x, y, z)
    ax.grid(False); ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    plt.tight_layout(); plt.savefig(save_path, format='svg', bbox_inches='tight'); plt.close(fig)


def plot_single_track_time(track_data: pd.DataFrame, save_path: Path, integration_time_ms: float = 100):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = track_data['x'].values
    y = track_data['y'].values
    z = track_data['z'].values if 'z' in track_data.columns else np.zeros_like(x)
    frames = track_data['frame'].values
    time_s = frames * (integration_time_ms / 1000.0)
    max_time_s = time_s.max() - time_s.min()
    if max_time_s > 90:
        time_values = time_s / 60.0; time_label = 't / min'
    else:
        time_values = time_s; time_label = 't / s'
    norm_time = (time_values - time_values.min()) / (time_values.max() - time_values.min() + 1e-10)
    colors = cm.plasma(norm_time)
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=2, alpha=0.8)
    ax.scatter(x[0], y[0], z[0], color='green', s=80, marker='o', edgecolors='white', linewidths=2, zorder=10)
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=80, marker='o', edgecolors='white', linewidths=2, zorder=10)
    sm = cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=time_values.min(), vmax=time_values.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8); cbar.set_label(time_label)
    ax.set_xlabel('x / nm'); ax.set_ylabel('y / nm'); ax.set_zlabel('z / nm')
    set_axes_equal_from_data(ax, x, y, z)
    ax.grid(False); ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    plt.tight_layout(); plt.savefig(save_path, format='svg', bbox_inches='tight'); plt.close(fig)


def plot_single_track_snr(track_data: pd.DataFrame, save_path: Path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = track_data['x'].values
    y = track_data['y'].values
    z = track_data['z'].values if 'z' in track_data.columns else np.zeros_like(x)
    snr = track_data['SNR'].values if 'SNR' in track_data.columns else np.zeros_like(x)
    norm_snr = (snr - snr.min()) / (snr.max() - snr.min() + 1e-10)
    colors = cm.cividis(norm_snr)
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=2, alpha=0.8)
    ax.scatter(x[0], y[0], z[0], color='green', s=80, marker='o', edgecolors='white', linewidths=2, zorder=10)
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=80, marker='o', edgecolors='white', linewidths=2, zorder=10)
    sm = cm.ScalarMappable(cmap='cividis', norm=plt.Normalize(vmin=snr.min(), vmax=snr.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8); cbar.set_label('SNR')
    ax.set_xlabel('x / nm'); ax.set_ylabel('y / nm'); ax.set_zlabel('z / nm')
    set_axes_equal_from_data(ax, x, y, z)
    ax.grid(False); ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    plt.tight_layout(); plt.savefig(save_path, format='svg', bbox_inches='tight'); plt.close(fig)


def export_all_visualizations(tracks: pd.DataFrame, folders: dict, n_tracks_to_plot='all', integration_time_ms: float = 100, progress_callback=None):
    track_lengths_df = tracks.groupby('particle', as_index=False).size()
    track_lengths_df.columns = ['particle', 'count']
    track_lengths = dict(zip(track_lengths_df['particle'], track_lengths_df['count']))
    if n_tracks_to_plot == 'all':
        particle_ids = sorted(tracks['particle'].unique())
        plot_desc = "alle Tracks"
    else:
        sorted_tracks = sorted(track_lengths.items(), key=lambda x: x[1], reverse=True)
        particle_ids = sorted([p for p, _ in sorted_tracks[:n_tracks_to_plot]])
        plot_desc = f"the {n_tracks_to_plot} longest tracks"
    total_tracks = len(particle_ids)
    if progress_callback:
        progress_callback(f"Erstelle {plot_desc} ({total_tracks} SVG-Plots)...")
    for idx, particle_id in enumerate(particle_ids, 1):
        track_data = tracks[tracks['particle'] == particle_id].copy().sort_values('frame')
        base = f'Track_{int(particle_id):04d}'
        plot_single_track_raw(track_data, folders['raw'] / f'{base}.svg')
        plot_single_track_time(track_data, folders['time'] / f'{base}_time.svg', integration_time_ms=integration_time_ms)
        plot_single_track_snr(track_data, folders['snr'] / f'{base}_snr.svg')
        if progress_callback and (idx % 5 == 0 or idx == total_tracks):
            progress_callback(f"  {idx}/{total_tracks} fertig")
    if progress_callback:
        progress_callback("Visualisierungen erstellt (SVG)")


def export_to_excel(tracks: pd.DataFrame, output_path: Path, progress_callback=None):
    if progress_callback:
        progress_callback("Erstelle Excel-Datei...")
    tr = tracks.copy().reset_index(drop=True)
    tr = tr.rename(columns={'x': 'x [nm]', 'y': 'y [nm]', 'z': 'z [nm]'})
    with pd.ExcelWriter(output_path) as writer:
        lengths = tr.groupby('particle').size().reset_index(name='count')
        summary = {'Track ID': lengths['particle'].values, 'Length [frames]': lengths['count'].values}
        if 'SNR' in tr.columns:
            summary['Mean SNR'] = tr.groupby('particle')['SNR'].mean().values
            summary['Median SNR'] = tr.groupby('particle')['SNR'].median().values
        if 'intensity [photon]' in tr.columns:
            summary['Mean Intensity'] = tr.groupby('particle')['intensity [photon]'].mean().values
        if 'uncertainty [nm]' in tr.columns:
            summary['Mean Uncertainty'] = tr.groupby('particle')['uncertainty [nm]'].mean().values
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
        for pid in sorted(tr['particle'].unique()):
            sheet = f'Track_{int(pid)}'[:31]
            traj = tr[tr['particle'] == pid].copy().sort_values('frame')
            cols = ['frame', 'x [nm]', 'y [nm]'] + (['z [nm]'] if 'z [nm]' in traj.columns else [])
            extras = [c for c in ['SNR','intensity [photon]','uncertainty [nm]','sigma1 [nm]','sigma2 [nm]','offset [photon]','bkgstd [photon]'] if c in traj.columns]
            traj[cols + extras].to_excel(writer, sheet_name=sheet, index=False)
    if progress_callback:
        progress_callback("Excel-Datei erstellt")


def export_z_histogram(tracks: pd.DataFrame, save_path: Path, progress_callback=None):
    try:
        if 'z' not in tracks.columns:
            if progress_callback:
                progress_callback("Kein 'z' in Tracks – Histogramm übersprungen.")
            return
        z = pd.to_numeric(tracks['z'], errors='coerce').dropna().values
        if z.size == 0:
            if progress_callback:
                progress_callback("Keine z-Werte – Histogramm übersprungen.")
            return
        # Robust Bins (Freedman–Diaconis, Fallback Scott, sonst ~30 bins)
        z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
            bins = 30
        else:
            q25, q75 = np.quantile(z, [0.25, 0.75])
            iqr = q75 - q25
            n = z.size
            bin_width = 2 * iqr * (n ** (-1/3)) if iqr > 0 else 0
            if not np.isfinite(bin_width) or bin_width <= 0:
                sigma = np.nanstd(z)
                bin_width = 3.49 * sigma * (n ** (-1/3))
            if not np.isfinite(bin_width) or bin_width <= 0:
                bin_width = max(1.0, (z_max - z_min) / 30.0)
            bins = int(np.clip(np.ceil((z_max - z_min) / bin_width), 10, 80))
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.hist(z, bins=bins, color='#6E6E6E', edgecolor='black', linewidth=0.8)
        ax.set_xlabel('z / nm')
        ax.set_ylabel('Anzahl')
        ax.grid(False)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        if progress_callback:
            progress_callback(f"z-Histogramm erstellt → {save_path.name}")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Histogramm-Fehler: {e}")


def export_interactive_tracks(tracks: pd.DataFrame, folder: Path, n_longest: int = 5, integration_time_ms: float = 100, progress_callback=None):
    try:
        try:
            import plotly.graph_objects as go
        except Exception:
            if progress_callback:
                progress_callback("Plotly nicht installiert – interaktive Exporte übersprungen.")
            return
        g = tracks.copy().sort_values(['particle', 'frame'])
        if g.empty or 'particle' not in g.columns:
            if progress_callback:
                progress_callback("Keine Tracks für interaktive Exporte.")
            return
        lengths = g.groupby('particle').size().sort_values(ascending=False)
        top_ids = list(lengths.head(max(1, n_longest)).index)
        index_lines = ["<html><head><meta charset='utf-8'><title>Interactive Tracks</title></head><body>",
                       f"<h3>Längste {len(top_ids)} Tracks (interaktiv)</h3>", "<ul>"]
        for pid in top_ids:
            tr = g[g['particle'] == pid]
            x = tr['x'].values
            y = tr['y'].values
            z = tr['z'].values if 'z' in tr.columns else np.zeros_like(x)
            frames = tr['frame'].values.astype(float)
            t = frames * (integration_time_ms / 1000.0)
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                       mode='lines+markers',
                                       line=dict(color='#1f77b4', width=4),
                                       marker=dict(size=3, color=t, colorscale='Plasma', showscale=True,
                                                   colorbar=dict(title='t / s')),
                                       name=f'Track {int(pid)}'))
            # Start/End markers
            fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers',
                                       marker=dict(size=6, color='green'), name='Start'))
            fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers',
                                       marker=dict(size=6, color='red'), name='Ende'))
            fig.update_layout(scene=dict(xaxis_title='x / nm', yaxis_title='y / nm', zaxis_title='z / nm'),
                              template='plotly_white', margin=dict(l=0, r=0, t=30, b=0))
            out_path = folder / f"Track_{int(pid):04d}.html"
            fig.write_html(str(out_path), include_plotlyjs='cdn', full_html=True)
            index_lines.append(f"<li><a href='{out_path.name}'>Track {int(pid):04d}</a> (n={len(tr)})</li>")
        index_lines.append("</ul><p>Tipp: Mit Maus und Scrollrad zoomen/rotieren.</p></body></html>")
        (folder / 'index.html').write_text("\n".join(index_lines), encoding='utf-8')
        if progress_callback:
            progress_callback(f"Interaktive HTML-Tracks erstellt → {folder}")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Interactive-Export-Fehler: {e}")


def export_rf_classification_workflow(tracks: pd.DataFrame, base_dir: Path, n_tracks_to_plot='all',
                                      integration_time_ms: float = 100, progress_callback=None):
    """
    Perform full RF classification workflow: classify, visualize, and export analysis.

    Args:
        tracks: Tracked particles DataFrame
        base_dir: Base directory for output
        n_tracks_to_plot: Number of tracks to plot or 'all'
        integration_time_ms: Integration time in ms
        progress_callback: Optional progress callback
    """
    if not RF_AVAILABLE:
        if progress_callback:
            progress_callback("RF module not available - skipping RF classification")
        return

    try:
        # Load RF model
        if progress_callback:
            progress_callback("=" * 70)
            progress_callback("STARTING RF CLASSIFICATION")

        model, scaler, metadata = load_rf_model(Path.cwd())

        if model is None:
            if progress_callback:
                progress_callback("RF model not found - skipping RF classification")
                progress_callback("Expected files: rf_diffusion_classifier_*.pkl, feature_scaler_*.pkl, model_metadata_*.json")
            return

        if progress_callback:
            progress_callback("RF model loaded successfully")

        # Label names
        label_names = {
            0: 'Normal Diffusion',
            1: 'Subdiffusion (fBm)',
            2: 'Confined Diffusion',
            3: 'Superdiffusion'
        }

        # Classify all tracks
        dt = integration_time_ms / 1000.0  # Convert to seconds

        tracks_classified = perform_rf_classification_on_tracks(
            tracks=tracks,
            model=model,
            scaler=scaler,
            metadata=metadata,
            window_sizes=[10, 20, 30, 40, 50, 100, 150, 200],
            overlap=0.75,
            min_seg_length=10,
            dt=dt,
            progress_callback=progress_callback
        )

        # Create output structure
        folders = create_output_structure(base_dir, include_rf=True)

        # Export visualizations
        export_rf_visualizations(
            tracks_classified=tracks_classified,
            output_dir=folders['main'],
            label_names=label_names,
            n_tracks_to_plot=n_tracks_to_plot,
            progress_callback=progress_callback
        )

        # Export analysis
        export_rf_analysis(
            tracks_classified=tracks_classified,
            output_dir=folders['main'],
            label_names=label_names,
            dt=dt,
            progress_callback=progress_callback
        )

        if progress_callback:
            progress_callback("=" * 70)
            progress_callback("RF CLASSIFICATION COMPLETE!")
            progress_callback("  07_RF_Model_Classification/ (SVG plots by diffusion type)")
            progress_callback("  08_RF_Analysis/ (Excel, CSV, boxplots)")

    except Exception as e:
        if progress_callback:
            progress_callback(f"Error in RF classification workflow: {e}")
            import traceback
            progress_callback(traceback.format_exc())


class TrackingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("3D Single-Particle Tracking Tool (V2)")
        self.root.geometry("850x800")

        self.df_original: pd.DataFrame | None = None
        self.df_track: pd.DataFrame | None = None
        self.tracks: pd.DataFrame | None = None

        self.csv_filepath = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value=".")
        self.use_prefilter = tk.BooleanVar(value=True)
        self.auto_mode = tk.BooleanVar(value=True)
        self.search_range = tk.DoubleVar(value=500)
        self.memory = tk.IntVar(value=2)
        self.min_length = tk.IntVar(value=10)
        self.n_tracks_option = tk.StringVar(value="10")
        self.integration_time = tk.DoubleVar(value=100)
        self.use_rf_classification = tk.BooleanVar(value=RF_AVAILABLE)

        # Batch selection state (folder, polymerization_time)
        self.batch_dirs: list[tuple[Path, float]] = []

        # Refractive indices (GUI input)
        self.n_oil = tk.DoubleVar(value=1.518)
        self.n_glass = tk.DoubleVar(value=1.523)
        self.n_polymer = tk.DoubleVar(value=1.470)
        # Advanced optical params
        self.na = tk.DoubleVar(value=1.50)
        self.d_glass_nm = tk.DoubleVar(value=170000.0)

        self._build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _build_gui(self):
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        main = ttk.Frame(scrollable, padding="15")
        main.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        row = 0

        ttk.Label(main, text="3D Single-Particle Tracking Tool (V2)", font=('Arial', 18, 'bold')).grid(row=row, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        row += 1
        ttk.Label(main, text="Schritt 1: Daten laden", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1
        ttk.Label(main, text="CSV-Datei:").grid(row=row, column=0, sticky=tk.W, pady=5)
        csv_frame = ttk.Frame(main); csv_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Entry(csv_frame, textvariable=self.csv_filepath, width=45).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(csv_frame, text="...", command=self.browse_csv, width=4).pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        ttk.Label(main, text="Ausgabe-Ordner:").grid(row=row, column=0, sticky=tk.W, pady=5)
        out_frame = ttk.Frame(main); out_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Entry(out_frame, textvariable=self.output_dir, width=45).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_frame, text="...", command=self.browse_output, width=4).pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        # Refractive indices input
        ttk.Label(main, text="Brechungsindizes (n):", font=('Arial', 11, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(8, 2))
        ri_frame = ttk.Frame(main); ri_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(8, 2))
        ttk.Label(ri_frame, text="Öl:").pack(side=tk.LEFT)
        ttk.Entry(ri_frame, textvariable=self.n_oil, width=7).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(ri_frame, text="Glas:").pack(side=tk.LEFT)
        ttk.Entry(ri_frame, textvariable=self.n_glass, width=7).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(ri_frame, text="Polymer:").pack(side=tk.LEFT)
        ttk.Entry(ri_frame, textvariable=self.n_polymer, width=7).pack(side=tk.LEFT, padx=(4, 0))
        row += 1
        # Advanced optics
        ttk.Label(main, text="Optik (erweitert):", font=('Arial', 11, 'bold')).grid(row=row, column=0, sticky=tk.W)
        opt_frame = ttk.Frame(main); opt_frame.grid(row=row, column=1, sticky=(tk.W, tk.E))
        ttk.Label(opt_frame, text="NA:").pack(side=tk.LEFT)
        ttk.Entry(opt_frame, textvariable=self.na, width=7).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(opt_frame, text="d_glass [nm]:").pack(side=tk.LEFT)
        ttk.Entry(opt_frame, textvariable=self.d_glass_nm, width=9).pack(side=tk.LEFT, padx=(4, 0))
        row += 1
        # Batch folder selection UI
        ttk.Label(main, text="Batch-Verarbeitung: Ordnerliste", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 3))
        row += 1
        batch_frame = ttk.Frame(main)
        batch_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        # Listbox with scrollbar
        list_frame = ttk.Frame(batch_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.batch_listbox = tk.Listbox(list_frame, height=6, width=70, selectmode=tk.EXTENDED)
        self.batch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.batch_listbox.yview)
        lb_scroll.pack(side=tk.LEFT, fill='y')
        self.batch_listbox.config(yscrollcommand=lb_scroll.set)
        # Buttons
        btns = ttk.Frame(batch_frame)
        btns.pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Hinzufügen...", command=self.batch_add_dir).pack(fill=tk.X, pady=2)
        ttk.Button(btns, text="Entfernen", command=self.batch_remove_selected).pack(fill=tk.X, pady=2)
        ttk.Button(btns, text="Leeren", command=self.batch_clear).pack(fill=tk.X, pady=2)
        row += 1
        ttk.Checkbutton(main, text="Pre-Filter aktivieren (empfohlen)", variable=self.use_prefilter).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1
        self.load_btn = ttk.Button(main, text="Datei laden", command=self.load_data)
        self.load_btn.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        ttk.Separator(main, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        ttk.Label(main, text="Schritt 2: Tracking", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1
        ttk.Checkbutton(main, text="Automodus (empfohlen)", variable=self.auto_mode, command=self.update_auto_mode_ui).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1
        ttk.Label(main, text="Search Range [nm]:").grid(row=row, column=0, sticky=tk.W, pady=5)
        sr_frame = ttk.Frame(main); sr_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        self.search_scale = ttk.Scale(sr_frame, from_=50, to=2000, variable=self.search_range, orient=tk.HORIZONTAL, command=lambda v: self.search_label.config(text=f"{int(float(v))}"))
        self.search_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_label = ttk.Label(sr_frame, text="500", width=8, font=('Arial', 10, 'bold'))
        self.search_label.pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        ttk.Label(main, text="Memory [frames]:").grid(row=row, column=0, sticky=tk.W, pady=5)
        mem_frame = ttk.Frame(main); mem_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        self.memory_scale = ttk.Scale(mem_frame, from_=0, to=20, variable=self.memory, orient=tk.HORIZONTAL, command=lambda v: self.memory_label.config(text=f"{int(float(v))}"))
        self.memory_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.memory_label = ttk.Label(mem_frame, text="2", width=8, font=('Arial', 10, 'bold'))
        self.memory_label.pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        ttk.Label(main, text="Min. Track Length:").grid(row=row, column=0, sticky=tk.W, pady=5)
        len_frame = ttk.Frame(main); len_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        self.length_scale = ttk.Scale(len_frame, from_=2, to=50, variable=self.min_length, orient=tk.HORIZONTAL, command=lambda v: self.length_label.config(text=f"{int(float(v))}"))
        self.length_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.length_label = ttk.Label(len_frame, text="10", width=8, font=('Arial', 10, 'bold'))
        self.length_label.pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        ttk.Label(main, text="Integration Time [ms]:").grid(row=row, column=0, sticky=tk.W, pady=5)
        it_frame = ttk.Frame(main); it_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Entry(it_frame, textvariable=self.integration_time, width=10).pack(side=tk.LEFT)
        ttk.Label(it_frame, text="(for time axis in time plots)", foreground='#666', font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 0))
        row += 1
        ttk.Label(main, text="Tracks zum Plotten:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(main, textvariable=self.n_tracks_option, values=["5","10","20","50","all"], state="readonly", width=20).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        self.track_btn = ttk.Button(main, text="Tracking starten", command=self.run_tracking, state=tk.DISABLED)
        self.track_btn.grid(row=row, column=0, columnspan=2, pady=12)
        row += 1

        ttk.Separator(main, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        ttk.Label(main, text="Schritt 3: Export", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1

        # RF Classification checkbox
        if RF_AVAILABLE:
            rf_check = ttk.Checkbutton(main, text="RF Diffusion Classification aktivieren (empfohlen)", variable=self.use_rf_classification)
            rf_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1
        else:
            ttk.Label(main, text="RF Classification nicht verfügbar (rf_analysis.py fehlt)", foreground='#888', font=('Arial', 9)).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1

        self.export_btn = ttk.Button(main, text="Alles exportieren (SVG + Excel + RF)", command=self.export_all, state=tk.DISABLED)
        self.export_btn.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        # Batch start button
        self.batch_btn = ttk.Button(main, text="Batch starten (Track + Export für Ordnerliste)", command=self.run_batch)
        self.batch_btn.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Time Series button
        if TIMESERIES_AVAILABLE and RF_AVAILABLE:
            self.timeseries_btn = ttk.Button(main, text="Time Series Analyse (benötigt RF + Poly-Zeiten)", command=self.run_time_series, state=tk.NORMAL)
            self.timeseries_btn.grid(row=row, column=0, columnspan=2, pady=5)
            row += 1
        else:
            ttk.Label(main, text="Time Series nicht verfügbar (benötigt RF + time_series_analysis.py)", foreground='#888', font=('Arial', 9)).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
            row += 1

        ttk.Separator(main, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        ttk.Label(main, text="Status & Log:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1
        self.log_text = ScrolledText(main, height=12, width=95, wrap=tk.WORD, font=('Consolas', 9), state=tk.DISABLED)
        self.log_text.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        main.columnconfigure(1, weight=1)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.log("GUI loaded. Choose a ThunderSTORM CSV file.")
        self.log("Tip: Pre-filter + Auto mode give robust results.")
        # Apply initial state for auto mode
        self.update_auto_mode_ui()

    def update_auto_mode_ui(self):
        state = tk.DISABLED if self.auto_mode.get() else tk.NORMAL
        for w in [self.search_scale, self.search_label, self.memory_scale, self.memory_label, self.length_scale, self.length_label]:
            try:
                w.config(state=state)
            except Exception:
                pass

    def log(self, message: str):
        def _append():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _append)

    def browse_csv(self):
        filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            self.csv_filepath.set(filename)

    def browse_output(self):
        dirname = filedialog.askdirectory(title="Select output folder")
        if dirname:
            self.output_dir.set(dirname)

    def load_data(self):
        if not self.csv_filepath.get():
            messagebox.showerror("Error", "Please select a CSV file first!")
            return
        try:
            self.log("=" * 70)
            self.log(f"Lade Datei: {Path(self.csv_filepath.get()).name}")
            use_filter = bool(self.use_prefilter.get())
            if use_filter:
                self.log("Pre-Filter aktiviert...")
            self.df_original, self.df_track, filter_stats = load_and_prepare_data(
                self.csv_filepath.get(),
                apply_prefilter=use_filter,
                uncertainty_max=30,
                chi2_percentile=95,
                snr_min_percentile=10,
                snr_max_percentile=99,
                apply_zcorr=True,
                n_oil=self.n_oil.get(),
                n_glass=self.n_glass.get(),
                n_polymer=self.n_polymer.get(),
                NA=self.na.get(),
                d_glass_nm=self.d_glass_nm.get(),
            )
            self.log(f"z-Korrektur angewendet mit n_oil={self.n_oil.get():.3f}, n_glass={self.n_glass.get():.3f}, n_polymer={self.n_polymer.get():.3f}, NA={self.na.get():.2f}, d_glass={self.d_glass_nm.get():.0f} nm")
            if filter_stats:
                self.log(f"  Original: {filter_stats['n_start']:,} Lokalisierungen")
                self.log(f"  Nach Uncertainty: {filter_stats['n_after_uncertainty']:,}")
                self.log(f"  Nach chi2: {filter_stats['n_after_chi2']:,}")
                self.log(f"  Final: {filter_stats['n_final']:,} ({filter_stats['removed_percent']:.1f}% entfernt)")
            self.log("")
            self.log(f"Frames: {self.df_track['frame'].min()} bis {self.df_track['frame'].max()}")
            if 'SNR' in self.df_original.columns:
                self.log(f"SNR: {self.df_original['SNR'].mean():.2f} +/- {self.df_original['SNR'].std():.2f}")
            if 'uncertainty [nm]' in self.df_original.columns:
                self.log(f"Uncertainty (Median): {self.df_original['uncertainty [nm]'].median():.2f} nm")
            self.log("Ready for 3D tracking!")
            self.track_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden:\n{e}")
            self.log(f"Fehler: {e}")

    def run_tracking(self):
        def tracking_thread():
            try:
                self.log("=" * 70)
                self.log("STARTE 3D-TRACKING...")
                if self.auto_mode.get():
                    sr_list, mem_list, min_len_auto, base_sr = propose_auto_parameters(self.df_track)
                    min_len_use = max(int(self.min_length.get()), int(min_len_auto))
                    self.log(f"Auto-Vorschlag: base SR ~ {int(base_sr)} nm; SR {sr_list}; memory {mem_list}; min_len {min_len_use}")
                    best_sr, best_mem = scan_parameters_and_select(self.df_track, min_len_use, progress=self.log)
                    search_range_to_use = best_sr
                    memory_to_use = best_mem
                    self.root.after(0, lambda: self.search_range.set(best_sr))
                    self.root.after(0, lambda: self.memory.set(best_mem))
                    self.root.after(0, lambda: self.min_length.set(min_len_use))
                else:
                    search_range_to_use = float(self.search_range.get())
                    memory_to_use = int(self.memory.get())
                self.log("Parameter:")
                self.log(f"  Search Range: {int(search_range_to_use)} nm")
                self.log(f"  Memory: {memory_to_use} frames")
                self.log(f"  Min. length: {self.min_length.get()} frames")
                self.tracks, stats = perform_3d_tracking(
                    self.df_track,
                    search_range=search_range_to_use,
                    memory=memory_to_use,
                    min_track_length=self.min_length.get(),
                )
                self.log("Tracking abgeschlossen!")
                self.log("ERGEBNISSE:")
                self.log(f"  Tracks: {stats['n_tracks']}")
                self.log(f"  Lokalisierungen: {stats['n_localizations']:,}")
                self.log(f"  Lengths: {stats['min_length']}-{stats['max_length']} frames")
                self.log(f"        (avg {stats['mean_length']:.1f}, median {stats['median_length']:.0f})")
                self.log("Ready for export!")
                self.root.after(0, lambda: self.export_btn.config(state=tk.NORMAL))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Tracking error:\n{e}"))
                self.log(f"Error: {e}")

        self.track_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=tracking_thread, daemon=True)
        thread.start()

    def export_all(self):
        def export_thread():
            try:
                self.log("=" * 70)
                self.log("START EXPORT")
                use_rf = bool(self.use_rf_classification.get())
                folders = create_output_structure(self.output_dir.get(), include_rf=use_rf)
                self.log(f"Ordner: {folders['main']}")
                n_opt = self.n_tracks_option.get()
                n_tracks = 'all' if n_opt == 'all' else int(n_opt)
                export_all_visualizations(
                    self.tracks,
                    folders,
                    n_tracks_to_plot=n_tracks,
                    integration_time_ms=self.integration_time.get(),
                    progress_callback=self.log,
                )
                # Extra exports
                export_z_histogram(self.tracks, folders['hist'] / 'z_histogram.svg', progress_callback=self.log)
                export_interactive_tracks(self.tracks, folders['interactive'], n_longest=5, integration_time_ms=self.integration_time.get(), progress_callback=self.log)
                excel_path = folders['excel'] / 'all_trajectories.xlsx'
                export_to_excel(self.tracks, excel_path, progress_callback=self.log)

                # RF Classification if enabled
                if use_rf:
                    export_rf_classification_workflow(
                        tracks=self.tracks,
                        base_dir=Path(self.output_dir.get()),
                        n_tracks_to_plot=n_tracks,
                        integration_time_ms=self.integration_time.get(),
                        progress_callback=self.log
                    )

                self.log("=" * 70)
                self.log("EXPORT ABGESCHLOSSEN!")
                self.log(f"Ausgabe: {folders['main']}")
                self.log("  01_Raw_Tracks/ (SVG)")
                self.log("  02_Time_Resolved_Tracks/ (SVG)")
                self.log("  03_SNR_Tracks/ (SVG)")
                self.log("  04_Tracks/ (Excel)")
                self.log("  05_Histogramm/ (SVG)")
                self.log("  06_Interactive/ (HTML)")
                if use_rf:
                    self.log("  07_RF_Model_Classification/ (SVG - diffusion type plots)")
                    self.log("  08_RF_Analysis/ (Excel, CSV, boxplots)")
                self.log(f"Integration Time: {self.integration_time.get()} ms")
                folder_path = str(folders['main'])
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Export complete!\n\nFolder: {folder_path}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Export error:\n{e}"))
                self.log(f"Error: {e}")
                import traceback
                self.log(traceback.format_exc())

        self.export_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()

    # Batch UI actions
    def batch_add_dir(self):
        dirname = filedialog.askdirectory(title="Ordner für Batch hinzufügen")
        if dirname:
            p = Path(dirname)

            # Check if already added
            if any(folder == p for folder, _ in self.batch_dirs):
                messagebox.showwarning("Warnung", "Ordner bereits in der Liste!")
                return

            # Prompt for polymerization time
            poly_time_dialog = tk.Toplevel(self.root)
            poly_time_dialog.title("Polymerisationszeit eingeben")
            poly_time_dialog.geometry("400x150")
            poly_time_dialog.transient(self.root)
            poly_time_dialog.grab_set()

            ttk.Label(poly_time_dialog, text=f"Ordner: {p.name}", font=('Arial', 10, 'bold')).pack(pady=10)
            ttk.Label(poly_time_dialog, text="Polymerisationszeit (beliebige Einheit):").pack(pady=5)

            poly_entry = ttk.Entry(poly_time_dialog, width=20)
            poly_entry.pack(pady=5)
            poly_entry.insert(0, "0.0")
            poly_entry.focus()

            result = {"confirmed": False, "time": 0.0}

            def on_ok():
                try:
                    result["time"] = float(poly_entry.get())
                    result["confirmed"] = True
                    poly_time_dialog.destroy()
                except ValueError:
                    messagebox.showerror("Fehler", "Bitte gültige Zahl eingeben!")

            def on_cancel():
                poly_time_dialog.destroy()

            btn_frame = ttk.Frame(poly_time_dialog)
            btn_frame.pack(pady=10)
            ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Abbrechen", command=on_cancel).pack(side=tk.LEFT, padx=5)

            poly_time_dialog.wait_window()

            if result["confirmed"]:
                self.batch_dirs.append((p, result["time"]))
                self.batch_listbox.insert(tk.END, f"{p.name}  [t={result['time']:.2f}]")

    def batch_remove_selected(self):
        sel = list(self.batch_listbox.curselection())
        if not sel:
            return
        # remove from end to start to keep indices valid
        for idx in reversed(sel):
            self.batch_listbox.delete(idx)
            try:
                del self.batch_dirs[idx]
            except Exception:
                pass

    def batch_clear(self):
        self.batch_listbox.delete(0, tk.END)
        self.batch_dirs.clear()

    def run_batch(self):
        def batch_thread():
            try:
                if not self.batch_dirs:
                    messagebox.showerror("Fehler", "Bitte zuerst eine oder mehrere Ordner zur Batchliste hinzufügen.")
                    return
                self.log("=" * 70)
                self.log(f"STARTE BATCH für {len(self.batch_dirs)} Ordner...")
                use_filter = bool(self.use_prefilter.get())
                auto = bool(self.auto_mode.get())
                # Iterate directories
                for i, (folder, poly_time) in enumerate(self.batch_dirs, start=1):
                    try:
                        self.log("-" * 40)
                        self.log(f"[{i}/{len(self.batch_dirs)}] Ordner: {folder} [Polyzeit: {poly_time:.2f}]")
                        csv_path = find_valid_csv_in_directory(folder)
                        if not csv_path:
                            self.log("  Keine geeignete CSV gefunden (erwartet Spalten: x [nm], y [nm], frame)")
                            continue
                        self.log(f"  CSV: {csv_path.name}")
                        # Load and prepare
                        df_original, df_track, filter_stats = load_and_prepare_data(
                            csv_path,
                            apply_prefilter=use_filter,
                            uncertainty_max=30,
                            chi2_percentile=95,
                            snr_min_percentile=10,
                            snr_max_percentile=99,
                            apply_zcorr=True,
                            n_oil=self.n_oil.get(),
                            n_glass=self.n_glass.get(),
                            n_polymer=self.n_polymer.get(),
                            NA=self.na.get(),
                            d_glass_nm=self.d_glass_nm.get(),
                        )
                        self.log(f"    z-Korrektur: n_oil={self.n_oil.get():.3f}, n_glass={self.n_glass.get():.3f}, n_polymer={self.n_polymer.get():.3f}, NA={self.na.get():.2f}, d_glass={self.d_glass_nm.get():.0f} nm")
                        if filter_stats:
                            self.log(f"    Original: {filter_stats['n_start']:,} Lokalisierungen")
                            self.log(f"    Final: {filter_stats['n_final']:,} ({filter_stats['removed_percent']:.1f}% entfernt)")
                        # Parameters
                        if auto:
                            sr_list, mem_list, min_len_auto, base_sr = propose_auto_parameters(df_track)
                            min_len_use = max(int(self.min_length.get()), int(min_len_auto))
                            self.log(f"    Auto-Vorschlag: base SR ~ {int(base_sr)} nm; SR {sr_list}; memory {mem_list}; min_len {min_len_use}")
                            best_sr, best_mem = scan_parameters_and_select(df_track, min_len_use, progress=self.log)
                            search_range_to_use = best_sr
                            memory_to_use = best_mem
                        else:
                            search_range_to_use = float(self.search_range.get())
                            memory_to_use = int(self.memory.get())
                            min_len_use = int(self.min_length.get())
                        self.log("    Parameter:")
                        self.log(f"      Search Range: {int(search_range_to_use)} nm")
                        self.log(f"      Memory: {memory_to_use} frames")
                        self.log(f"      Min. length: {min_len_use} frames")
                        # Tracking
                        tracks, stats = perform_3d_tracking(
                            df_track,
                            search_range=search_range_to_use,
                            memory=memory_to_use,
                            min_track_length=min_len_use,
                        )
                        if stats['n_tracks'] == 0:
                            self.log("    Keine Tracks gefunden – überspringe Export.")
                            continue
                        self.log(f"    Ergebnis: Tracks={stats['n_tracks']}, Locs={stats['n_localizations']:,}, Längen {stats['min_length']}-{stats['max_length']} (avg {stats['mean_length']:.1f})")
                        # Export into this folder
                        use_rf = bool(self.use_rf_classification.get())
                        folders = create_output_structure(folder, include_rf=use_rf)
                        n_opt = self.n_tracks_option.get()
                        n_tracks = 'all' if n_opt == 'all' else int(n_opt)
                        export_all_visualizations(
                            tracks,
                            folders,
                            n_tracks_to_plot=n_tracks,
                            integration_time_ms=self.integration_time.get(),
                            progress_callback=self.log,
                        )
                        # Extra exports
                        export_z_histogram(tracks, folders['hist'] / 'z_histogram.svg', progress_callback=self.log)
                        export_interactive_tracks(tracks, folders['interactive'], n_longest=5, integration_time_ms=self.integration_time.get(), progress_callback=self.log)
                        excel_path = folders['excel'] / 'all_trajectories.xlsx'
                        export_to_excel(tracks, excel_path, progress_callback=self.log)

                        # RF Classification if enabled
                        if use_rf:
                            export_rf_classification_workflow(
                                tracks=tracks,
                                base_dir=folder,
                                n_tracks_to_plot=n_tracks,
                                integration_time_ms=self.integration_time.get(),
                                progress_callback=self.log
                            )

                        self.log(f"    Export OK → {folders['main']}")
                    except Exception as e:
                        self.log(f"    Fehler in Ordner {folder}: {e}")
                self.log("=" * 70)
                self.log("BATCH ABGESCHLOSSEN")
                messagebox.showinfo("Batch", "Batch-Verarbeitung abgeschlossen.")
            except Exception as e:
                messagebox.showerror("Batch-Fehler", f"Fehler in Batch-Verarbeitung:\n{e}")
        # run in background
        thread = threading.Thread(target=batch_thread, daemon=True)
        thread.start()

    def run_time_series(self):
        """Run time series analysis on batch folders."""
        def timeseries_thread():
            try:
                if not self.batch_dirs:
                    messagebox.showerror("Fehler", "Bitte zuerst Ordner zur Batchliste hinzufügen!")
                    return

                if not TIMESERIES_AVAILABLE:
                    messagebox.showerror("Fehler", "Time Series Analyse nicht verfügbar (time_series_analysis.py fehlt)")
                    return

                if not RF_AVAILABLE:
                    messagebox.showerror("Fehler", "Time Series Analyse benötigt RF Classification!")
                    return

                # Check if all folders have RF results
                missing_rf = []
                for folder, poly_time in self.batch_dirs:
                    rf_dir = folder / '3D_Tracking_Results' / '08_RF_Analysis'
                    if not (rf_dir / 'track_summary.csv').exists():
                        missing_rf.append(folder.name)

                if missing_rf:
                    msg = "Folgende Ordner haben keine RF-Analyse:\n\n" + "\n".join(missing_rf) + "\n\nBitte zuerst Batch mit RF-Classification durchführen!"
                    messagebox.showerror("Fehler", msg)
                    return

                # Ask for output directory
                self.log("=" * 70)
                self.log("TIME SERIES ANALYSE")
                self.log(f"Analysiere {len(self.batch_dirs)} Zeitpunkte...")

                output_dir = Path(self.output_dir.get())

                # Run analysis
                export_time_series_analysis(
                    folders_with_times=self.batch_dirs,
                    output_dir=output_dir,
                    progress_callback=self.log
                )

                self.log("=" * 70)
                self.log("TIME SERIES ANALYSE ABGESCHLOSSEN!")
                ts_path = output_dir / 'timeSeries'
                self.root.after(0, lambda: messagebox.showinfo("Erfolg", f"Time Series Analyse abgeschlossen!\n\nOrdner: {ts_path}"))

            except Exception as e:
                self.log(f"FEHLER: {e}")
                import traceback
                self.log(traceback.format_exc())
                messagebox.showerror("Fehler", f"Time Series Analyse fehlgeschlagen:\n{e}")

        thread = threading.Thread(target=timeseries_thread, daemon=True)
        thread.start()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Really quit the application?"):
            self.root.destroy()


def main():
    root = tk.Tk()
    app = TrackingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
