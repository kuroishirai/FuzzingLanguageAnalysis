#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ4: Patch Coverage Analysis (Docker-ready, DB/utils preserved, relative paths)

- Reads per-project patch-coverage CSVs under ../data/RQ4_Patch_Coverage/projects/
- Computes per-project median patch coverage (filter: Matched_Lines > 0, project has >=10 entries)
- Runs Kruskal–Wallis (overall) then pairwise Mann–Whitney tests with Holm correction (only)
- Outputs exactly the following files under ../data/RQ4_Patch_Coverage/results/ :
    01_kruskal_result.csv
    pairwise_tests_summary_heatmaps_coverage.pdf   (Mann–Whitney + Holm)
    patch_coverage_distribution_filtered.pdf
    summary_statistics.csv

Notes
- Keeps DB/utils import for compatibility with your environment.
- All paths are relative to this script location.
- Comments and prints are in English (double-blind friendly).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

from scipy import stats
from statsmodels.stats.multitest import multipletests

# ==============================================================================
# Keep user-defined modules (DB/utils) and env path, but reference them relatively
# ==============================================================================
from configparser import ConfigParser

BASE_DIR = Path(__file__).resolve().parent                 # .../program/MSR2026/program
ROOT_DIR = BASE_DIR.parent                                 # .../program/MSR2026
MODULE_DIR = ROOT_DIR / "__module"                         # .../program/MSR2026/__module
CONFIG_PATH = MODULE_DIR / "envFile.ini"

print(f"[module_path] {MODULE_DIR}")
print(f"[config_path] {CONFIG_PATH}")

if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
try:
    import utils
    from dbFile import DB
except Exception as e:
    print(f"[WARN] Failed to import utils/dbFile from {MODULE_DIR}: {e}")
    utils = None
    DB = None  # Script can still run without DB

# ==============================================================================
# Small helpers
# ==============================================================================

def lighten_rgba(color: str, alpha: float = 0.7, factor: float = 0.6):
    """
    Make a given color slightly lighter with transparency.
    - color: '#rrggbb' etc.
    - alpha: 0..1
    - factor: brightness factor (smaller => lighter)
    """
    r, g, b, _ = mcolors.to_rgba(color)
    r = 1 - (1 - r) * factor
    g = 1 - (1 - g) * factor
    b = 1 - (1 - b) * factor
    return (r, g, b, alpha)

# ==============================================================================
# Data loading & aggregation
# ==============================================================================

def load_coverage_data(input_dir: Path) -> pd.DataFrame:
    """
    Load all per-project coverage CSVs from input_dir and concatenate them.
    Accepts either 'Covered_Lines' or 'Full_Covered' as coverage numerator.
    Skips empty/header-only files and those missing required columns.
    """
    csv_files = sorted(glob.glob(str(input_dir / "*.csv")))
    if not csv_files:
        print(f"[ERROR] No CSV found under: {input_dir}")
        return pd.DataFrame()

    print(f"[INFO] Loading {len(csv_files)} CSV files from: {input_dir}")
    df_list = []
    base_required_cols = {'Project', 'Language', 'Matched_Lines'}

    for f in tqdm(csv_files, desc="CSV Loading"):
        try:
            df = pd.read_csv(f)
            if df.empty:
                print(f"[INFO] Skip header-only/empty CSV: {Path(f).name}")
                continue

            # Basic columns
            if not base_required_cols.issubset(df.columns):
                print(f"[WARN] Missing required columns in {Path(f).name}, skip.")
                continue

            # Coverage numerator column normalization
            if 'Covered_Lines' in df.columns:
                pass  # as-is
            elif 'Full_Covered' in df.columns:
                df = df.rename(columns={'Full_Covered': 'Covered_Lines'})
            else:
                print(f"[WARN] No coverage column (Covered_Lines/Full_Covered): {Path(f).name}, skip.")
                continue

            df_list.append(df)

        except pd.errors.EmptyDataError:
            print(f"[INFO] Skip truly empty CSV: {Path(f).name}")
        except Exception as e:
            print(f"[WARN] Failed to load {Path(f).name}: {e}")

    if not df_list:
        print("[ERROR] No valid data loaded.")
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def calculate_project_median_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-project median coverage rate.
    - Filter to rows with Matched_Lines > 0
    - Keep only projects having >=10 valid entries
    Returns: columns [project, language, median_coverage]
    """
    print("[INFO] Calculating per-project median coverage...")

    df_filtered = df.loc[df['Matched_Lines'] > 0].copy()
    if df_filtered.empty:
        print("[WARN] No rows with Matched_Lines > 0.")
        return pd.DataFrame()

    df_filtered['Coverage_Rate'] = df_filtered['Covered_Lines'] / df_filtered['Matched_Lines']

    # Count valid entries per (Project, Language)
    counts = df_filtered.groupby(['Project', 'Language']).size().reset_index(name='valid_count')
    valid_projects = counts.loc[counts['valid_count'] >= 10, 'Project']
    df_filtered = df_filtered[df_filtered['Project'].isin(valid_projects)]

    print(f"[INFO] Valid projects: {df_filtered['Project'].nunique()} (Matched_Lines>0 & >=10 entries)")

    project_median = (
        df_filtered.groupby(['Project', 'Language'])['Coverage_Rate']
        .median()
        .reset_index()
        .rename(columns={'Project': 'project',
                         'Language': 'language',
                         'Coverage_Rate': 'median_coverage'})
    )
    return project_median

# ==============================================================================
# Statistics: KW then MW+Holm
# ==============================================================================

def run_kw_and_pairwise_mwholm(df_project_median: pd.DataFrame,
                               out_dir: Path,
                               alpha: float = 0.05,
                               display_language_map: Dict[str, str] | None = None) -> None:
    """
    1) Kruskal–Wallis on median_coverage across languages
    2) If significant, pairwise Mann–Whitney tests across all language pairs,
       adjust p-values using Holm method, save a lower-triangular heatmap PDF.

    Outputs:
      - 01_kruskal_result.csv
      - pairwise_tests_summary_heatmaps_coverage.pdf
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional label normalization, e.g., jvm -> java
    if display_language_map:
        df_project_median = df_project_median.copy()
        df_project_median['language'] = df_project_median['language'].replace(display_language_map)

    langs = sorted(df_project_median['language'].unique().tolist())
    if len(langs) < 2:
        print("[INFO] Fewer than 2 language groups. Skip statistical tests.")
        return

    # Prepare groups
    groups = [df_project_median.loc[df_project_median['language'] == lg, 'median_coverage'].dropna().values
              for lg in langs]

    # 1) Kruskal–Wallis
    h_stat, p_value = stats.kruskal(*groups)
    kw_df = pd.DataFrame([{"statistic": h_stat, "pvalue": p_value, "groups": ", ".join(langs)}])
    kw_path = out_dir / "01_kruskal_result.csv"
    kw_df.to_csv(kw_path, index=False)
    print(f"[INFO] KW result saved: {kw_path} (H={h_stat:.4f}, p={p_value:.6g})")

    if p_value >= alpha:
        print(f"[INFO] KW p >= {alpha}; skip post-hoc pairwise tests.")
        return

    # 2) Pairwise MW + Holm
    print("[INFO] Running pairwise Mann–Whitney (two-sided) with Holm correction...")
    pairs: List[Tuple[str, str]] = []
    pvals: List[float] = []

    # Compute raw p-values for all pairs
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            la, lb = langs[i], langs[j]
            xa = df_project_median.loc[df_project_median['language'] == la, 'median_coverage'].dropna().values
            xb = df_project_median.loc[df_project_median['language'] == lb, 'median_coverage'].dropna().values

            # Guard small samples: still compute MW if possible; SciPy handles ties with method='auto'
            if len(xa) < 1 or len(xb) < 1:
                p = np.nan
            else:
                try:
                    _, p = stats.mannwhitneyu(xa, xb, alternative='two-sided', method='auto')
                except Exception as e:
                    print(f"[WARN] MW failed for ({la}, {lb}): {e}")
                    p = np.nan

            pairs.append((la, lb))
            pvals.append(p)

    # Holm correction over all valid p-values
    pvals_array = np.array(pvals, dtype=float)
    valid_mask = ~np.isnan(pvals_array)
    adj_pvals = np.full_like(pvals_array, np.nan, dtype=float)

    if valid_mask.sum() > 0:
        _, p_adj, _, _ = multipletests(pvals_array[valid_mask], alpha=alpha, method="holm")
        adj_pvals[valid_mask] = p_adj

    # Build a symmetric matrix (lower-triangular plotted)
    mat = pd.DataFrame(np.ones((len(langs), len(langs))), index=langs, columns=langs, dtype=float)
    for (la, lb), p in zip(pairs, adj_pvals):
        if np.isnan(p):
            continue
        mat.loc[la, lb] = p
        mat.loc[lb, la] = p
    np.fill_diagonal(mat.values, 0.0)

    # Plot lower-triangle heatmap (vmin=0, vmax=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(mat, dtype=bool))  # Hide upper triangle
    sns.heatmap(
        mat, mask=mask, annot=True, fmt=".3f",
        cmap="coolwarm_r", vmin=0, vmax=1, cbar=False,
        linewidths=0.3, annot_kws={'fontsize': 8}, ax=ax
    )
    ax.set_title("Pairwise Mann–Whitney (Holm-adjusted p-values)", fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()

    pdf_path = out_dir / "pairwise_tests_summary_heatmaps_coverage.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"[INFO] MW-Holm heatmap saved: {pdf_path}")

# ==============================================================================
# Visualization
# ==============================================================================

def plot_coverage_distribution(df: pd.DataFrame, out_path: Path, visual_cfg: Dict[str, Any]) -> None:
    """
    Violin (base color) + Box (light color) by language for per-project median coverage.
    """
    if df.empty:
        print("[WARN] No data to plot.")
        return

    plot_data = df.copy()
    languages = sorted(plot_data['language'].unique())

    base_palette = {lang: visual_cfg['colors'].get(lang, '#cccccc') for lang in languages}
    light_palette = {lang: lighten_rgba(base_palette[lang], alpha=0.7, factor=0.6) for lang in languages}

    counts = plot_data.groupby('language')['project'].nunique()
    labels = [f"{lang}\n(n={counts.get(lang, 0)})" for lang in languages]

    fig, ax = plt.subplots(figsize=visual_cfg['figsize'])

    # Violin
    sns.violinplot(
        x='language', y='median_coverage', data=plot_data, ax=ax,
        inner=None, palette=base_palette, order=languages, cut=0,
        hue='language', dodge=False, legend=False
    )

    # Box
    sns.boxplot(
        x='language', y='median_coverage', data=plot_data, ax=ax,
        order=languages, width=0.30, showfliers=visual_cfg['show_outlier'],
        palette=light_palette, hue='language', dodge=False, legend=False,
        boxprops={'linewidth': 1.2, 'edgecolor': 'black'},
        medianprops={'color': 'black', 'linewidth': 1.3},
        whiskerprops={'linewidth': 1.0, 'color': 'black'},
        capprops={'linewidth': 1.0, 'color': 'black'}
    )

    ax.set_xticklabels(labels, fontsize=visual_cfg['tick_fontsize'])
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{int(y*100)}" for y in ax.get_yticks()])
    ax.set_ylabel("Median Patch Coverage per Project (%)", fontsize=visual_cfg['label_fontsize'])
    ax.grid(axis='y', linestyle=':', color='lightgray', zorder=0)
    ax.set_xlabel("")

    plt.tight_layout(pad=0.2)
    pdf_path = out_path / "patch_coverage_distribution_filtered.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Distribution plot saved: {pdf_path}")

# ==============================================================================
# Main process
# ==============================================================================

def process(db: DB | None) -> None:
    """
    Orchestrates the full RQ4 pipeline:
      - Load raw coverage CSVs
      - Compute per-project median coverage
      - Save summaries and run statistics
      - Produce required PDFs/CSVs
    """
    # -------- Paths (relative) --------
    data_dir = ROOT_DIR / "data" / "RQ4_Patch_Coverage"
    input_dir = ROOT_DIR / "data" / "Used_Data" / "projects"     # <-- place input CSVs here
    output_dir = data_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------- Config (visual & label map) --------
    config: Dict[str, Any] = {
        "display_language_map": {"jvm": "java"},  # normalize labels if needed
        "visual": {
            "colors": {
                "c": "#1f77b4", "c++": "#ff7f0e", "rust": "#2ca02c", "python": "#9467bd",
                "java": "#8c564b", "go": "#bcbd22", "ruby": "#d62728",
            },
            "figsize": (6.5, 4),
            "show_outlier": True,
            "tick_fontsize": 12,
            "label_fontsize": 12,
        }
    }

    # -------- Load all coverage CSVs --------
    df_all = load_coverage_data(input_dir)
    if df_all.empty:
        print("[INFO] No input data. Stop.")
        return

    # Lowercase for consistency
    if 'Language' in df_all.columns:
        df_all['Language'] = df_all['Language'].astype(str).str.lower()

    # -------- Compute per-project median coverage --------
    df_median = calculate_project_median_coverage(df_all)
    if df_median.empty:
        print("[INFO] No valid project data after filtering. Stop.")
        return

    # Apply display-language mapping (e.g., jvm -> java)
    if config.get("display_language_map"):
        df_median["language"] = df_median["language"].replace(config["display_language_map"])

    # Save per-project medians
    med_path = output_dir / "project_median_coverage.csv"
    df_median.to_csv(med_path, index=False)
    print(f"[INFO] Saved per-project medians: {med_path}")

    # -------- Summary statistics by language --------
    summary_stats = df_median.groupby('language')['median_coverage'].describe()
    summary_path = output_dir / "summary_statistics.csv"
    summary_stats.to_csv(summary_path)
    print(f"[INFO] Saved summary statistics: {summary_path}")

    # -------- KW then MW+Holm heatmap --------
    run_kw_and_pairwise_mwholm(
        df_project_median=df_median,
        out_dir=output_dir,
        alpha=0.05,
        display_language_map=config.get("display_language_map", None)
    )

    # -------- Distribution plot (violin+box) --------
    plot_coverage_distribution(df_median, output_dir, config['visual'])

    # -------- Console summary --------
    total_entries = len(df_all)
    total_projects = df_median["project"].nunique()
    print("\n📊 Analysis Summary")
    print(f"  - Total raw patch coverage rows: {total_entries:,}")
    print(f"  - Total valid projects: {total_projects:,}")

    # (Optional) If you still want to keep some cleanup logic, you can keep it here.
    # Removed absolute paths and destructive operations for Docker safety.

# ==============================================================================
# Entrypoint
# ==============================================================================

def main(task_id: int) -> None:
    start_time = None
    if utils and hasattr(utils, "return_now_datetime_jst"):
        start_time = utils.return_now_datetime_jst()

    try:
        # Create DB instance if needed
        db = None
        if DB is not None:
            try:
                # If your DB constructor relies on CONFIG_PATH, implement it inside dbFile.DB
                db = DB()
            except Exception as e:
                print(f"[WARN] DB initialization failed: {e}")

        process(db)

    except Exception as e:
        print(f"[ERROR] Unhandled error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Dependency check (minimal)
    try:
        import pandas, matplotlib, seaborn, scipy, statsmodels  # noqa: F401
    except ImportError as e:
        pkg = getattr(e, "name", "a required package")
        print(f"[ERROR] Missing dependency: {pkg}")
        print("Please install requirements: pandas matplotlib seaborn scipy statsmodels tqdm")
        sys.exit(1)

    # Parse args (optional task_id)
    task_id = 0
    if len(sys.argv) >= 2:
        try:
            task_id = int(sys.argv[1])
        except ValueError:
            print(f"[WARN] Non-integer task_id given: {sys.argv[1]} (default to 0)")
    main(task_id)
