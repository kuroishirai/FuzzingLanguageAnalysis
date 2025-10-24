# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ==============================================================================
# Environment setup (relative path, Docker-friendly)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "RQ1_Frequency"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VIS_DIR = DATA_DIR / "a_visualization"
STAT_DIR = DATA_DIR / "b_statistics"
CSV_DIR = DATA_DIR / "c_csv_export"
for d in [VIS_DIR, STAT_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Module import (DB and utils)
# ==============================================================================
module_folder = os.environ.get("MODULE_FOLDER", str(PROJECT_ROOT / "__module"))
print("[module_path]", module_folder)

if module_folder not in sys.path:
    sys.path.append(module_folder)

import utils
from dbFile import DB

config_path = str(Path(module_folder) / "envFile.ini")
print("[config_path]", config_path)


# ==============================================================================
# Data fetching from database
# ==============================================================================
def fetch_data(db: DB) -> pd.DataFrame:
    query = r"""
WITH project_counts AS (
    SELECT
        project
    FROM
        issue_report
    WHERE
        build_type IS NULL
        AND status ILIKE '%fix%'
    GROUP BY
        project
    HAVING COUNT(*) >= 10
)
SELECT
    main_counts.project,
    pi.language,
    main_counts.total_issue_count,
    main_counts.bug_count,
    main_counts.vulnerability_count,
    main_counts.fuzzing_build_count
FROM (
    SELECT
        COALESCE(ir_counts.project, ba_counts.project) AS project,
        COALESCE(ir_counts.total_issue_count, 0) AS total_issue_count,
        COALESCE(ir_counts.bug_count, 0) AS bug_count,
        COALESCE(ir_counts.vulnerability_count, 0) AS vulnerability_count,
        COALESCE(ba_counts.fuzzing_build_count, 0) AS fuzzing_build_count
    FROM (
        SELECT
            project,
            COUNT(*) AS total_issue_count,
            COUNT(CASE WHEN type = 'Bug' THEN 1 END) AS bug_count,
            COUNT(CASE WHEN type = 'Vulnerability' THEN 1 END) AS vulnerability_count
        FROM
            issue_report
        WHERE
            build_type IS NULL
            AND status ILIKE '%fix%'
        GROUP BY
            project
    ) AS ir_counts
    FULL OUTER JOIN (
        SELECT
            project,
            COUNT(*) AS fuzzing_build_count
        FROM
            buildlog_analysis
        WHERE
            build_type = 'Fuzzing'
        GROUP BY
            project
    ) AS ba_counts
    ON ir_counts.project = ba_counts.project
) AS main_counts
INNER JOIN project_info AS pi
    ON main_counts.project = pi.project
JOIN project_counts pc
    ON main_counts.project = pc.project
WHERE
    main_counts.total_issue_count > 0
ORDER BY
    main_counts.project;
    """
    df = db.executeDict(query)
    print(f"[INFO] Fetched rows: {len(df)}")
    return df if not df.empty else pd.DataFrame()


# ==============================================================================
# Visualization (keep original 2.frequency_issue.pdf style)
# ==============================================================================
def lighten_rgba(c, a=0.6, f=0.5):
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(c)
    r = 1 - (1 - r) * f
    g = 1 - (1 - g) * f
    b = 1 - (1 - b) * f
    return (r, g, b, a)


def build_palette_for(languages: List[str], base_colors: Dict[str, str]):
    return {lang: base_colors.get(lang, '#999999') for lang in languages}


def build_box_palette_for(languages: List[str], base_colors: Dict[str, str]):
    return {lang: lighten_rgba(base_colors.get(lang, '#999999')) for lang in languages}


def violin_with_box(ax, data, x, y, *,
                    languages_order: List[str],
                    base_colors: Dict[str, str],
                    show_outlier: float,
                    tick_fontsize: int,
                    label_fontsize: int):
    pal_violin = build_palette_for(languages_order, base_colors)
    pal_box = build_box_palette_for(languages_order, base_colors)
    sns.violinplot(data=data, x=x, y=y, order=languages_order,
                   hue=x, hue_order=languages_order, dodge=False,
                   palette=pal_violin, inner=None, cut=0, ax=ax, legend=False)
    sns.boxplot(data=data, x=x, y=y, order=languages_order,
                hue=x, hue_order=languages_order, dodge=False,
                palette=pal_box, showcaps=True, showfliers=True,
                width=0.2, boxprops=dict(edgecolor="black"),
                whiskerprops={'color': "black"},
                capprops={'color': "black"},
                medianprops={'color': "black"}, ax=ax, legend=False)
    ax.set_xticks(np.arange(len(languages_order)))
    ax.set_xticklabels(languages_order, fontsize=tick_fontsize)
    ax.set_ylabel(y, fontsize=label_fontsize)
    if show_outlier > 0:
        ax.set_ylim(0, show_outlier)
    ax.grid(axis='y', linestyle=':', color='lightgray', zorder=0)


def visualize_issue_frequency(df: pd.DataFrame):
    base_colors = {
        "Bug": "#3d6bc0",
        "Vulnerability": "#c03d3d",
        "c": "#1f77b4", "c++": "#ff7f0e", "rust": "#2ca02c",
        "python": "#9467bd", "java": "#8c564b", "go": "#bcbd22",
    }
    show_outlier = 0.5
    tick_fs, label_fs = 12, 12

    denom = df["fuzzing_build_count"].replace(0, np.nan)
    df["issue_freq"] = df["total_issue_count"] / denom
    data_issue = df.copy()
    langs_issue = sorted(data_issue["language"].unique())

    fig, ax = plt.subplots(figsize=(6.5, 4))
    violin_with_box(ax, data_issue, "language", "issue_freq",
                    languages_order=langs_issue,
                    base_colors=base_colors,
                    show_outlier=show_outlier,
                    tick_fontsize=tick_fs,
                    label_fontsize=label_fs)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.tight_layout(pad=0.2)
    out_path = VIS_DIR / "2.frequency_issue.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


# ==============================================================================
# Statistics: Kruskal–Wallis → Mann–Whitney (Holm)
# ==============================================================================
def run_kw_mw_holm(df: pd.DataFrame) -> Dict[str, Any]:
    df_valid = df.dropna(subset=["issue_freq"])
    langs = sorted(df_valid["language"].unique())
    groups = [df_valid[df_valid["language"] == l]["issue_freq"] for l in langs]

    try:
        kw_stat, kw_p = stats.kruskal(*groups)
    except Exception as e:
        print("[WARN] Kruskal–Wallis failed:", e)
        kw_stat, kw_p = np.nan, np.nan

    pairs, pvals = [], []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            g1 = df_valid[df_valid["language"] == langs[i]]["issue_freq"]
            g2 = df_valid[df_valid["language"] == langs[j]]["issue_freq"]
            try:
                _, p = mannwhitneyu(g1, g2)
            except Exception:
                p = np.nan
            pairs.append((langs[i], langs[j]))
            pvals.append(p)

    _, p_holm, _, _ = multipletests(pvals, method="holm")
    return {"kw_stat": kw_stat, "kw_p": kw_p, "pairs": pairs, "langs": langs, "p_holm": p_holm}


def save_heatmap(dfstat: Dict[str, Any]):
    langs = dfstat["langs"]
    pairs, pvals = dfstat["pairs"], dfstat["p_holm"]
    mat = pd.DataFrame(np.ones((len(langs), len(langs))), index=langs, columns=langs)
    for (a, b), p in zip(pairs, pvals):
        mat.loc[a, b] = mat.loc[b, a] = p
    np.fill_diagonal(mat.values, 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(mat, dtype=bool))
    sns.heatmap(mat, mask=mask, annot=True, fmt=".3f", cmap="coolwarm_r", vmin=0, vmax=1,
                annot_kws={'fontsize': 8}, ax=ax)
    ax.set_title("Mann–Whitney (Holm-corrected p-values)", fontsize=12)
    out_path = STAT_DIR / "issue_freq_pairwise_heatmaps.pdf"
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


# ==============================================================================
# CSV export
# ==============================================================================
def export_csv(df: pd.DataFrame, dfstat: Dict[str, Any]):
    denom = df["fuzzing_build_count"].replace(0, np.nan)
    df["issue_freq"] = df["total_issue_count"] / denom
    summary = (
        df.groupby("language")["issue_freq"]
        .agg(median="median", mean="mean", std="std", count="count")
        .reset_index()
        .rename(columns={
            "median": "issue_freq_median",
            "mean": "issue_freq_mean",
            "std": "issue_freq_std",
            "count": "issue_freq_count"
        })
    )
    summary.to_csv(CSV_DIR / "languages_summary.csv", index=False)

    stats_csv = pd.DataFrame([{
        "metric": "issue_freq",
        "test_type": "kruskal",
        "statistic": dfstat["kw_stat"],
        "pvalue": dfstat["kw_p"]
    }])
    stats_csv.to_csv(CSV_DIR / "statistical_tests.csv", index=False)
    print(f"[OK] Saved CSVs in {CSV_DIR}")


# ==============================================================================
# Main pipeline
# ==============================================================================
def process(db: DB):
    df_raw = fetch_data(db)
    if df_raw.empty:
        print("[INFO] No data retrieved.")
        return

    df = df_raw[~df_raw["language"].isin(["javascript", "swift"])].copy()
    df["language"] = df["language"].replace({"jvm": "java"})

    visualize_issue_frequency(df)
    dfstat = run_kw_mw_holm(df)
    save_heatmap(dfstat)
    export_csv(df, dfstat)
    print("[DONE] RQ1_Frequency pipeline completed.")


# ==============================================================================
# Entry point
# ==============================================================================
def main(task_id: int):
    start_time = utils.return_now_datetime_jst()
    utils.save_to_file(f"task_id: {task_id}")

    db = utils.setup_db(config_path)
    db.connect()
    try:
        process(db)
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(0)
    elif len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        print("Usage: python frequency.py [TASK_ID]")
        sys.exit(1)
