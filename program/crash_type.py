# RQ2_crash_type_analysis.py
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import re

# ==============================================================================
# Import custom modules (relative for Docker)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent           # .../MSR2026/program
ROOT_DIR = BASE_DIR.parent                           # .../MSR2026
DATA_DIR = ROOT_DIR / "data"
MODULE_DIR = ROOT_DIR / "__module"
CONFIG_PATH = MODULE_DIR / "envFile.ini"

if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
import utils
from dbFile import DB

# ==============================================================================
# Configuration
# ==============================================================================
CWE_CSV_PATH = DATA_DIR / "Used_Data" / "crash_to_root_cwe_report.csv"
OUTPUT_DIR = DATA_DIR / "RQ2_Crash_Type"

CONFIG: Dict[str, Any] = {
    "output_dir": str(OUTPUT_DIR),
    "merge_cwe_row": "Root_CWE",
    "exclude_languages": ["javascript", "swift"],
    "display_language_map": {"jvm": "java"},
    "display_cwe_map": {
        "CWE-664": "Resource Management",
        "CWE-682": "Incorrect Calculation",
        "CWE-691": "Control Flow",
        "CWE-693": "Protection Mechanism",
        "CWE-703": "Exceptional Handling",
        "CWE-707": "Neutralization",
        "CWE-710": "Coding Standards",
    },
    "visual": {
        "heatmap_color_min": "#ffffff",
        "heatmap_color_max": "#b41f1f",
        "heatmap_figsize_mode": "auto",
        "heatmap_scale_factor": 0.8,
        "heatmap_margin_width": 2.5,
        "heatmap_margin_height": 2.0,
        "heatmap_figsize_manual": (6, 6),
        "font_sizes": {"cell": 10},
        "tick_fontsize": 12,
    },
}


# ==============================================================================
# Data fetching and preprocessing
# ==============================================================================
def fetch_data(db: DB) -> pd.DataFrame:
    """Fetch issues joined with project info and normalize fields."""
    query = r"""
WITH project_counts AS (
    SELECT project
    FROM issue_report
    WHERE build_type IS NULL
      AND status ILIKE '%fix%'
    GROUP BY project
    HAVING COUNT(*) >= 10
)
SELECT
    ir.*,
    pi.language,
    CASE
        WHEN LOWER(ir.fuzzing_engine) = 'afl-qemu' THEN 'afl'
        ELSE ir.fuzzing_engine
    END AS fuzzing_engine_norm,
    REGEXP_REPLACE(
        REGEXP_REPLACE(LOWER(ir.crash_type), '\s+', '-', 'g'),
        '-?(\{[^}]*\}|\([^)]*\)|\d+)$',
        '',
        'g'
    ) AS crash_type_norm
FROM issue_report ir
JOIN project_info pi ON ir.project = pi.project
JOIN project_counts pc ON ir.project = pc.project
WHERE ir.build_type IS NULL
  AND ir.status ILIKE '%fix%';
"""
    df = db.executeDict(query)
    print(f"[INFO] Fetched rows: {len(df)}")

    if df.empty:
        return pd.DataFrame()

    if "fuzzing_engine" in df.columns:
        df.drop(columns=["fuzzing_engine"], inplace=True)
    df.rename(columns={"fuzzing_engine_norm": "fuzzing_engine"}, inplace=True)

    df["language"] = df["language"].replace(CONFIG["display_language_map"])
    df["issue_type"] = (
        df.get("type", pd.Series(index=df.index, dtype=object))
        .astype(str)
        .str.extract(r"(Bug|Vulnerability)", expand=False)
    )
    df.dropna(subset=["issue_type"], inplace=True)
    df = df[~df["language"].isin(CONFIG["exclude_languages"])].copy()
    return df


def merge_cwe(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Merge crash type → Root_CWE mapping from external CSV."""
    cwe_col = config["merge_cwe_row"]
    try:
        cwe_df = pd.read_csv(CWE_CSV_PATH)
    except FileNotFoundError:
        print(f"[WARN] Missing CWE mapping file: {CWE_CSV_PATH}")
        return df

    if "crash_type_norm" not in cwe_df.columns or cwe_col not in cwe_df.columns:
        print("[WARN] Required columns not found in CWE CSV.")
        return df

    merged = pd.merge(df, cwe_df[["crash_type_norm", cwe_col]], on="crash_type_norm", how="left")
    disp_map = config.get("display_cwe_map")
    if disp_map:
        merged[cwe_col] = merged[cwe_col].replace(disp_map)
    return merged


# ==============================================================================
# Heatmap visualization (original logic preserved)
# ==============================================================================
def plot_percentage_heatmap(contingency_table: pd.DataFrame,
                            config: Dict[str, Any],
                            output_dir: Path,
                            label_suffix: str,
                            transpose: bool = True):
    """
    Draws a percentage heatmap identical to the original implementation.
    Auto-adjusts figure size to keep cells square and shrinks colorbar slightly.
    """
    contingency_table = contingency_table.sort_index()
    percent_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    percent_table = percent_table.loc[:, (percent_table != 0).any(axis=0)]
    if percent_table.empty:
        print(f"[WARN] Empty target ({label_suffix}). Skipped heatmap.")
        return

    if "display_cwe_map" in config:
        sorted_keys = sorted(config["display_cwe_map"].keys(),
                             key=lambda x: int(re.search(r"\d+", x).group()))
        ordered_cols = [config["display_cwe_map"][k]
                        for k in sorted_keys
                        if config["display_cwe_map"][k] in percent_table.columns]
        if ordered_cols:
            percent_table = percent_table.reindex(columns=ordered_cols)

    if transpose:
        percent_table = percent_table.T

    visual_cfg = config["visual"]
    mode = visual_cfg.get("heatmap_figsize_mode", "auto")
    aspect_equal = False

    if mode == "manual":
        figsize = visual_cfg.get("heatmap_figsize_manual", (12, 8))
    else:
        aspect_equal = True
        num_rows, num_cols = percent_table.shape
        target_width = visual_cfg.get("heatmap_figsize_manual", (10, 8))[0]
        margin_w = visual_cfg.get("heatmap_margin_width", 2.5)
        margin_h = visual_cfg.get("heatmap_margin_height", 2.0)
        heatmap_body_width = target_width - margin_w
        if heatmap_body_width <= 0:
            heatmap_body_width = target_width * 0.7
        cell_size = heatmap_body_width / num_cols
        heatmap_body_height = cell_size * num_rows
        target_height = heatmap_body_height + margin_h
        figsize = (target_width, target_height)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        percent_table,
        annot=True,
        fmt=".1f",
        cmap=LinearSegmentedColormap.from_list(
            "custom_cmap",
            [visual_cfg.get("heatmap_color_min", "#ffffff"),
             visual_cfg.get("heatmap_color_max", "#b41f1f")]
        ),
        vmin=0, vmax=100,
        linewidths=0.5,
        linecolor="lightgray",
        ax=ax,
        cbar_kws={"shrink": 0.92, "label": ""},
        annot_kws={"size": visual_cfg.get("font_sizes", {}).get("cell", 10)},
    )

    if aspect_equal:
        ax.set_aspect("equal")
        ax.set_adjustable("box")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=visual_cfg.get("tick_fontsize", 12))
    ax.tick_params(axis="y", labelsize=visual_cfg.get("tick_fontsize", 12))
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    out_path = output_dir / f"percentage_heatmap_{label_suffix}_T.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[INFO] Saved heatmap: {out_path.name}")


# ==============================================================================
# Main process
# ==============================================================================
def process(db: DB):
    """Pipeline: fetch → merge → save summary → draw heatmap."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_data(db)
    if df.empty:
        print("[WARN] No data fetched from DB.")
        return

    df = merge_cwe(df, CONFIG)
    cwe_col = CONFIG["merge_cwe_row"]

    # Save issue summary CSV
    cols = ["id", "project", "language", "crash_type", "crash_type_norm", cwe_col]
    existing = [c for c in cols if c in df.columns]
    summary_path = OUTPUT_DIR / "issue_summary.csv"
    df[existing].to_csv(summary_path, index=False)
    print(f"[INFO] Saved summary: {summary_path.name}")

    # Build contingency table for all issues
    df_vis = df.dropna(subset=[cwe_col]).copy()
    ct = pd.crosstab(df_vis["language"], df_vis[cwe_col]).sort_index()
    ct = ct.loc[:, (ct != 0).any(axis=0)]
    if ct.empty:
        print("[WARN] Empty contingency table. Skip plot.")
        return

    # Plot percentage heatmap (transpose=True)
    plot_percentage_heatmap(ct, CONFIG, OUTPUT_DIR, label_suffix="all", transpose=True)


# ==============================================================================
# Entrypoint
# ==============================================================================
def main(task_id: int = 0):
    start_time = utils.return_now_datetime_jst()
    utils.save_to_file(f"task_id: {task_id}")

    db = utils.setup_db(str(CONFIG_PATH))
    db.connect()
    try:
        process(db)
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(0)
    else:
        try:
            main(int(sys.argv[1]))
        except ValueError:
            print(f"[ERROR] Invalid task ID: {sys.argv[1]}")
            sys.exit(1)
