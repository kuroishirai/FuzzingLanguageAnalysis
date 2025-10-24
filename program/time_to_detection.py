# -*- coding: utf-8 -*-
"""
Time-to-Detection (TTD) Analysis
Replication package version: relative paths + KW/MW(Holm) only.
"""

from __future__ import annotations
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ==============================================================================
# Path configuration (relative paths for Docker/replication)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODULE_DIR = BASE_DIR.parent / "__module"
DATA_DIR = BASE_DIR.parent / "data" / "RQ4_Time_to_Detection"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = MODULE_DIR / "envFile.ini"

print(f"[module_path] {MODULE_DIR}")
print(f"[config_path] {CONFIG_PATH}")
print(f"[data_path] {DATA_DIR}")

# ==============================================================================
# Import user-defined modules
# ==============================================================================
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

import utils
from dbFile import DB


# ==============================================================================
# Fetch data
# ==============================================================================
def fetch_data(db: DB) -> pd.DataFrame:
    """Fetch and preprocess issue_report data from DB."""
    query = r"""
WITH project_counts AS (
    SELECT project
    FROM issue_report
    WHERE build_type IS NULL
      AND status ILIKE '%fix%'
      AND regressed_buildtime IS NOT NULL
      AND array_length(regressed_buildtime, 1) >= 2
    GROUP BY project
    HAVING COUNT(*) >= 10
)
SELECT
    ir.*,
    pi.language,
    CASE
        WHEN ir.regressed_buildtime IS NOT NULL AND array_length(ir.regressed_buildtime, 1) >= 2
            THEN ir.regressed_buildtime[2]
        ELSE ir.crash_buildtime[1]
    END AS including_time
FROM issue_report ir
JOIN project_info pi ON ir.project = pi.project
JOIN project_counts pc ON ir.project = pc.project
WHERE ir.build_type IS NULL
  AND ir.status ILIKE '%fix%'
  AND ir.regressed_buildtime IS NOT NULL
  AND array_length(ir.regressed_buildtime, 1) >= 2
ORDER BY ir.project, ir.reported_time;
"""
    df = db.executeDict(query)
    print(f"[INFO] fetched rows: {len(df)}")
    if df.empty:
        return pd.DataFrame()

    df["reported_time"] = pd.to_datetime(df["reported_time"], errors="coerce")
    df["including_time"] = pd.to_datetime(df["including_time"], errors="coerce")
    df["time_diff"] = df["reported_time"] - df["including_time"]

    print(f"[INFO] Added 'time_diff' column (dtype={df['time_diff'].dtype})")
    return df


# ==============================================================================
# Visualization (ProjectMedian-level violin + box)
# ==============================================================================
def lighten_rgba(c, a=0.6, f=0.5):
    """Lighten color and return RGBA."""
    r, g, b, _ = mcolors.to_rgba(c)
    r = 1 - (1 - r) * f
    g = 1 - (1 - g) * f
    b = 1 - (1 - b) * f
    return (r, g, b, a)


def plot_violin_and_box(df: pd.DataFrame, config: dict):
    """Generate violin + box plot for project median-level TTD."""
    vis = config["visual"]
    y_col = "time_diff_days"
    df = df.copy()
    df[y_col] = df[y_col].clip(lower=0)

    order = sorted(df["language"].unique())
    palette = {lang: vis["colors"].get(lang, "#CCCCCC") for lang in order}

    plt.figure(figsize=vis["figsize"])
    ax = sns.violinplot(
        data=df, x="language", y=y_col,
        order=order, palette=palette,
        inner=None, linewidth=1.5, cut=0
    )
    sns.boxplot(
        data=df, x="language", y=y_col, order=order,
        width=0.4, showcaps=True, showfliers=True, ax=ax,
        boxprops={"zorder": 3}, linewidth=1.5,
        flierprops={"marker": "o", "color": "black", "alpha": 0.5, "markersize": 4, "zorder": 4}
    )

    import matplotlib as mpl
    boxes = [p for p in ax.findobj(mpl.patches.PathPatch)]
    boxes = boxes[-len(order):]
    for lang, patch in zip(order, boxes):
        base = palette.get(lang, "#CCCCCC")
        patch.set_facecolor(lighten_rgba(base, a=0.55, f=0.65))
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    ax.set_xlabel("")
    ax.set_ylabel("Time Difference (Days)", fontsize=vis["label_fontsize"])
    ax.set_yscale(vis.get("scale", "linear"))
    ax.set_ylim(0, vis.get("show_outlier", 200))
    ax.grid(axis="y", linestyle=":", color="lightgray", zorder=0)

    lang_counts = df.groupby("language")["project"].nunique().to_dict()
    xtick_labels = [f"{lang}\n(n={lang_counts.get(lang, 0)})" for lang in order]
    plt.xticks(range(len(order)), xtick_labels, fontsize=vis["tick_fontsize"])

    plt.tight_layout(pad=0.2)
    outpath = DATA_DIR / "plot_violin_box_projectmedian.pdf"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"[OK] saved {outpath}")


# ==============================================================================
# KW + MW(Holm) tests
# ==============================================================================
def run_kw_mw_holm(df: pd.DataFrame) -> Dict[str, Any]:
    """Run KW + pairwise Mann–Whitney (Holm)."""
    df_valid = df.dropna(subset=["time_diff"]).copy()
    df_valid["time_diff_days"] = df_valid["time_diff"].dt.total_seconds() / (24 * 3600)
    df_valid = df_valid[df_valid["time_diff_days"] >= 0]

    langs = sorted(df_valid["language"].unique())
    groups = [df_valid[df_valid["language"] == l]["time_diff_days"] for l in langs]

    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"[KW] statistic={kw_stat:.4f}, p={kw_p:.4e}")

    pairs, pvals = [], []
    for i, li in enumerate(langs):
        for lj in langs[i+1:]:
            g1 = df_valid.loc[df_valid["language"] == li, "time_diff_days"].values
            g2 = df_valid.loc[df_valid["language"] == lj, "time_diff_days"].values
            try:
                _, p = mannwhitneyu(g1, g2, alternative="two-sided")
            except Exception as e:
                print(f"[WARN] MW failed for {li}-{lj}: {e}")
                p = np.nan
            pairs.append((li, lj))
            pvals.append(p)

    _, pvals_holm, _, _ = multipletests(pvals, method="holm")

    return {"kw": {"stat": kw_stat, "p": kw_p},
            "languages": langs, "pairs": pairs, "mw_holm": pvals_holm}


def save_mw_holm_heatmap(res: Dict[str, Any]):
    """Save Mann–Whitney (Holm corrected) heatmap."""
    langs = res["languages"]
    pairs = res["pairs"]
    pvals = res["mw_holm"]

    mat = pd.DataFrame(1.0, index=langs, columns=langs)
    for (l1, l2), p in zip(pairs, pvals):
        if not pd.isna(p):
            mat.loc[l1, l2] = mat.loc[l2, l1] = p
    np.fill_diagonal(mat.values, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(mat, dtype=bool))
    sns.heatmap(mat, mask=mask, annot=True, fmt=".3f",
                cmap="coolwarm_r", vmin=0, vmax=1,
                cbar=True, linewidths=0.3, annot_kws={"fontsize": 8}, ax=ax)
    ax.set_title("Pairwise Mann–Whitney (Holm corrected)", fontsize=12)
    plt.tight_layout()

    pdf_path = DATA_DIR / "pairwise_tests_summary_heatmaps.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {pdf_path}")


# ==============================================================================
# Main process
# ==============================================================================
def process(db: DB):
    config = {
        "visual": {
            "colors": {
                "Bug": "#3d6bc0",
                "Vulnerability": "#c03d3d",
                "c": "#1f77b4",
                "c++": "#ff7f0e",
                "rust": "#2ca02c",
                "python": "#9467bd",
                "java": "#8c564b",
                "go": "#bcbd22",
            },
            "figsize": (6.5, 4),
            "tick_fontsize": 12,
            "label_fontsize": 12,
            "scale": "linear",
            "show_outlier": 200,
        }
    }

    df = fetch_data(db)
    if df.empty:
        print("[INFO] No data found.")
        return

    df = df[~df["language"].isin(["javascript", "swift", "go"])].copy()
    df["language"] = df["language"].replace({"jvm": "java"})
    df["issue_type"] = df["type"].str.capitalize().str.extract(r"(Bug|Vulnerability)")[0]
    df.dropna(subset=["issue_type"], inplace=True)

    # === Project median-level ===
    df["time_diff_days"] = df["time_diff"].dt.total_seconds() / (24 * 3600)
    df_proj = (
        df.groupby(["language", "project", "issue_type"])["time_diff_days"]
        .median()
        .reset_index()
    )
    plot_violin_and_box(df_proj, config)

    # === KW + MW(Holm) ===
    res = run_kw_mw_holm(df)
    save_mw_holm_heatmap(res)

    # === Save KW CSV ===
    kw_out = DATA_DIR / "kw_test_result.csv"
    pd.DataFrame([{
        "test": "Kruskal-Wallis",
        "statistic": res["kw"]["stat"],
        "p_value": res["kw"]["p"],
        "significant": "Yes" if res["kw"]["p"] < 0.05 else "No"
    }]).to_csv(kw_out, index=False)
    print(f"[OK] saved {kw_out}")

    # === Save language summary ===
    lang_summary = (
        df.groupby("language", as_index=False)
        .agg(
            count=("time_diff_days", "count"),
            median_TTD=("time_diff_days", "median"),
            mean_TTD=("time_diff_days", "mean"),
            std_TTD=("time_diff_days", "std")
        )
        .sort_values("median_TTD")
    )
    lang_out = DATA_DIR / "language_ttd_summary.csv"
    lang_summary.to_csv(lang_out, index=False)
    print(f"[OK] saved {lang_out}")


# ==============================================================================
# Main execution
# ==============================================================================
def main(task_id: int):
    start_time = utils.return_now_datetime_jst()
    utils.save_to_file(f"task_id: {task_id}")

    db = utils.setup_db(CONFIG_PATH)
    db.connect()

    try:
        process(db)
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(0)
    elif len(sys.argv) == 2:
        try:
            main(int(sys.argv[1]))
        except ValueError:
            print("[ERROR] Task ID must be integer.")
            sys.exit(1)
    else:
        print("Usage: python time_to_detection.py <TASK_ID>")
        sys.exit(1)
