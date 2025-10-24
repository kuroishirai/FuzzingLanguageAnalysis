# RQ3_Replication.py
# ------------------------------------------------------------------------------
# Replication-ready script (Docker-friendly, relative paths, English comments).
# Outputs (under data/RQ3_Replication/):
#   1) 06_reproducible_ratio_by_project_violin_box.pdf
#   2) 01_kruskal_result.csv
#   3) pairwise_tests_summary_heatmaps_reproducibility.pdf  (Mann–Whitney + Holm)
#   4) language_summary.csv
# Statistical pipeline: Kruskal–Wallis -> Pairwise Mann–Whitney (Holm correction)
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------------------------------------------------------------
# Resolve base paths (relative; no absolute /work/... paths)
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent              # MSR2026/program
ROOT_DIR = BASE_DIR.parent                              # MSR2026/
DATA_DIR = ROOT_DIR / "data" / "RQ3_Reproducibility"        # MSR2026/data/RQ3_Reproducibility
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Optional local modules under MSR2026/__module
MODULE_DIR = ROOT_DIR / "__module"
if MODULE_DIR.exists():
    sys.path.append(str(MODULE_DIR))

# Try to import user-provided modules (kept for DB access)
try:
    import utils
except Exception as _:
    utils = None  # Allowed to be absent in replication; user can provide later

try:
    from dbFile import DB
except Exception as e:
    print("[ERROR] dbFile.py is required but not found in MSR2026/__module.")
    print("Place dbFile.py under MSR2026/__module and try again.")
    raise

# Path to envFile.ini (relative)
CONFIG_PATH = MODULE_DIR / "envFile.ini"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "exclude_languages": ["javascript", "swift"],
    "display_language_map": {"jvm": "java"},  # unify label
    "visual": {
        "colors": {
            "c": "#1f77b4",
            "c++": "#ff7f0e",
            "rust": "#2ca02c",
            "python": "#9467bd",
            "java": "#8c564b",
            "javascript": "#e377c2",
            "swift": "#7f7f7f",
            "go": "#bcbd22",
        },
        "figsize": (6.5, 4.0),
        "tick_fontsize": 12,
        "label_fontsize": 12,
        "show_titles": False,
        "show_xlabels": False,
        "show_outlier": True,
    },
}

# ------------------------------------------------------------------------------
# Data fetch
# ------------------------------------------------------------------------------
def fetch_data(db: Any) -> pd.DataFrame:
    """
    Query issues and join language; normalize fields; return cleaned DataFrame.
    - Keeps only fixed issues (status ILIKE '%fix%') with build_type IS NULL
    - Projects with >= 10 such issues (project_counts)
    - Normalizes fuzzing_engine (afl-qemu -> afl)
    - Normalizes crash_type (lower-case, hyphenated, trailing tokens removed)
    - Adds boolean flags for hotlists: reproducible/unreproducible/fuzz-blocker
    """
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
    HAVING
        COUNT(*) >= 10
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
    ) AS crash_type_norm,
    'Reproducible' = ANY(ir.hotlists)   AS is_reproducible,
    'Unreproducible' = ANY(ir.hotlists) AS is_unreproducible,
    'Fuzz-Blocker' = ANY(ir.hotlists)   AS is_fuzz_blocker
FROM
    issue_report ir
JOIN
    project_info pi
    ON ir.project = pi.project
JOIN
    project_counts pc
    ON ir.project = pc.project
WHERE
    ir.build_type IS NULL
    AND ir.status ILIKE '%fix%'
    """
    df = db.executeDict(query)
    print(f"[INFO] Raw rows: {len(df)}")

    if df.empty:
        return pd.DataFrame()

    # Normalize fuzzing_engine column
    if "fuzzing_engine" in df.columns:
        df.drop(columns=["fuzzing_engine"], inplace=True)
    df.rename(columns={"fuzzing_engine_norm": "fuzzing_engine"}, inplace=True)

    # Language filtering and label unification
    df["language"] = df["language"].replace(CONFIG["display_language_map"])
    df = df[~df["language"].isin(CONFIG["exclude_languages"])]

    # Extract issue type (Bug | Vulnerability)
    if "type" in df.columns:
        df["issue_type"] = (
            df["type"].astype(str).str.capitalize().str.extract(r"(Bug|Vulnerability)")[0]
        )
        df.dropna(subset=["issue_type"], inplace=True)

    # Sanity checks for reproducibility flags
    for col in ["is_reproducible", "is_unreproducible"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
        else:
            print(f"[WARN] Missing column: {col}. Check SQL schema.")
            df[col] = False

    return df


# ------------------------------------------------------------------------------
# Visualization: Violin + Box of project-level reproducible ratio per language
# ------------------------------------------------------------------------------
def plot_reproducible_violin_box(df: pd.DataFrame) -> None:
    """
    Compute project-level reproducible ratio (0-100%) and plot violin+box by language.
    Saves: 06_reproducible_ratio_by_project_violin_box.pdf
    """
    if df.empty:
        print("[WARN] Empty dataframe for plotting.")
        return

    # Keep mutually exclusive reproducibility labels
    mask = ((df["is_reproducible"] == True) & (df["is_unreproducible"] == False)) | \
           ((df["is_reproducible"] == False) & (df["is_unreproducible"] == True))
    df_rep = df.loc[mask].copy()
    if df_rep.empty:
        print("[WARN] No mutually exclusive reproducibility rows.")
        return

    proj_ratio = (
        df_rep.groupby(["project", "language"])["is_reproducible"]
        .mean()
        .reset_index(name="reproducible_ratio")
    )
    proj_ratio["reproducible_ratio"] = proj_ratio["reproducible_ratio"] * 100.0

    print("[INFO] Project-level reproducible ratio by language (summary):")
    print(
        proj_ratio.groupby("language")["reproducible_ratio"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .sort_index()
        .to_string()
    )

    if proj_ratio.empty:
        print("[WARN] No data to plot after aggregation.")
        return

    # Helper to lighten colors
    def lighten(c, f=0.6):
        r, g, b, _ = mcolors.to_rgba(c)
        r = 1 - (1 - r) * f
        g = 1 - (1 - g) * f
        b = 1 - (1 - b) * f
        return (r, g, b, 1.0)

    languages = sorted(proj_ratio["language"].unique())
    base_palette = {lang: CONFIG["visual"]["colors"].get(lang, "#cccccc") for lang in languages}
    light_palette = {lang: lighten(base_palette[lang]) for lang in languages}

    fig, ax = plt.subplots(figsize=CONFIG["visual"]["figsize"])

    # Violin
    sns.violinplot(
        x="language", y="reproducible_ratio", data=proj_ratio,
        ax=ax, inner=None, palette=base_palette, order=languages, cut=0,
        hue="language", legend=False,
    )

    # Box on top of each language
    dark_gray = "#333333"
    for lang in languages:
        lang_data = proj_ratio[proj_ratio["language"] == lang]
        sns.boxplot(
            x="language", y="reproducible_ratio", data=lang_data, ax=ax,
            order=languages, width=0.3, showfliers=CONFIG["visual"]["show_outlier"],
            boxprops={"facecolor": light_palette[lang], "edgecolor": dark_gray, "linewidth": 1.3, "zorder": 2},
            whiskerprops={"color": dark_gray, "linewidth": 1.3},
            capprops={"color": dark_gray, "linewidth": 1.3},
            medianprops={"color": dark_gray, "linewidth": 1.3},
        )

    # X tick labels with (n=projects)
    counts = proj_ratio.groupby("language")["project"].nunique()
    new_labels = [f"{lang}\n(n={counts.get(lang, 0)})" for lang in languages]
    ax.set_xticks(ax.get_xticks(), new_labels)

    # Axes and labels
    ax.set_ylim(-5, 105)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_ylabel("Reproducible Ratio per Project (%)", fontsize=CONFIG["visual"]["label_fontsize"])
    ax.set_xlabel("Programming Language" if CONFIG["visual"]["show_xlabels"] else "", fontsize=CONFIG["visual"]["label_fontsize"])
    if CONFIG["visual"]["show_titles"]:
        ax.set_title("Distribution of Reproducible Ratios by Language")
    ax.tick_params(axis="x", rotation=0, labelsize=CONFIG["visual"]["tick_fontsize"])
    ax.tick_params(axis="y", labelsize=CONFIG["visual"]["tick_fontsize"])
    ax.grid(axis="y", linestyle=":", color="lightgray", zorder=0)

    out_pdf = DATA_DIR / "06_reproducible_ratio_by_project_violin_box.pdf"
    plt.tight_layout(pad=0.2)
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_pdf}")


# ------------------------------------------------------------------------------
# Statistics: KW then pairwise MW with Holm correction; heatmap PDF + CSVs
# ------------------------------------------------------------------------------
def run_kw_and_mwholm(df: pd.DataFrame) -> None:
    """
    - Compute per-project reproducible ratio (0-100%) per language.
    - Kruskal–Wallis across languages -> save CSV (01_kruskal_result.csv).
    - Pairwise Mann–Whitney (two-sided); Holm correction -> triangular heatmap PDF.
    - Also export language-level summary CSV (language_summary.csv).
    """
    if df.empty:
        print("[WARN] Empty dataframe for statistics.")
        return

    mask = ((df["is_reproducible"] == True) & (df["is_unreproducible"] == False)) | \
           ((df["is_reproducible"] == False) & (df["is_unreproducible"] == True))
    df_rep = df.loc[mask].copy()
    if df_rep.empty:
        print("[WARN] No mutually exclusive reproducibility rows for stats.")
        return

    proj_ratio = (
        df_rep.groupby(["project", "language"])["is_reproducible"]
        .mean()
        .reset_index(name="reproducible_ratio")
    )
    proj_ratio["reproducible_ratio"] = proj_ratio["reproducible_ratio"] * 100.0

    # ---- Kruskal–Wallis across languages ----
    langs = sorted(proj_ratio["language"].unique())
    groups = [proj_ratio.loc[proj_ratio["language"] == l, "reproducible_ratio"].dropna().astype(float) for l in langs]

    try:
        kw_stat, kw_p = kruskal(*groups)
    except Exception as e:
        kw_stat, kw_p = np.nan, np.nan
        print(f"[ERROR] Kruskal–Wallis failed: {e}")

    kw_df = pd.DataFrame([{"statistic": kw_stat, "pvalue": kw_p}])
    kw_path = DATA_DIR / "01_kruskal_result.csv"
    kw_df.to_csv(kw_path, index=False)
    print(f"[SAVE] {kw_path}")

    # ---- Pairwise Mann–Whitney; Holm correction ----
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            la, lb = langs[i], langs[j]
            xa = proj_ratio.loc[proj_ratio["language"] == la, "reproducible_ratio"].dropna().astype(float)
            xb = proj_ratio.loc[proj_ratio["language"] == lb, "reproducible_ratio"].dropna().astype(float)

            # Skip if too small or identical constants (MW may fail)
            if len(xa) < 3 or len(xb) < 3:
                pairs.append((la, lb, np.nan))
                continue
            if xa.nunique() == 1 and xb.nunique() == 1 and float(xa.iloc[0]) == float(xb.iloc[0]):
                pairs.append((la, lb, np.nan))
                continue

            try:
                _, p = mannwhitneyu(xa, xb, alternative="two-sided")
            except Exception as e:
                print(f"[WARN] MW failed for {la} vs {lb}: {e}")
                p = np.nan
            pairs.append((la, lb, p))

    df_pairs = pd.DataFrame(pairs, columns=["lang_a", "lang_b", "pvalue_raw"])

    # Holm correction on available p-values
    mask = df_pairs["pvalue_raw"].notna()
    if mask.any():
        _, p_adj, _, _ = multipletests(df_pairs.loc[mask, "pvalue_raw"], alpha=0.05, method="holm")
        df_pairs.loc[mask, "pvalue_holm"] = p_adj
    else:
        df_pairs["pvalue_holm"] = np.nan

    # ---- Language summary CSV (based on per-project ratios) ----
    language_summary = (
        proj_ratio.groupby("language")["reproducible_ratio"]
        .agg(
            project_count="count",
            mean_reproducible_ratio="mean",
            median_reproducible_ratio="median",
            min_reproducible_ratio="min",
            max_reproducible_ratio="max",
        )
        .reset_index()
        .round(2)
        .sort_values("language")
    )
    lang_csv = DATA_DIR / "language_summary.csv"
    language_summary.to_csv(lang_csv, index=False)
    print(f"[SAVE] {lang_csv}")

    # ---- Triangular heatmap PDF of Holm-adjusted p-values ----
    mat = pd.DataFrame(1.0, index=langs, columns=langs, dtype=float)
    for _, r in df_pairs.dropna(subset=["pvalue_holm"]).iterrows():
        a, b, p = r["lang_a"], r["lang_b"], float(r["pvalue_holm"])
        mat.loc[a, b] = p
        mat.loc[b, a] = p
    np.fill_diagonal(mat.values, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    mask_tri = np.triu(np.ones_like(mat, dtype=bool))
    sns.heatmap(
        mat, mask=mask_tri, annot=True, fmt=".3f", cmap="coolwarm_r",
        vmin=0, vmax=1, cbar=False, linewidths=0.3,
        annot_kws={"fontsize": 8}, ax=ax,
    )
    ax.set_title("Pairwise Mann–Whitney (Holm-adjusted p-values)" if CONFIG["visual"]["show_titles"] else "")
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    plt.tight_layout()

    pdf_path = DATA_DIR / "pairwise_tests_summary_heatmaps_reproducibility.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"[SAVE] {pdf_path}")


# ------------------------------------------------------------------------------
# Main process
# ------------------------------------------------------------------------------
def process(db: Any) -> None:
    df_raw = fetch_data(db)
    if df_raw.empty:
        print("[WARN] No rows returned from database. Exit.")
        return

    # Minimal modeling-friendly cleanup (stringify categorical/random fields if needed)
    for col in ["language", "project", "fuzzing_engine"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(
                lambda x: "_".join(map(str, x)) if hasattr(x, "__iter__") and not isinstance(x, str) else x
            ).astype(str)

    # Plot (only reproducible ratio violin+box)
    plot_reproducible_violin_box(df_raw)

    # Statistics: KW -> MW (Holm) + heatmap + CSVs
    run_kw_and_mwholm(df_raw)


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main(task_id: int = 0) -> None:
    # Use utils.setup_db if available; otherwise instantiate DB directly if possible.
    if utils is not None and hasattr(utils, "setup_db"):
        db = utils.setup_db(str(CONFIG_PATH))
    else:
        # Fallback: try a direct DB() if your DB class supports config path
        try:
            db = DB(str(CONFIG_PATH))
        except Exception as e:
            print("[ERROR] Cannot initialize DB. Provide utils.setup_db or compatible DB(config).")
            raise

    db.connect()
    try:
        process(db)
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}")
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    # Simple CLI: optional integer task_id
    if len(sys.argv) >= 2:
        try:
            tid = int(sys.argv[1])
        except Exception:
            print(f"[WARN] task_id must be int; got: {sys.argv[1]!r}. Using 0.")
            tid = 0
    else:
        tid = 0
    main(tid)
