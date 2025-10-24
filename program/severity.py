from __future__ import annotations
import os
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configparser import ConfigParser
from matplotlib.patches import Patch  # ★ 追加

# ================================================================
# Relative paths (Replication Package / Docker)
# ================================================================
BASE_DIR = Path(__file__).resolve().parent              # MSR2026/program
ROOT_DIR = BASE_DIR.parent                              # MSR2026/
DATA_DIR = ROOT_DIR / "data"                            # data
OUT_DIR = DATA_DIR / "RQ2_Severity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional local modules (__module)
MODULE_DIR = ROOT_DIR / "__module"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
try:
    import utils
    from dbFile import DB
except Exception:
    utils = None
    DB = None

CONFIG_PATH = MODULE_DIR / "envFile.ini"

# ================================================================
# Fetch data
# ================================================================
def fetch_data(db) -> pd.DataFrame:
    """
    Fetch issues with project info. Normalize fuzzing_engine.
    """
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
    END AS fuzzing_engine_norm
FROM issue_report ir
JOIN project_info pi ON ir.project = pi.project
JOIN project_counts pc ON ir.project = pc.project
WHERE ir.build_type IS NULL
  AND ir.status ILIKE '%fix%';
"""
    df = db.executeDict(query)
    print(f"[INFO] Raw rows: {len(df)}")
    if df.empty:
        return pd.DataFrame()
    df.rename(columns={"fuzzing_engine_norm": "fuzzing_engine"}, inplace=True)
    return df

# ================================================================
# Helpers
# ================================================================
def normalize_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize severity to S1–S4; others -> Unknown."""
    df = df.copy()
    s = df["severity"].astype(str).str.strip().str.upper()
    valid = {"S1", "S2", "S3", "S4"}
    df["severity"] = np.where(s.isin(valid), s, "Unknown")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Vulnerability issues only, filter languages, normalize severity."""
    df = df.copy()
    exclude = {"swift", "javascript"}
    df = df[~df["language"].isin(exclude)]
    df["language"] = df["language"].replace({"jvm": "java"})
    df["issue_type"] = df["type"].str.capitalize().str.extract(r"(Bug|Vulnerability)")[0]
    df = df[df["issue_type"] == "Vulnerability"]
    df = normalize_severity(df)
    return df

# ================================================================
# Plot
# ================================================================
def plot_vulnerability_severity(df: pd.DataFrame) -> None:
    """Plot 100% stacked bar (Vulnerability only)."""
    if df.empty:
        print("[INFO] No Vulnerability data.")
        return

    ct = pd.crosstab(df["language"], df["severity"])
    order = [c for c in ["S1", "S2", "S3", "S4"] if c in ct.columns]
    if "Unknown" in ct.columns:
        order += ["Unknown"]
    ct = ct.reindex(columns=order).sort_index()

    # 表示順：S4, S3, S2, S1（※元の処理を維持）
    pct = ct.div(ct.sum(axis=1), axis=0) * 100.0
    plot_order = ["S4", "S3", "S2", "S1"]
    if "Unknown" in pct.columns:
        plot_order.append("Unknown")
    pct = pct.reindex(columns=plot_order)

    # 色定義
    colors = {
        "S1": "#c03d3d",   # red
        "S2": "#e6a234",   # orange
        "S3": "#40964c",   # green
        "S4": "#3c77a1",   # blue
        "Unknown": "#909399"
    }
    color_list = [colors.get(c, "#999999") for c in pct.columns]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    pct.plot(kind="bar", stacked=True, color=color_list, width=0.8, ax=ax)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle=":", color="lightgray", zorder=0)
    plt.tight_layout()

    # ★ 凡例をハンドル指定で明示的に色とラベルを対応付け
    legend_order = [c for c in ["S1", "S2", "S3", "S4", "Unknown"] if c in ct.columns or c == "Unknown"]
    handles = [Patch(facecolor=colors[c], label=c) for c in legend_order]
    leg = ax.legend(handles=handles, title="Severity", loc="upper right", frameon=True, fontsize=10)
    leg.get_frame().set_alpha(0.85)

    out = OUT_DIR / "severity_language_stacked_bar_Vulnerability.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out}")

# ================================================================
# Main pipeline
# ================================================================
def run_pipeline(db):
    df = fetch_data(db)
    df = preprocess(df)
    plot_vulnerability_severity(df)

def main(task_id: int = 0):
    if utils and hasattr(utils, "setup_db") and DB:
        db = utils.setup_db(str(CONFIG_PATH))
        db.connect()
    else:
        raise RuntimeError("DB utilities not found (ensure __module and envFile.ini exist).")
    try:
        run_pipeline(db)
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    tid = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    main(tid)
