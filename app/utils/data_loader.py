"""
Data loading utilities for the Streamlit app.
Updated to load from notebook reports with HR@10/NDCG@10 metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

def find_repo_root() -> Path:
    """Find the repository root directory."""
    current = Path(__file__).resolve().parent
    for p in [current, *current.parents]:
        if (p / "PROJECT_STATE.md").exists():
            return p
    return current.parent.parent

REPO_ROOT = find_repo_root()

# Dataset configurations
DATASETS = {
    "XuetangX": {
        "name": "XuetangX",
        "description": "Large Chinese MOOC Platform",
        "raw_path": REPO_ROOT / "data" / "raw" / "xuetangx",
        "interim_path": REPO_ROOT / "data" / "interim" / "xuetangx",
        "processed_path": REPO_ROOT / "data" / "processed" / "xuetangx",
        "results_path": REPO_ROOT / "results",
        "models_path": REPO_ROOT / "models" / "baselines",
        "reports_path": REPO_ROOT / "reports",
    },
}

# Current best results from notebooks (hardcoded for reliability)
CURRENT_RESULTS = {
    "XuetangX": {
        "vanilla_maml": {
            "notebook": "07_maml_xuetangx",
            "test_HR@10": 47.35,
            "test_NDCG@10": 37.41,
            "description": "Vanilla MAML (random init)",
        },
        "reliability_maml": {
            "notebook": "11_reliability_weighted_maml_xuetangx",
            "test_HR@10": 48.34,
            "test_NDCG@10": 37.71,
            "description": "Reliability-Weighted MAML",
        },
        "warmstart_reliability_maml": {
            "notebook": "12_warmstart_reliability_maml_xuetangx",
            "test_HR@10": 55.62,
            "test_NDCG@10": 44.80,
            "description": "Warm-Start + Reliability-Weighted MAML",
        },
        "gru_baseline": {
            "notebook": "06_baselines_xuetangx",
            "test_HR@10": 33.55,
            "test_NDCG@10": 25.0,
            "description": "GRU4Rec Global Baseline",
        },
    }
}

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a dataset."""
    return DATASETS.get(dataset_name, DATASETS["XuetangX"])

def load_json(path: Path) -> Optional[Dict]:
    """Load a JSON file."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Load a parquet file."""
    if path.exists():
        return pd.read_parquet(path)
    return None

def get_latest_report(notebook_name: str) -> Optional[Dict]:
    """Get the latest report for a notebook."""
    reports_dir = REPO_ROOT / "reports" / notebook_name

    if not reports_dir.exists():
        return None

    # Find all run directories
    run_dirs = [d for d in reports_dir.iterdir() if d.is_dir()]

    if not run_dirs:
        return None

    # Sort by name (which is timestamp-based) and get the latest
    run_dirs.sort(key=lambda x: x.name, reverse=True)

    for run_dir in run_dirs:
        report_path = run_dir / "report.json"
        if report_path.exists():
            return load_json(report_path)

    return None

def load_raw_data(dataset_name: str) -> Optional[pd.DataFrame]:
    """Load raw data for a dataset."""
    config = get_dataset_config(dataset_name)

    if dataset_name == "XuetangX":
        # XuetangX uses JSON files, load interim instead
        interim_file = config["interim_path"] / "interactions.parquet"
        if interim_file.exists():
            return pd.read_parquet(interim_file)
    return None

def load_interactions(dataset_name: str) -> Optional[pd.DataFrame]:
    """Load interactions from interim data."""
    if dataset_name == "XuetangX":
        # XuetangX uses a different file name and structure
        path = REPO_ROOT / "data" / "interim" / "xuetangx_events_raw.parquet"
        df = load_parquet(path)
        if df is not None:
            # Normalize column names: rename course_id to item_id, timestamp to ts_epoch
            df = df.rename(columns={"course_id": "item_id"})
            # Convert timestamp to epoch if needed
            if "timestamp" in df.columns and "ts_epoch" not in df.columns:
                df["ts_epoch"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9
        return df
    else:
        # MARS uses standard path
        config = get_dataset_config(dataset_name)
        path = config["interim_path"] / "interactions.parquet"
        return load_parquet(path)

def load_sessions(dataset_name: str) -> Optional[pd.DataFrame]:
    """Load sessionized data."""
    config = get_dataset_config(dataset_name)
    path = config["processed_path"] / "sessions" / "sessions.parquet"
    return load_parquet(path)

def load_pairs(dataset_name: str, split: str = "train") -> Optional[pd.DataFrame]:
    """Load prefix-target pairs."""
    config = get_dataset_config(dataset_name)
    path = config["processed_path"] / "pairs" / f"pairs_{split}.parquet"
    return load_parquet(path)

def load_pairs_with_reliability(dataset_name: str) -> Optional[pd.DataFrame]:
    """Load prefix-target pairs with reliability scores."""
    config = get_dataset_config(dataset_name)
    path = config["processed_path"] / "pairs_with_reliability" / "pairs.parquet"
    return load_parquet(path)

def load_episodes(dataset_name: str, split: str = "train", K: int = 5, Q: int = 10) -> Optional[pd.DataFrame]:
    """Load episodes for meta-learning."""
    config = get_dataset_config(dataset_name)
    path = config["processed_path"] / "episodes" / f"episodes_{split}_K{K}_Q{Q}.parquet"
    return load_parquet(path)

def load_vocab(dataset_name: str) -> Optional[Dict]:
    """Load vocabulary mapping."""
    config = get_dataset_config(dataset_name)

    # Try different vocab file names
    for filename in ["item2id.json", "course2id.json"]:
        path = config["processed_path"] / "vocab" / filename
        if path.exists():
            return load_json(path)
    return None

def load_baseline_results(dataset_name: str) -> Optional[Dict]:
    """Load baseline results from NB06 report."""
    # Try to load from reports
    report = get_latest_report("06_baselines_xuetangx")

    if report and report.get("metrics"):
        return report

    # Fallback to hardcoded results
    return {
        "baselines": {
            "gru_global": {
                "HR@10": CURRENT_RESULTS["XuetangX"]["gru_baseline"]["test_HR@10"],
                "NDCG@10": CURRENT_RESULTS["XuetangX"]["gru_baseline"]["test_NDCG@10"],
            }
        }
    }

def load_maml_results(dataset_name: str, variant: str = "vanilla") -> Optional[Dict]:
    """
    Load MAML results for a specific variant.

    Variants:
    - "vanilla": Basic MAML (NB07)
    - "reliability": Reliability-Weighted MAML (NB11)
    - "warmstart_reliability": Warm-Start + Reliability (NB12)
    """
    notebook_map = {
        "vanilla": "07_maml_xuetangx",
        "basic": "07_maml_xuetangx",
        "reliability": "11_reliability_weighted_maml_xuetangx",
        "warmstart_reliability": "12_warmstart_reliability_maml_xuetangx",
        "combined": "12_warmstart_reliability_maml_xuetangx",
    }

    notebook_name = notebook_map.get(variant, "07_maml_xuetangx")
    report = get_latest_report(notebook_name)

    if report:
        return report

    # Fallback to hardcoded results
    variant_key_map = {
        "vanilla": "vanilla_maml",
        "basic": "vanilla_maml",
        "reliability": "reliability_maml",
        "warmstart_reliability": "warmstart_reliability_maml",
        "combined": "warmstart_reliability_maml",
    }

    key = variant_key_map.get(variant, "vanilla_maml")
    hardcoded = CURRENT_RESULTS["XuetangX"].get(key, {})

    return {
        "results": {
            "test_HR@10": hardcoded.get("test_HR@10", 0),
            "test_NDCG@10": hardcoded.get("test_NDCG@10", 0),
        },
        "description": hardcoded.get("description", ""),
    }

def get_all_maml_comparison(dataset_name: str = "XuetangX") -> pd.DataFrame:
    """Get comparison dataframe of all MAML variants."""
    data = []

    # Vanilla MAML (NB07)
    vanilla = load_maml_results(dataset_name, "vanilla")
    if vanilla:
        results = vanilla.get("results", vanilla.get("metrics", {}))
        data.append({
            "Method": "Vanilla MAML",
            "Notebook": "NB07",
            "HR@10": results.get("test_HR@10", CURRENT_RESULTS["XuetangX"]["vanilla_maml"]["test_HR@10"]),
            "NDCG@10": results.get("test_NDCG@10", CURRENT_RESULTS["XuetangX"]["vanilla_maml"]["test_NDCG@10"]),
        })

    # Reliability-Weighted MAML (NB11)
    reliability = load_maml_results(dataset_name, "reliability")
    if reliability:
        results = reliability.get("results", reliability.get("metrics", {}))
        data.append({
            "Method": "Reliability-Weighted MAML",
            "Notebook": "NB11",
            "HR@10": results.get("test_HR@10", CURRENT_RESULTS["XuetangX"]["reliability_maml"]["test_HR@10"]),
            "NDCG@10": results.get("test_NDCG@10", CURRENT_RESULTS["XuetangX"]["reliability_maml"]["test_NDCG@10"]),
        })

    # Warm-Start + Reliability (NB12)
    combined = load_maml_results(dataset_name, "warmstart_reliability")
    if combined:
        results = combined.get("results", combined.get("metrics", {}))
        data.append({
            "Method": "Warm-Start + Reliability",
            "Notebook": "NB12",
            "HR@10": results.get("test_HR@10", CURRENT_RESULTS["XuetangX"]["warmstart_reliability_maml"]["test_HR@10"]),
            "NDCG@10": results.get("test_NDCG@10", CURRENT_RESULTS["XuetangX"]["warmstart_reliability_maml"]["test_NDCG@10"]),
        })

    return pd.DataFrame(data)

def compute_gap_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute gap statistics from interactions dataframe."""
    # Handle different timestamp column names
    ts_col = "ts_epoch" if "ts_epoch" in df.columns else "timestamp"

    # Convert timestamp to numeric if needed
    if ts_col == "timestamp":
        df = df.copy()
        df["ts_epoch"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9
        ts_col = "ts_epoch"

    df = df.sort_values(["user_id", ts_col]).reset_index(drop=True)
    df["prev_ts"] = df.groupby("user_id")[ts_col].shift(1)
    df["gap_seconds"] = df[ts_col] - df["prev_ts"]

    gaps = df["gap_seconds"].dropna()

    if len(gaps) == 0:
        return {}

    return {
        "n_gaps": len(gaps),
        "min_seconds": float(gaps.min()),
        "max_seconds": float(gaps.max()),
        "mean_seconds": float(gaps.mean()),
        "median_seconds": float(gaps.median()),
        "p25_seconds": float(np.percentile(gaps, 25)),
        "p75_seconds": float(np.percentile(gaps, 75)),
        "p90_seconds": float(np.percentile(gaps, 90)),
        "p95_seconds": float(np.percentile(gaps, 95)),
        "p99_seconds": float(np.percentile(gaps, 99)),
        "pct_within_30min": float((gaps <= 1800).mean() * 100),
        "gaps": gaps,
    }
