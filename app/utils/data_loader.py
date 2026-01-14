"""
Data loading utilities for the Streamlit app.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

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
    "MARS": {
        "name": "MARS",
        "description": "English MOOC Dataset",
        "raw_path": REPO_ROOT / "data" / "raw" / "mars",
        "interim_path": REPO_ROOT / "data" / "interim" / "mars",
        "processed_path": REPO_ROOT / "data" / "processed" / "mars",
        "results_path": REPO_ROOT / "results",
        "models_path": REPO_ROOT / "models" / "baselines" / "mars",
        "reports_path": REPO_ROOT / "reports",
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

def load_raw_data(dataset_name: str) -> Optional[pd.DataFrame]:
    """Load raw data for a dataset."""
    config = get_dataset_config(dataset_name)

    if dataset_name == "MARS":
        raw_file = config["raw_path"] / "explicit_ratings_en.csv"
        if raw_file.exists():
            return pd.read_csv(raw_file)
    elif dataset_name == "XuetangX":
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
    """Load baseline results."""
    config = get_dataset_config(dataset_name)

    if dataset_name == "MARS":
        path = config["results_path"] / "mars_baselines_K5_Q10.json"
    else:
        path = config["results_path"] / "baselines_K5_Q10.json"

    return load_json(path)

def load_maml_results(dataset_name: str, variant: str = "basic") -> Optional[Dict]:
    """Load MAML results for a specific variant."""
    config = get_dataset_config(dataset_name)

    # Map variant to filename
    variant_files = {
        "basic": "maml_K5_Q10.json",
        "warmstart": "maml_warmstart_K5_Q10.json",
        "residual": "maml_residual_K5_Q10.json",
        "warmstart_residual": "maml_warmstart_residual_K5_Q10.json",
    }

    filename = variant_files.get(variant, "maml_K5_Q10.json")

    if dataset_name == "MARS":
        filename = filename.replace("maml_", "mars_maml_")

    path = config["results_path"] / filename
    data = load_json(path)

    if data is None:
        return None

    # Normalize the format - extract key metrics into standard fields
    normalized = data.copy()

    # Find the MAML results section (different keys for different variants)
    maml_section = None
    for key in ["maml", "maml_warmstart", "maml_residual", "maml_warmstart_residual"]:
        if key in data:
            maml_section = data[key]
            break

    if maml_section:
        # Extract zero-shot and few-shot metrics
        normalized["zero_shot_metrics"] = maml_section.get("zero_shot", {})
        normalized["few_shot_metrics"] = maml_section.get("few_shot_K5", {})
        # Also set final_metrics to few_shot for backwards compatibility
        normalized["final_metrics"] = maml_section.get("few_shot_K5", {})

    # Extract baseline
    if "baseline" in data:
        normalized["baseline_metrics"] = data["baseline"].get("gru_global", {})

    # Extract ablation results if present
    normalized["ablation_support_size"] = data.get("ablation_support_size", {})
    normalized["ablation_adaptation_steps"] = data.get("ablation_adaptation_steps", {})

    # Extract sweep/tuning info and USE TUNED RESULTS if available
    if "metrics" in data:
        sweep = data["metrics"]
        normalized["sweep_results"] = sweep

        # If tuned results exist, use them as the primary few_shot_metrics
        if "maml_few_shot_K5_acc1" in sweep:
            # Update with tuned accuracy
            tuned_acc = sweep["maml_few_shot_K5_acc1"]
            if normalized.get("few_shot_metrics"):
                normalized["few_shot_metrics"]["accuracy@1"] = tuned_acc
            else:
                normalized["few_shot_metrics"] = {"accuracy@1": tuned_acc}

            # Also update improvement percentage
            if "improvement_over_baseline_pct" in sweep:
                normalized["improvement_over_baseline_pct"] = sweep["improvement_over_baseline_pct"]

    return normalized


def load_all_maml_results(dataset_name: str) -> Dict[str, Optional[Dict]]:
    """Load all MAML variant results for comparison."""
    variants = ["basic", "warmstart", "residual", "warmstart_residual"]
    results = {}
    for variant in variants:
        results[variant] = load_maml_results(dataset_name, variant)
    return results

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
