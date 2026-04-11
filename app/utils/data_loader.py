"""
data_loader.py — Load pipeline report JSONs for the Streamlit app.
All data comes exclusively from ./reports/*/report.json files.
"""

import json
import sys
from pathlib import Path

import streamlit as st


# ── Repo root resolution ──────────────────────────────────────────────────────

def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "PROJECT_STATE.md").exists():
            return p
    raise RuntimeError("Cannot find repo root (PROJECT_STATE.md not found)")


REPO_ROOT = find_repo_root()
# Reports are stored inside app/data/ so they are committed and available on Streamlit Cloud.
# Fall back to repo-root reports/ for local development if app/data/ doesn't exist.
_app_data = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = _app_data if _app_data.exists() else REPO_ROOT / "reports"

# Canonical notebook name prefix per dataset
DATASET_PREFIX = {
    "XuetangX": "xuetangx",
    "MARS": "mars",
}

# All pipeline notebook stages (suffix appended to dataset prefix)
PIPELINE_STAGES = [
    "01_ingest",
    "02_sessionize",
    "03_vocab_pairs",
    "03b_srs_scores",
    "04_user_split",
    "05_episode_index",
    "06_base_model_selection",
    "07_standard_maml",
    "08_warmstart_maml",
    "09_srs_validation",
    "10_srs_adaptive_maml",
    "11_warmstart_srs_adaptive_maml",
]


# ── Core loader ───────────────────────────────────────────────────────────────

def load_latest(nb_name: str):
    """Return metrics dict from the most recent report.json for nb_name, or None."""
    d = REPORTS_DIR / nb_name
    if not d.exists():
        return None
    for run in reversed(sorted(d.iterdir())):
        rp = run / "report.json"
        if rp.exists():
            return json.loads(rp.read_text("utf-8"))
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_dataset_reports(dataset: str) -> dict:
    """
    Load all available pipeline reports for the given dataset.
    Returns a dict keyed by stage name (e.g. '01_ingest') → full report dict.
    Missing stages have None values.
    """
    prefix = DATASET_PREFIX[dataset]
    reports = {}
    for stage in PIPELINE_STAGES:
        nb_name = f"{stage}_{prefix}"
        reports[stage] = load_latest(nb_name)
    return reports


@st.cache_data(show_spinner=False)
def load_both_datasets() -> dict:
    """Load reports for both datasets; returns {'XuetangX': {...}, 'MARS': {...}}."""
    return {ds: load_dataset_reports(ds) for ds in DATASET_PREFIX}


# ── Convenience accessors ─────────────────────────────────────────────────────

def metrics(reports: dict, stage: str) -> dict:
    """Return the metrics sub-dict for a stage, or {} if unavailable."""
    r = reports.get(stage)
    if r is None:
        return {}
    return r.get("metrics", {})


def key_findings(reports: dict, stage: str) -> list:
    r = reports.get(stage)
    if r is None:
        return []
    return r.get("key_findings", [])


def run_tag(reports: dict, stage: str) -> str:
    r = reports.get(stage)
    if r is None:
        return "n/a"
    return r.get("run_tag", "n/a")
