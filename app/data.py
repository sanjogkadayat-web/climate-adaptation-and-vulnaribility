from pathlib import Path
import pandas as pd
import streamlit as st


def _find_repo_root() -> Path:
    """Walk up from this file until we hit the folder containing data/processed/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "processed").is_dir():
            return parent
    raise FileNotFoundError(
        "Could not locate data/processed/. Ensure it sits at the repo root "
        "alongside the app/ package (and is committed for Cloud deploys)."
    )


PROCESSED = _find_repo_root() / "data" / "processed"


@st.cache_data(show_spinner=False)
def load_country_scored() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "country_scored.csv")


@st.cache_data(show_spinner=False)
def load_model_scored() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "model_scored.csv")