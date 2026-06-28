from pathlib import Path

import pandas as pd
import streamlit as st

from app.data import PROCESSED

# Switch to "claude-opus-4-8" for richer briefs, or "claude-haiku-4-5-20251001" for speed.
MODEL = "claude-sonnet-4-6"

SKILL_PATH = Path(__file__).resolve().parent / "skills" / "policy_brief.md"

_FALLBACK_SKILL = (
    "Write a structured, evidence-based climate adaptation aid policy brief grounded strictly in "
    "the provided figures, with an executive summary, context, the allocation gap, why it "
    "persists, an outlook to 2030, a numbered multi-step recommendation plan, and caveats."
)


@st.cache_data(show_spinner=False)
def _skill() -> str:
    try:
        return SKILL_PATH.read_text(encoding="utf-8")
    except Exception:
        return _FALLBACK_SKILL


@st.cache_data(show_spinner=False)
def _scored():
    return pd.read_csv(PROCESSED / "country_scored.csv")


@st.cache_data(show_spinner=False)
def _proj():
    return pd.read_csv(PROCESSED / "country_projection.csv")


@st.cache_data(show_spinner=False)
def _regional():
    return pd.read_csv(PROCESSED / "regional_gap_trend.csv")


def has_api_key() -> bool:
    try:
        return bool(st.secrets.get("ANTHROPIC_API_KEY"))
    except Exception:
        return False


def country_list():
    return sorted(_scored()["country"].unique())


def input_facts(country: str) -> str:
    """The exact figures block passed to the model, exposed for transparency in the UI."""
    return _facts_block(_facts(country))


def _facts(country: str) -> dict:
    s = _scored()
    row = s[s["country"] == country].iloc[0]
    f = {
        "country": country, "iso3": row["iso3"], "region": row["region"],
        "profile": row["profile"], "misallocation_mean": float(row["misallocation_mean"]),
        "income_tier": row["income_tier"], "tier_rank": int(row["tier_rank"]),
        "tier_n": int(row["tier_n"]), "vulnerability_mean": float(row["vulnerability_mean"]),
        "aid_usd_m_mean": float(row["aid_usd_m_mean"]), "n_years": int(row["n_years"]),
        "thin_data": bool(row["thin_data"]),
    }
    reg = _regional()
    rr = reg[reg["region"] == row["region"]]
    if len(rr):
        rr = rr.iloc[0]
        f["region_status"] = rr["status"]
        f["region_gap_2023"] = float(rr["gap_2023"])
        f["region_gap_2030"] = float(rr["gap_2030"])
    p = _proj()
    pr = p[p["iso3"] == row["iso3"]]
    if len(pr):
        pr = pr.iloc[0]
        f["aid_2030_proj_usd_m"] = float(pr["aid_2030_proj_usd_m"])
        f["pct_change_23_30"] = float(pr["pct_change_23_30"])
        f["annual_pct"] = float(pr["annual_pct"])
        f["trend_significant"] = bool(pr["trend_significant"])
    return f


def _facts_block(f: dict) -> str:
    lines = [
        f"Country: {f['country']} (region: {f['region']})",
        f"Funding profile: {f['profile']}",
        f"Mean misallocation residual: {f['misallocation_mean']:+.2f} "
        "(negative means it receives less adaptation aid than the model predicts)",
        f"Within its {f['income_tier']} income tier, ranks {f['tier_rank']} of {f['tier_n']} "
        "by misallocation (1 = most underfunded)",
        f"Mean ND-GAIN vulnerability: {f['vulnerability_mean']:.2f}",
        f"Mean annual adaptation aid observed: ${f['aid_usd_m_mean']:.2f} million "
        f"over {f['n_years']} years",
    ]
    if f.get("thin_data"):
        lines.append("Data caveat: based on only one or two years, so figures are uncertain.")
    if "region_status" in f:
        lines.append(
            f"Regional gap for {f['region']} is {f['region_status']}: {f['region_gap_2023']:+.2f} "
            f"in 2023, projected to {f['region_gap_2030']:+.2f} by 2030.")
    if "aid_2030_proj_usd_m" in f:
        sig = "statistically significant" if f["trend_significant"] else "not statistically significant"
        lines.append(
            f"If its 2010 to 2023 trend holds, adaptation aid is projected near "
            f"${f['aid_2030_proj_usd_m']:,.0f} million by 2030 "
            f"({f['pct_change_23_30']:+.0f}% vs 2023, {f['annual_pct']:+.1f}% per year, "
            f"trend {sig}).")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def generate_brief(country: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    facts = _facts_block(_facts(country))
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=_skill(),
        messages=[{"role": "user", "content":
                   f"Project figures for {country}:\n\n{facts}\n\n"
                   "Write the policy brief now, following the skill exactly."}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")