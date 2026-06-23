import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st

from app.data import PROCESSED

FORMULA = (
    "log_aid ~ vulnerability + log_gdp + gov_effectiveness + log_population "
    "+ is_landlocked + is_sids "
    "+ C(income_group, Treatment(reference='L')) + C(year)"
)

LABELS = {
    "Intercept": "Intercept",
    "vulnerability": "Vulnerability (ND-GAIN)",
    "log_gdp": "Log GDP per capita",
    "gov_effectiveness": "Governance effectiveness",
    "log_population": "Log population",
    "is_landlocked": "Landlocked",
    "is_sids": "Small island state (SIDS)",
    "C(income_group, Treatment(reference='L'))[T.H]": "Income: High (vs Low)",
    "C(income_group, Treatment(reference='L'))[T.LM]": "Income: Lower-middle (vs Low)",
    "C(income_group, Treatment(reference='L'))[T.UM]": "Income: Upper-middle (vs Low)",
}

# substantive predictors first, income dummies and intercept last
ORDER = [
    "vulnerability", "log_population", "log_gdp", "gov_effectiveness",
    "is_landlocked", "is_sids",
    "C(income_group, Treatment(reference='L'))[T.H]",
    "C(income_group, Treatment(reference='L'))[T.LM]",
    "C(income_group, Treatment(reference='L'))[T.UM]",
    "Intercept",
]


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


@st.cache_data(show_spinner=False)
def fit_model():
    """Refit the final FE model and return (coefficient table, fit stats)."""
    df = pd.read_csv(PROCESSED / "model_scored.csv")
    fit = smf.ols(FORMULA, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["iso3"]}
    )
    mask = ~fit.params.index.str.contains(r"C\(year\)")  # hide year dummies
    raw = fit.params.index[mask]
    order = {k: i for i, k in enumerate(ORDER)}

    tbl = pd.DataFrame({
        "Predictor": [LABELS.get(k, k) for k in raw],
        "Coefficient": fit.params[mask].round(3).values,
        "Std. error": fit.bse[mask].round(3).values,
        "p-value": [_pfmt(p) for p in fit.pvalues[mask].values],
        "Sig.": [_stars(p) for p in fit.pvalues[mask].values],
        "_o": [order.get(k, 99) for k in raw],
    }).sort_values("_o").drop(columns="_o").reset_index(drop=True)

    stats = {
        "r2": fit.rsquared,
        "adj_r2": fit.rsquared_adj,
        "n": int(fit.nobs),
        "year_fe": int(df["year"].nunique()),
    }
    return tbl, stats