import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data import PROCESSED
from app.model import fit_model

TIER = {"H": "High", "UM": "Upper-middle", "LM": "Lower-middle", "L": "Low"}
TIER_ORDER = ["High", "Upper-middle", "Lower-middle", "Low"]


@st.cache_data(show_spinner=False)
def _load_scored():
    df = pd.read_csv(PROCESSED / "model_scored.csv")
    df["Income tier"] = df["income_group"].map(TIER)
    return df


def _coef_plot(tbl):
    d = tbl[tbl["Predictor"] != "Intercept"].copy()
    d["err"] = 1.96 * d["Std. error"]
    d["lo"], d["hi"] = d["Coefficient"] - d["err"], d["Coefficient"] + d["err"]
    d["Significance"] = np.where((d["lo"] > 0) | (d["hi"] < 0),
                                 "Significant (95%)", "Not significant")
    order = list(d["Predictor"])[::-1]
    fig = px.scatter(
        d, x="Coefficient", y="Predictor", color="Significance", error_x="err",
        color_discrete_map={"Significant (95%)": "#2c6e8f", "Not significant": "#b8b8b8"},
    )
    fig.update_yaxes(categoryorder="array", categoryarray=order, title=None)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
    return fig


def _coef_value(tbl, label):
    """Pull a single coefficient by its display label from the fitted table."""
    row = tbl.loc[tbl["Predictor"] == label].iloc[0]
    return float(row["Coefficient"])


def _fit_scatter(d, lo, hi):
    fig = px.scatter(
        d, x="predicted_log_aid", y="log_aid", color="Income tier",
        category_orders={"Income tier": TIER_ORDER},
        hover_data={"country": True, "year": True, "misallocation_score": ":.2f",
                    "predicted_log_aid": ":.2f", "log_aid": ":.2f", "Income tier": False},
        labels={"predicted_log_aid": "Predicted log aid", "log_aid": "Actual log aid"},
        opacity=0.6,
    )
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                             line=dict(dash="dash", color="gray")))
    fig.update_xaxes(range=[lo - 0.3, hi + 0.3])
    fig.update_yaxes(range=[lo - 0.3, hi + 0.3])
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
    return fig


def _fit_takeaway(d, label):
    corr = d["predicted_log_aid"].corr(d["log_aid"])
    big = (d["misallocation_score"].abs() > 1).mean()
    w = d.loc[d["misallocation_score"].idxmin()]
    return (
        f"Within **{label}** ({len(d):,} country-years), predicted and actual aid correlate "
        f"**r = {corr:.2f}**, and **{big:.0%}** miss by more than one log unit. "
        f"Widest single gap: **{w['country']} ({int(w['year'])})**, "
        f"{w['misallocation_score']:+.2f} log units below prediction."
    )


def _fit_by_tier(df):
    """Bar chart of model fit (predicted vs actual correlation) within each income tier."""
    rows = []
    for t in TIER_ORDER:
        d = df[df["Income tier"] == t]
        if len(d) > 2:
            rows.append({"tier": t, "r": d["predicted_log_aid"].corr(d["log_aid"]), "n": len(d)})
    f = pd.DataFrame(rows)
    colors = ["#b8b8b8" if r < 0.3 else "#2c6e8f" for r in f["r"]]
    fig = go.Figure(go.Bar(
        x=f["r"], y=f["tier"], orientation="h", marker_color=colors,
        text=[f"r = {r:.2f}  (n={n})" for r, n in zip(f["r"], f["n"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: r = %{x:.2f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.05], title="Correlation of predicted vs. actual aid")
    fig.update_yaxes(categoryorder="array", categoryarray=TIER_ORDER[::-1], title=None)
    fig.update_layout(height=240, margin=dict(l=0, r=40, t=10, b=0))
    return fig


def _trajectory(d):
    d = d.sort_values("year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["year"], y=d["predicted_log_aid"], mode="lines", name="Predicted by model",
        line=dict(color="#9aa0a6", dash="dash"),
        hovertemplate="%{x}<br>Predicted %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=d["year"], y=d["log_aid"], mode="lines+markers", name="Actual aid received",
        line=dict(color="#2c6e8f"), fill="tonexty", fillcolor="rgba(120,120,120,0.12)",
        customdata=d["misallocation_score"],
        hovertemplate="%{x}<br>Actual %{y:.2f}<br>Gap %{customdata:+.2f}<extra></extra>"))
    fig.update_xaxes(dtick=2, title="Year")
    fig.update_yaxes(title="Log adaptation aid")
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None))
    return fig


def _trajectory_stats(d):
    """Headline plus the scorecard figures for one country's residual path."""
    d = d.sort_values("year")
    country = d["country"].iloc[0]
    n = len(d)
    under = int((d["misallocation_score"] < 0).sum())
    mean = float(d["misallocation_score"].mean())
    head = (f"{country} has been persistently underfunded" if mean < -0.5 else
            f"{country} has been consistently over-resourced" if mean > 0.5 else
            f"{country} tracks the model closely")
    if n >= 3:
        aslope = np.polyfit(d["year"], d["misallocation_score"].abs(), 1)[0]
        trend = ("Narrowing" if aslope < -0.01 else
                 "Widening" if aslope > 0.01 else "Steady")
    else:
        trend = "Too few years"
    return {"head": head, "n": n, "under": under, "mean": mean, "trend": trend}


def render():
    st.title("Aid tracks need and size; the half the model can't explain is the misallocation")
    st.markdown(
        "This regression is the benchmark behind every score on the map. It predicts log "
        "adaptation aid from vulnerability, population size, income, governance, and geography, "
        "with errors clustered by country and year fixed effects. Two things stand out. First, "
        "aid genuinely tracks vulnerability, but only once population is in the model, since "
        "larger countries draw more aid regardless of need. Second, the model explains only "
        "about half of where aid actually goes (R² = 0.50), and that unexplained half, the "
        "residual, is exactly what the project flags as over- or under-funding."
    )

    tbl, stats = fit_model()
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{stats['r2']:.3f}")
    c2.metric("Adjusted R²", f"{stats['adj_r2']:.3f}")
    c3.metric("Observations", f"{stats['n']:,}")

    t_drivers, t_fit, t_country, t_table = st.tabs(
        ["What drives aid", "Model fit", "A country over time", "Coefficient table"])

    with t_drivers:
        st.subheader("Need and size move aid; geography and income tier don't")
        st.plotly_chart(_coef_plot(tbl), use_container_width=True, key="coef_plot")

        st.markdown("**The four drivers that actually move aid**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Vulnerability · need", f"{_coef_value(tbl, 'Vulnerability (ND-GAIN)'):+.2f}")
        d2.metric("Population · size", f"{_coef_value(tbl, 'Log population'):+.2f}")
        d3.metric("GDP per capita · wealth", f"{_coef_value(tbl, 'Log GDP per capita'):+.2f}")
        d4.metric("Governance", f"{_coef_value(tbl, 'Governance effectiveness'):+.3f}")
        st.caption(
            "Coefficients on log adaptation aid; all four are significant at 95%. Units differ "
            "across predictors, so magnitudes are not directly comparable, but the signs and the "
            "coefficient plot above tell the story."
        )

        st.markdown(
            "**Need is the strongest pull, but only because size is controlled.** Vulnerability "
            "carries the largest coefficient by far, yet it reaches significance only once "
            "population is in the model: large countries draw aid regardless of need, so without a "
            "size control, need looks irrelevant (this is the result that flipped the project's "
            "core finding). **Poorer and better-governed countries get more,** with higher GDP per "
            "capita lowering aid and stronger governance modestly raising it. **Geography and "
            "category add nothing:** landlocked status, small-island status, and every income-tier "
            "dummy are statistically indistinguishable from zero once need, size, wealth, and "
            "governance are accounted for."
        )

    with t_fit:
        st.subheader("The model captures about half of aid; the spread is the misallocation")
        df = _load_scored()
        lo = float(min(df["predicted_log_aid"].min(), df["log_aid"].min()))
        hi = float(max(df["predicted_log_aid"].max(), df["log_aid"].max()))
        tier = st.selectbox("Filter by income tier", ["All tiers"] + TIER_ORDER, index=0)
        d = df if tier == "All tiers" else df[df["Income tier"] == tier]
        st.plotly_chart(_fit_scatter(d, lo, hi), use_container_width=True, key="fit_scatter")
        label = "all tiers" if tier == "All tiers" else f"the {tier}-income tier"
        st.markdown(_fit_takeaway(d, label))

        st.markdown("**Where the model works, and where it stops working**")
        st.plotly_chart(_fit_by_tier(df), use_container_width=True, key="fit_by_tier")
        st.markdown(
            "**Fit is strong for poor and middle-income recipients and collapses at the top.** "
            "For low and lower-middle income countries the model explains aid well (r ≈ 0.7), so "
            "their residuals are trustworthy signals of over- or under-funding. For high-income "
            "recipients the correlation is just 0.12: aid there appears to follow logic outside "
            "need and size (politics, strategic ties, co-financing), so those residuals should be "
            "read with more caution. **The remaining spread is the project's raw material:** every "
            "point off the 45° line is a country-year the model cannot explain, and the country "
            "averages of those gaps are the misallocation scores on the Home map."
        )

    with t_country:
        df = _load_scored()
        countries = sorted(df["country"].unique())
        grp = df.groupby("country")["misallocation_score"].agg(["size", "mean"])
        default = grp[grp["size"] >= 10]["mean"].idxmin()
        country = st.selectbox("Select a country", countries,
                               index=countries.index(default), key="traj_country")
        d = df[df["country"] == country]
        s = _trajectory_stats(d)
        st.subheader(s["head"])
        st.plotly_chart(_trajectory(d), use_container_width=True, key="trajectory")
        st.caption(
            "Solid line is actual aid received; dashed line is the model's prediction. "
            "The shaded gap between them is the misallocation, in log units."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Years below the model", f"{s['under']} of {s['n']}")
        m2.metric("Average gap", f"{s['mean']:+.2f} log")
        m3.metric("Gap over time", s["trend"])
        st.caption(
            "A gap that is narrowing means aid is converging on what the model predicts; widening "
            "means it is drifting further from it. Trend needs at least three observed years."
        )

    with t_table:
        st.subheader("The exact benchmark behind every misallocation score")
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.caption(
            f"Year fixed effects across {stats['year_fe']} years are included but not shown. "
            "Significance: * p<0.05, ** p<0.01, *** p<0.001."
        )

        n_sig = int(((tbl["Predictor"] != "Intercept") & (tbl["Sig."] != "")).sum())
        b1, b2, b3 = st.columns(3)
        b1.metric("Significant drivers (95%)", f"{n_sig}")
        b2.metric("Standard errors", "Clustered by country")
        b3.metric("Year fixed effects", f"{stats['year_fe']} years")
        st.markdown(
            "**Clustering by country** stops repeated observations of the same country from "
            "overstating precision, which is why some predictors with large point estimates still "
            "fail the significance test. **Year fixed effects absorb the growing global aid pool,** "
            "so each coefficient reflects within-year allocation rather than aid simply rising over "
            "time. **Income tier enters as a control,** so the misallocation residual already nets "
            "out average differences between income groups rather than penalizing a country for the "
            "tier it sits in."
        )