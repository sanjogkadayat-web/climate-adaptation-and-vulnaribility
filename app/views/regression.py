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


def _trajectory_takeaway(d):
    d = d.sort_values("year")
    country = d["country"].iloc[0]
    n = len(d)
    under = int((d["misallocation_score"] < 0).sum())
    mean = d["misallocation_score"].mean()
    ext = d.loc[d["misallocation_score"].abs().idxmax()]
    head = (f"{country} has been persistently underfunded" if mean < -0.5 else
            f"{country} has been consistently over-resourced" if mean > 0.5 else
            f"{country} tracks the model closely")
    bullets = [
        f"- Actual aid fell **below** the model's prediction in **{under} of {n}** observed "
        f"years, averaging **{mean:+.2f}** log units.",
        f"- Its largest single-year deviation was **{int(ext['year'])}** at "
        f"**{ext['misallocation_score']:+.2f}** log units.",
    ]
    if n >= 3:
        aslope = np.polyfit(d["year"], d["misallocation_score"].abs(), 1)[0]
        trend = ("narrowed toward the model" if aslope < -0.01 else
                 "widened away from the model" if aslope > 0.01 else "stayed roughly steady")
        bullets.append(f"- Over time the gap has **{trend}**.")
    return head, "\n".join(bullets)


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
        st.markdown(
            "- **Vulnerability dominates.** Its coefficient (3.64) is several times larger than "
            "any other driver, making need the single strongest pull on aid.\n"
            "- **Size is the gatekeeper.** Population is strongly positive, which is why "
            "vulnerability only becomes significant once it is in the model: large countries "
            "draw aid regardless of need.\n"
            "- **Poorer and better-governed countries get more.** Higher GDP per capita lowers "
            "aid; stronger governance modestly raises it.\n"
            "- **Geography and category do not matter.** Landlocked status, small-island status, "
            "and income tier are all indistinguishable from zero once need, size, wealth, and "
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
        st.markdown(
            "- **Half of allocation is predictable.** With R² = 0.50, the scatter around the "
            "45° line is the other half, which the project scores as misallocation.\n"
            "- **Fit is uneven across tiers.** Strong for low- and middle-income recipients "
            "(r ≈ 0.70), almost absent for high-income ones (r = 0.12), where aid seems to "
            "follow logic outside need and size.\n"
            "- **The biggest gaps are concrete.** Points far below the line are chronically "
            "underfunded country-years; the widest is China in 2023, 3.85 log units below "
            "prediction."
        )

    with t_country:
        df = _load_scored()
        countries = sorted(df["country"].unique())
        grp = df.groupby("country")["misallocation_score"].agg(["size", "mean"])
        default = grp[grp["size"] >= 10]["mean"].idxmin()
        country = st.selectbox("Select a country", countries,
                               index=countries.index(default), key="traj_country")
        d = df[df["country"] == country]
        head, bullets = _trajectory_takeaway(d)
        st.subheader(head)
        st.plotly_chart(_trajectory(d), use_container_width=True, key="trajectory")
        st.caption(
            "Solid line is actual aid received; dashed line is the model's prediction. "
            "The shaded gap between them is the misallocation, in log units."
        )
        st.markdown(bullets)

    with t_table:
        st.subheader("The exact benchmark behind every misallocation score")
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.caption(
            f"Year fixed effects across {stats['year_fe']} years are included but not shown. "
            "Significance: * p<0.05, ** p<0.01, *** p<0.001."
        )
        st.markdown(
            "- **Four drivers are significant** at the 95% level: vulnerability, population, "
            "GDP per capita, and governance effectiveness.\n"
            "- **Errors are clustered by country,** so repeated observations of one country do "
            "not overstate precision.\n"
            "- **Year fixed effects absorb the growing aid pool,** so each coefficient reflects "
            "within-year allocation, not aid simply rising over time.\n"
            "- **Income tier is a control,** so the misallocation residual already nets out "
            "average differences between income groups."
        )