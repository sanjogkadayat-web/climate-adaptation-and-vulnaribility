import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from app.geo import CENTROIDS

CFG = {
    "iso3": "iso3", "name": "country", "resid": "misallocation_mean",
    "income": "income_tier", "region": "region", "nyears": "n_years",
    "profile": "profile", "thin": "thin_data",
}

MAP_HEIGHT = 520   # master size dial: raise to make the whole map bigger
LABEL_N = 5        # how many of the darkest-red extremes to name on the map (0 hides)


def _colorbar_ticks(m: float):
    top = int(np.floor(m))
    vals = list(range(-top, top + 1))
    text = []
    for v in vals:
        if v == 0:
            word = "as predicted"
        elif v == -top:
            word = "far below predicted"
        elif v == top:
            word = "far above predicted"
        elif v < 0:
            word = "below predicted"
        else:
            word = "above predicted"
        num = "0" if v == 0 else f"{v:+d}"
        text.append(f"{num}  · {word}")
    return vals, text


def build_choropleth(df: pd.DataFrame):
    d = df.copy()
    d["_thin"] = d[CFG["thin"]].map(lambda x: "  ⚠ thin data" if bool(x) else "")
    r = d[CFG["resid"]]

    def _pos(v):
        if v < 0:
            return f"More underfunded than {(r > v).mean():.0%} of countries"
        if v > 0:
            return f"Better funded than {(r < v).mean():.0%} of countries"
        return "Right on the model line"
    d["_pos"] = r.apply(_pos)

    m = float(r.abs().max())
    tickvals, ticktext = _colorbar_ticks(m)

    fig = px.choropleth(
        d, locations=CFG["iso3"], color=CFG["resid"], hover_name=CFG["name"],
        color_continuous_scale="RdBu", color_continuous_midpoint=0, range_color=[-m, m],
        projection="natural earth",
        custom_data=[CFG["profile"], CFG["income"], CFG["region"],
                     CFG["nyears"], "_thin", "_pos"],
    )
    fig.update_traces(
        marker_line_width=0.4, marker_line_color="white",
        hovertemplate=(
            "<b>%{hovertext}</b><br>%{customdata[0]}<br>%{customdata[5]}<br>"
            "Tier %{customdata[1]} · %{customdata[2]}<br>"
            "Years observed: %{customdata[3]}%{customdata[4]}<extra></extra>"
        ),
    )
    # All countries drawn; background kept super-faint (no heavy panel)
    fig.update_geos(
        showframe=False, showcoastlines=False,
        showland=True, landcolor="#edf0f3",        # no-data countries: faint grey
        showcountries=True, countrycolor="#d9dee4", countrywidth=0.4,
        showocean=True, oceancolor="#f7f9fb",       # barely-there ocean
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=MAP_HEIGHT,
        coloraxis_colorbar=dict(title="Misallocation<br>(model residual)",
                                tickmode="array", tickvals=tickvals, ticktext=ticktext),
    )

    # Name the darkest-red extremes directly on the map. Thin-data countries are
    # excluded so a one or two year score is never headlined; a white text halo
    # keeps the labels legible over both pale and deep-red fills.
    if LABEL_N:
        lab = d[~d[CFG["thin"]].astype(bool)].nsmallest(LABEL_N, CFG["resid"])
        lab = lab[lab[CFG["iso3"]].isin(CENTROIDS)]
        if len(lab):
            fig.add_trace(go.Scattergeo(
                lat=[CENTROIDS[i][0] for i in lab[CFG["iso3"]]],
                lon=[CENTROIDS[i][1] for i in lab[CFG["iso3"]]],
                text=[str(n).split(",")[0] for n in lab[CFG["name"]]],
                mode="text",
                textfont=dict(size=11, color="#1b1b1b",
                              shadow="0px 0px 4px rgba(255,255,255,0.95)"),
                hoverinfo="skip", showlegend=False,
            ))

    return fig

LORENZ_LINE = "#2c6e8f"   # teal brand line; not a residual data color


def concentration_facts(df: pd.DataFrame) -> dict:
    """One source of truth for the aid-concentration figures.

    Returns the cumulative arrays the Lorenz curve plots plus the headline
    shares the Overview panel reports, so the chart annotation and the panel
    metrics can never disagree.
    """
    aid = np.sort(df["aid_usd_m_mean"].values.astype(float))
    aid = aid[aid >= 0]
    n = len(aid)
    cum_aid = np.concatenate([[0.0], np.cumsum(aid) / aid.sum()])
    cum_ctry = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    half_idx = int(np.searchsorted(cum_aid, 0.5))
    return {
        "n": n,
        "cum_aid": cum_aid,
        "cum_ctry": cum_ctry,
        "half_idx": half_idx,
        "half_x": float(cum_ctry[half_idx]),
        "top_half": (n - half_idx) / n,                 # top X% hold half of aid
        "bottom50": float(cum_aid[int(round(0.5 * n))]),  # bottom 50% of countries' share
        "top15": float(aid[::-1][:15].sum() / aid.sum()),
    }


def build_lorenz(df: pd.DataFrame):
    """Lorenz curve of adaptation aid across the scored countries.

    The teal curve is the observed cumulative distribution; the dashed diagonal
    is perfect equality. The gap between them is the inequality the Gini
    summarizes. Colors here are brand/neutral and never the diverging residual
    scale used on the map.
    """
    f = concentration_facts(df)
    cum_aid, cum_ctry = f["cum_aid"], f["cum_ctry"]
    half_x, top_share = f["half_x"], f["top_half"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
        line=dict(color="#b8b8b8", dash="dash", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=cum_ctry, y=cum_aid, mode="lines", name="Actual aid",
        line=dict(color=LORENZ_LINE, width=2.6),
        fill="tonexty", fillcolor="rgba(44,110,143,0.12)",
        hovertemplate=("Lowest-funded %{x:.0%} of countries<br>"
                       "hold %{y:.0%} of all aid<extra></extra>")))
    fig.add_trace(go.Scatter(
        x=[half_x], y=[0.5], mode="markers", showlegend=False,
        marker=dict(color=LORENZ_LINE, size=10, line=dict(color="white", width=1.5)),
        hovertemplate=(f"The top {top_share:.0%} of recipients<br>"
                       "hold half of all aid<extra></extra>")))
    fig.add_annotation(x=half_x, y=0.5, text=f"Top {top_share:.0%} hold<br>half the aid",
                       showarrow=True, arrowhead=0, arrowcolor="#9aa3af",
                       ax=40, ay=40, font=dict(size=12, color="#5b6670"),
                       align="left")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Cumulative share of countries",
                     constrain="domain")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Cumulative share of adaptation aid",
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None))
    return fig


RES_RED, RES_BLUE = "#b2182b", "#2166ac"   # residual encoding, same as the map and ranked strip


def build_residual_trajectory(mdf: pd.DataFrame, country: str):
    """Compact per-year residual path for one country.

    Plots misallocation_score (log aid received minus model prediction) by year
    against a zero 'as predicted' baseline. Marker color reuses the map's red/
    blue residual encoding, so the trajectory reads as a temporal echo of the
    country's color on the map. The faint flat line is the country's mean, which
    is exactly the score shown in the panel metric and on the map.
    """
    d = mdf[mdf["country"] == country].sort_values("year")
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#b8b8b8", line_width=1)
    if len(d) >= 3:
        fig.add_hline(y=float(d["misallocation_score"].mean()),
                      line_color="#d8d8d8", line_width=1)
    colors = [RES_RED if v < 0 else RES_BLUE for v in d["misallocation_score"]]
    fig.add_trace(go.Scatter(
        x=d["year"], y=d["misallocation_score"], mode="lines+markers",
        line=dict(color="#9aa3af", width=1.5),
        marker=dict(color=colors, size=7, line=dict(color="white", width=1)),
        hovertemplate="%{x}: %{y:+.2f} vs predicted<extra></extra>",
        showlegend=False, cliponaxis=False))
    fig.update_yaxes(title="Residual (log)", zeroline=False)
    fig.update_xaxes(title=None, dtick=2, tickformat="d")
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=8, b=0))
    return fig