import numpy as np
import plotly.express as px
import pandas as pd

# Column names confirmed against country_scored.csv
CFG = {
    "iso3":    "iso3",
    "name":    "country",
    "resid":   "misallocation_mean",   # residual from a log1p(aid) model
    "income":  "income_tier",          # modal tier
    "region":  "region",
    "nyears":  "n_years",
    "profile": "profile",              # Chronically Underfunded ... Over-Resourced
    "thin":    "thin_data",            # True when based on only 1-2 years
}


def _colorbar_ticks(m: float):
    """Every tick carries the precise residual plus a plain-language companion."""
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

    # position within the sample, consistent with the colored residual
    r = d[CFG["resid"]]
    def _pos(v):
        if v < 0:
            return f"More underfunded than {(r > v).mean():.0%} of countries"
        if v > 0:
            return f"Better funded than {(r < v).mean():.0%} of countries"
        return "Right on the model line"
    d["_pos"] = r.apply(_pos)

    # symmetric range so 0 sits at the white midpoint of the diverging scale
    m = float(r.abs().max())
    tickvals, ticktext = _colorbar_ticks(m)

    fig = px.choropleth(
        d,
        locations=CFG["iso3"],
        color=CFG["resid"],
        hover_name=CFG["name"],
        color_continuous_scale="RdBu",   # negative (underfunded) = red, positive = blue
        color_continuous_midpoint=0,
        range_color=[-m, m],
        projection="natural earth",
        custom_data=[CFG["profile"], CFG["income"], CFG["region"],
                     CFG["nyears"], "_thin", "_pos"],
    )
    fig.update_traces(
        marker_line_width=0.3,
        marker_line_color="white",
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "%{customdata[0]}<br>"
            "%{customdata[5]}<br>"
            "Tier %{customdata[1]} · %{customdata[2]}<br>"
            "Years observed: %{customdata[3]}%{customdata[4]}<extra></extra>"
        ),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        coloraxis_colorbar=dict(
            title="Misallocation<br>(model residual)",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
    )
    fig.update_geos(showframe=False, showcoastlines=False)
    return fig