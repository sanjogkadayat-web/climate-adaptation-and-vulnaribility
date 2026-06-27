import numpy as np
import plotly.express as px
import pandas as pd

CFG = {
    "iso3": "iso3", "name": "country", "resid": "misallocation_mean",
    "income": "income_tier", "region": "region", "nyears": "n_years",
    "profile": "profile", "thin": "thin_data",
}


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
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=520,
        coloraxis_colorbar=dict(title="Misallocation<br>(model residual)",
                                tickmode="array", tickvals=tickvals, ticktext=ticktext),
    )
    # base layer: every country shown as pale land with thin borders, data colored on top
    fig.update_geos(
        showframe=False, showcoastlines=False,
        showland=True, landcolor="#eef1f4",
        showcountries=True, countrycolor="#cbd2da", countrywidth=0.4,
        showocean=False, bgcolor="rgba(0,0,0,0)",
    )
    return fig