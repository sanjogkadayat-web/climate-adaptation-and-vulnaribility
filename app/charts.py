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

MAP_HEIGHT = 800   # master size dial: raise to make the whole map bigger
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