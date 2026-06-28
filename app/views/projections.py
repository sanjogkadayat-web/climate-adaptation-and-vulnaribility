import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data import PROCESSED

COL = {"closing": "#2c8f6e", "widening": "#c0573b"}
PROFILE_ORDER = ["Chronically Underfunded", "Underfunded", "Adequately Funded", "Over-Resourced"]
PROFILE_COL = {"Chronically Underfunded": "#b2182b", "Underfunded": "#ef8a62",
               "Adequately Funded": "#bdbdbd", "Over-Resourced": "#2c6e8f"}


@st.cache_data(show_spinner=False)
def _load_regional():
    return pd.read_csv(PROCESSED / "regional_gap_trend.csv")


@st.cache_data(show_spinner=False)
def _load_country_proj():
    return pd.read_csv(PROCESSED / "country_projection.csv")


@st.cache_data(show_spinner=False)
def _load_panel():
    return pd.read_csv(PROCESSED / "panel.csv")


def _regional_dumbbell(reg):
    r = reg.sort_values("gap_2030").reset_index(drop=True)
    fig = go.Figure()
    for _, x in r.iterrows():
        fig.add_trace(go.Scatter(
            x=[x.gap_2023, x.gap_2030], y=[x.region, x.region], mode="lines",
            line=dict(color=COL[x.status], width=3), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=r.gap_2023, y=r.region, mode="markers", name="2023",
        marker=dict(color="#b8b8b8", size=11, line=dict(color="white", width=1)),
        hovertemplate="%{y}<br>2023 gap %{x:+.2f}<extra></extra>"))
    for status, label in [("closing", "Closing by 2030"), ("widening", "Widening by 2030")]:
        s = r[r.status == status]
        fig.add_trace(go.Scatter(
            x=s.gap_2030, y=s.region, mode="markers", name=label,
            marker=dict(color=COL[status], size=13),
            hovertemplate="%{y}<br>2030 gap %{x:+.2f}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=list(r.region), title=None)
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="0 = needs-based", annotation_position="top")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_title="Misallocation gap (log units)", legend_title=None)
    return fig


def _country_forecast(row, hist):
    yrs = np.arange(2010, 2031)
    trend = np.exp(np.log(row.aid_2023_fit_usd_m) + row.slope * (yrs - 2023))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist["adaptation_aid_usd_m"], mode="markers", name="Actual aid",
        marker=dict(color="#2c6e8f", size=8), hovertemplate="%{x}<br>$%{y:.1f}M<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=yrs[yrs <= 2023], y=trend[yrs <= 2023], mode="lines", name="Fitted trend",
        line=dict(color="#2c6e8f")))
    fig.add_trace(go.Scatter(
        x=yrs[yrs >= 2023], y=trend[yrs >= 2023], mode="lines", name="Projected to 2030",
        line=dict(color="#2c6e8f", dash="dash")))
    fig.update_xaxes(dtick=2, title="Year")
    fig.update_yaxes(title="Adaptation aid (USD millions)")
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None))
    return fig


def _growth_box(cp):
    fig = px.box(cp, x="profile", y="annual_pct", color="profile",
                 category_orders={"profile": PROFILE_ORDER}, color_discrete_map=PROFILE_COL,
                 points="outliers")
    med = cp["annual_pct"].median()
    fig.add_hline(y=med, line_dash="dash", line_color="gray",
                  annotation_text=f"overall median {med:.0f}%/yr", annotation_position="top left")
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                      xaxis_title=None, yaxis_title="Annual aid growth (%/yr)",
                      yaxis_range=[-20, 50])
    return fig


def _country_meaning(row):
    """What the selected country's forecast means, tailored to profile and direction.
    Deliberately keyed to profile/direction/significance rather than the percent figure,
    which is extremely volatile for some countries."""
    country = row["country"]
    prof = row["profile"]
    pl = prof.lower()
    direction = str(row["direction"]).lower()  # rising / falling / flat
    under = prof in ("Chronically Underfunded", "Underfunded")
    over = prof == "Over-Resourced"

    if under and direction == "rising":
        body = (
            f"**What it shows.** {country}'s adaptation aid is on a rising path through 2030.\n\n"
            f"**The takeaway.** It starts {pl}, and rising dollars are not the same as catching up: "
            f"on this trajectory it is projected to remain below what its risk warrants.\n\n"
            f"**Why it matters.** Because aid grows at similar rates across funding profiles (see "
            f"the next tab), {country}'s relative shortfall persists unless donors steer new money "
            f"toward it rather than only enlarging the pool.")
    elif under and direction == "falling":
        body = (
            f"**What it shows.** {country} is already {pl}, and its aid is trending down.\n\n"
            f"**The takeaway.** A falling path on top of an existing shortfall means the gap is set "
            f"to widen further.\n\n"
            f"**Why it matters.** This is the profile most in need of a deliberate course "
            f"correction; the trend on its own will not fix it.")
    elif under:  # flat
        body = (
            f"**What it shows.** {country} is {pl}, and its aid is projected to stay roughly flat.\n\n"
            f"**The takeaway.** A flat path leaves the existing shortfall unaddressed.\n\n"
            f"**Why it matters.** Standing still here means falling further behind as climate need "
            f"grows.")
    elif over and direction == "falling":
        body = (
            f"**What it shows.** {country} receives more adaptation aid than the model predicts, and "
            f"that aid is trending down.\n\n"
            f"**The takeaway.** Its lead is narrowing, moving it back toward need.\n\n"
            f"**Why it matters.** This is allocation self-correcting, the exception rather than the "
            f"rule in this data.")
    elif over:  # rising or flat
        body = (
            f"**What it shows.** {country} already receives more adaptation aid than the model "
            f"predicts, and the trajectory is still {direction}.\n\n"
            f"**The takeaway.** Continued aid at this level widens that lead rather than redirecting "
            f"funds toward needier recipients.\n\n"
            f"**Why it matters.** Every dollar above need here is one not reaching an underfunded "
            f"country.")
    else:  # Adequately Funded
        body = (
            f"**What it shows.** {country} currently tracks need closely.\n\n"
            f"**The takeaway.** Its {direction} trajectory keeps it roughly on target through 2030.\n\n"
            f"**Why it matters.** Countries like this are where allocation is already working as "
            f"intended.")

    if not bool(row["trend_significant"]):
        n = int(row["n_years"])
        body += (
            f"\n\n_Caveat: this trend is not statistically significant ({n} observed years and a "
            f"noisy series), so read the 2030 figure as illustrative, not a firm forecast._")
    return body


def render():
    st.title("Aid is rising almost everywhere, but the allocation gaps mostly aren't closing")
    st.markdown(
        "Two forward views to 2030, plus the reason the gaps persist. Adaptation aid grew for "
        "87% of countries over 2010 to 2023, so a rising dollar trajectory is the norm and says "
        "little on its own. The real question is whether allocation is moving toward need, and "
        "for most regions it is not: four of six gaps are projected to widen."
    )

    t_region, t_country, t_persist = st.tabs(
        ["Regional gaps to 2030", "Country aid forecast", "Will the gap close on its own?"])

    with t_region:
        reg = _load_regional()
        st.subheader("Most regions are drifting further from need-based aid")
        st.plotly_chart(_regional_dumbbell(reg), use_container_width=True, key="regional_dumbbell")
        needy_close = reg[(reg.status == "closing") & (reg.gap_2023 < 0)]
        needy_widen = reg[(reg.status == "widening") & (reg.gap_2023 < 0)].sort_values("slope")
        n_widen = int((reg.status == "widening").sum())
        n_close = int((reg.status == "closing").sum())

        def _gap_delta(rows):
            """Change in gap MAGNITUDE, 2023 to 2030, for the lead row.
            Positive = drifting away from need (widening); negative = toward need."""
            if not len(rows):
                return None
            r0 = rows.iloc[0]
            return abs(r0["gap_2030"]) - abs(r0["gap_2023"])

        d_close = _gap_delta(needy_close)
        d_widen = _gap_delta(needy_widen)

        # delta_color="inverse" fixes one convention across all three cards:
        # red/up = widening (away from need), green/down = closing (toward need).
        # That holds whether the delta is a gap change or the widen-vs-close count.
        a, b, c = st.columns(3)
        a.metric("Regions widening", f"{n_widen} of {len(reg)}",
                 f"{n_widen - n_close:+d} vs closing", delta_color="inverse")
        b.metric("Closing, and still needy",
                 needy_close.iloc[0]["region"] if len(needy_close) else "—",
                 f"{d_close:+.2f} gap" if d_close is not None else None,
                 delta_color="inverse")
        c.metric("Widening fastest, already needy",
                 needy_widen.iloc[0]["region"] if len(needy_widen) else "—",
                 f"{d_widen:+.2f} gap" if d_widen is not None else None,
                 delta_color="inverse")
        st.caption("Grey dot is the 2023 gap, colored dot the 2030 projection; the dashed line is "
                   "a needs-based allocation (gap = 0). South Asia's path rests on only about nine "
                   "countries, so it is more volatile, though the trend is significant (p = 0.03).")

        close_region = needy_close.iloc[0]["region"] if len(needy_close) else None
        widen_region = needy_widen.iloc[0]["region"] if len(needy_widen) else None
        with st.container(border=True, key="card-region-meaning"):
            st.markdown(
                "**What it shows.** The dumbbell tracks each region's allocation gap, the distance "
                "between the aid it receives and what its need warrants, not its dollar total. Left "
                "of the dashed line is under-allocated; movement away from the line means the gap "
                "is widening.")
            takeaway = (f"**The takeaway.** {n_widen} of {len(reg)} regions are projected to drift "
                        f"further from need by 2030.")
            if close_region:
                takeaway += (f" The only needy region narrowing its gap is {close_region}, and even "
                             f"it stays underfunded.")
            if widen_region:
                takeaway += (f" Among regions that are both needy and worsening, {widen_region} is "
                             f"widening fastest.")
            st.markdown(takeaway)
            st.markdown(
                "**Why it matters.** Because aid grows at roughly the same rate regardless of how "
                "well a region is already funded (see the next tab), these gaps will not "
                "self-correct. Closing them means deliberately redirecting new aid toward "
                "under-allocated regions rather than enlarging the pool for everyone.")

    with t_country:
        cp = _load_country_proj()
        panel = _load_panel()
        countries = sorted(cp["country"].unique())
        cand = cp[(cp["profile"] == "Chronically Underfunded") & cp["trend_significant"]
                  & (cp["n_years"] == 14)].sort_values("trend_r2", ascending=False)
        default = cand.iloc[0]["country"] if len(cand) else countries[0]
        country = st.selectbox("Select a country", countries,
                               index=countries.index(default), key="proj_country")
        row = cp[cp["country"] == country].iloc[0]
        hist = panel[panel["iso3"] == row["iso3"]].sort_values("year")
        st.subheader(f"{country}: {row['profile'].lower()}, with aid {row['direction'].lower()}")
        st.plotly_chart(_country_forecast(row, hist), use_container_width=True, key="country_forecast")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projected 2030 aid", f"${row['aid_2030_proj_usd_m']:,.0f}M",
                  f"{row['pct_change_23_30']:+.0f}% vs 2023", delta_color="off")
        m2.metric("Annual growth", f"{row['annual_pct']:+.1f}%/yr")
        m3.metric("Trend fit (R²)", f"{row['trend_r2']:.2f}")
        m4.metric("Trend", "Significant" if row["trend_significant"] else "Not significant")
        st.caption("Dots are actual aid; the line is the fitted log-aid trend, dashed past 2023. "
                   "A projection assumes the 2010 to 2023 trend continues, and since 87% of "
                   "countries are rising, the direction alone does not show whether a country is "
                   "catching up.")

        with st.container(border=True, key="card-country-meaning"):
            st.markdown(_country_meaning(row))

    with t_persist:
        cp = _load_country_proj()
        st.subheader("Aid grows at similar rates regardless of how a country is funded")
        st.plotly_chart(_growth_box(cp), use_container_width=True, key="growth_box")

        i1, i2, i3 = st.columns(3)
        i1.metric("Typical growth, any profile", "~12%/yr")
        i2.metric("Difference by profile", "Not significant")
        i3.metric("Does the gap self-correct?", "No")

        left, right = st.columns(2)
        with left:
            with st.container(border=True, key="card-persist-tide"):
                st.markdown("**A rising tide, not a correction**")
                st.markdown(
                    "Every funding profile clusters around the same roughly 12% annual growth, so "
                    "a country's aid expands at about the same pace whether it is starved or "
                    "favored. Growth lifts all recipients together and leaves the relative gaps "
                    "intact."
                )
        with right:
            with st.container(border=True, key="card-persist-structural"):
                st.markdown("**The gap is structural, not temporary**")
                st.markdown(
                    "Underfunding is built in, not a lag that time will fix. The regional view "
                    "bears this out, with four of six gaps widening through 2030. Closing them "
                    "would take deliberately steering new aid toward the underfunded rather than "
                    "enlarging the pool for everyone, the lever the data shows is missing."
                )