import numpy as np
import streamlit as st
from app.data import load_country_scored, load_model_scored
from app.charts import build_choropleth, build_lorenz, concentration_facts, build_residual_trajectory
from app.details import country_summary


def _gini(values) -> float:
    """Standard Gini coefficient over non-negative values."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[x >= 0]
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _headline_kpis(df):
    """All four landing-strip stats, computed from the scored country table."""
    n_countries = len(df)

    aid = df["aid_usd_m_mean"]
    gini = _gini(aid.values)
    aid_sorted = aid.sort_values(ascending=False)
    top15_share = aid_sorted.head(15).sum() / aid_sorted.sum()

    region_mean = df.groupby("region")["misallocation_mean"].mean().sort_values()
    underfunded = region_mean[region_mean < 0]
    n_regions = len(region_mean)

    return {
        "n_countries": n_countries,
        "gini": gini,
        "top15_share": top15_share,
        "n_under": len(underfunded),
        "n_regions": n_regions,
        "under_regions": underfunded,  # Series: region -> mean gap, most negative first
    }


def _rank_table(frame, value_color):
    """Compact ranked list: index, country, signed residual in the data color."""
    rows = []
    for i, (_, r) in enumerate(frame.iterrows(), 1):
        name = str(r["country"]).split(",")[0]
        thin = " ⚠" if bool(r["thin_data"]) else ""
        rows.append(
            "<tr>"
            f'<td style="color:#9aa3af; padding:.18rem .5rem .18rem 0; width:1.3rem;">{i}</td>'
            f'<td style="padding:.18rem 0;">{name}{thin}</td>'
            '<td style="text-align:right; font-weight:700; font-variant-numeric:tabular-nums; '
            f'color:{value_color}; padding:.18rem 0;">{r["misallocation_mean"]:+.2f}</td>'
            "</tr>"
        )
    return ('<table style="width:100%; border-collapse:collapse; font-size:.95rem;">'
            + "".join(rows) + "</table>")


def _strip_header(text):
    return ('<div style="font-size:.78rem; letter-spacing:.04em; text-transform:uppercase; '
            f'color:#6B7280; margin-bottom:.35rem;">{text}</div>')


def _finding_card(col, title, impact, page, link_label, icon, key):
    """One key-finding card: headline, impact line, and a link to the page that proves it."""
    with col:
        with st.container(border=True, key=key):
            st.markdown(f"**{title}**")
            st.markdown(f"<span style='font-size:.92rem; color:#444;'>{impact}</span>",
                        unsafe_allow_html=True)
            if page is not None:
                st.page_link(page, label=link_label, icon=icon)


def render():
    st.title("Aid follows need on average, but not for everyone")
    st.markdown(
        "Once vulnerability, income, country size, and governance are accounted for, "
        "most countries receive close to what the model predicts. The darkest red "
        "countries are the systematic exceptions: they receive far less adaptation aid "
        "than their risk profile warrants."
    )

    df = load_country_scored()

    # --- Hero KPI strip: the thesis as four numbers, computed live from the data ---
    k = _headline_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries scored", f"{k['n_countries']}")
    c2.metric("Aid concentration (Gini)", f"{k['gini']:.2f}")
    c3.metric("Top 15 recipients' share", f"{k['top15_share']:.0%}")
    c4.metric("Chronically underfunded regions", f"{k['n_under']} of {k['n_regions']}")

    if len(k["under_regions"]):
        named = ", ".join(
            f"{region} ({gap:+.2f})" for region, gap in k["under_regions"].items()
        )
        st.caption(
            f"Concentration is high and need is only one of several pulls on aid. "
            f"Only two regions receive systematically less than their risk warrants: {named}."
        )

    st.write("")

    map_col, rail_col = st.columns([2, 1], gap="medium")
    with map_col:
        with st.container(border=True, key="card-map"):
            st.plotly_chart(build_choropleth(df), use_container_width=True)
            st.caption(
                "Red = under-allocated vs. model · Blue = over-allocated · Pale = on target. "
                "Grey countries are outside the sample. Hover any country for its detail."
            )
    with rail_col:
        with st.container(border=True, key="card-extremes"):
            st.markdown("**The widest gaps, named**")
            RED, BLUE = "#b2182b", "#2166ac"
            under = df.nsmallest(5, "misallocation_mean")
            over = df.nlargest(5, "misallocation_mean")
            st.markdown(_strip_header("Underfunded vs need") + _rank_table(under, RED),
                        unsafe_allow_html=True)
            st.markdown(_strip_header("Over-resourced") + _rank_table(over, BLUE),
                        unsafe_allow_html=True)
            st.caption("Mean residual, log units. ⚠ = one or two years of data.")

    # --- Key findings: the cross-page takeaways, each linking to its evidence ---
    nav = st.session_state.get("_nav_pages", {})
    worst = [str(c).split(",")[0] for c in
             df[~df["thin_data"].astype(bool)].nsmallest(3, "misallocation_mean")["country"]]
    worst_str = f"{worst[0]}, {worst[1]}, and {worst[2]}"

    st.write("")
    st.subheader("Key findings")
    f1, f2, f3 = st.columns(3, gap="medium")
    _finding_card(
        f1, "Aid follows need, but only after size",
        "Vulnerability shapes allocation once population is controlled, yet the model still "
        "leaves about half of all aid unexplained.",
        nav.get("regression"), "How the model works", "📈", key="card-find-model")
    _finding_card(
        f2, "A few countries are systematically short-changed",
        f"{worst_str} receive far less adaptation aid than their risk profile warrants.",
        nav.get("briefs"), "Generate a country brief", "📝", key="card-find-gaps")
    _finding_card(
        f3, "The gaps are widening, not closing",
        "Most regional allocation gaps are projected to drift further from need toward 2030 "
        "rather than close.",
        nav.get("projections"), "See 2030 projections", "🔮", key="card-find-trend")

    # --- Aid concentration ---
    st.write("")
    st.subheader("Most adaptation aid pools in a few large recipients")
    lz, facts = st.columns([3, 2], gap="large")
    with lz:
        st.plotly_chart(build_lorenz(df), use_container_width=True)
    with facts:
        cf = concentration_facts(df)
        st.metric("Lowest-funded half of countries", f"{cf['bottom50']:.0%} of aid")
        st.metric("Top 15 recipients", f"{cf['top15']:.0%} of aid")
        st.metric("Hold half of all aid", f"Top {cf['top_half']:.0%} of countries")
        st.caption(
            "Countries ranked by mean adaptation aid received, 2010 to 2023. "
            "The dashed line marks perfect equality."
        )

    st.divider()
    st.subheader("Country detail")

    countries = sorted(df["country"])
    default_country = df.loc[df["misallocation_mean"].idxmin(), "country"]
    choice = st.selectbox("Select a country", countries,
                          index=countries.index(default_country))

    metrics, country_md, region_md = country_summary(df, choice)

    c1, c2, c3 = st.columns(3)
    c1.metric("Misallocation (log)", f"{metrics['resid']:+.2f}")
    c2.metric("Rank in income tier", f"{metrics['rank']} / {metrics['tier_n']}")
    c3.metric("Vulnerability", f"{metrics['vuln']:.2f}")

    mdf = load_model_scored()
    n_years = int((mdf["country"] == choice).sum())
    st.plotly_chart(build_residual_trajectory(mdf, choice), use_container_width=True)
    if n_years <= 2:
        st.caption(
            f"{choice} has only {n_years} scored year{'' if n_years == 1 else 's'}, "
            "so this trajectory is sparse. Dashed line = as predicted; "
            "red = underfunded that year, blue = over-resourced."
        )
    else:
        st.caption(
            "Year-by-year gap from the model. Dashed line = as predicted (0); the faint "
            "flat line is this country's average, which is the score above and on the map. "
            "Red = underfunded that year, blue = over-resourced."
        )

    st.markdown(country_md)
    st.markdown(region_md)

    briefs_page = st.session_state.get("_nav_pages", {}).get("briefs")
    if briefs_page is not None:
        if st.button(f"Generate a policy brief for {choice}", icon="📝"):
            st.session_state["brief_country"] = choice   # preselect on the Briefs page
            st.session_state["brief_for"] = choice        # generate on arrival
            st.switch_page(briefs_page)
    else:
        st.caption(f"↗ For a full AI-generated policy brief on {choice}, "
                   "open **Policy briefs** in the sidebar.")

    st.divider()
    st.caption("↗ For data sources, the model specification, and limitations, "
               "see **Methods & data** in the sidebar.")