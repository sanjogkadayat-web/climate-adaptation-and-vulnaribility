import numpy as np
import streamlit as st
from app.data import load_country_scored
from app.charts import build_choropleth
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

    with st.container(border=True):
        st.plotly_chart(build_choropleth(df), use_container_width=True)
        st.caption(
            "Red = under-allocated vs. model · Blue = over-allocated · Pale = on target. "
            "Grey countries are outside the sample. Hover any country for its detail."
        )

    st.write("")
    st.subheader("The widest gaps sit in a handful of countries")
    RED, BLUE = "#b2182b", "#2166ac"
    under = df.nsmallest(6, "misallocation_mean")
    over = df.nlargest(6, "misallocation_mean")
    u_col, o_col = st.columns(2)
    u_col.markdown(_strip_header("Most underfunded vs need") + _rank_table(under, RED),
                   unsafe_allow_html=True)
    o_col.markdown(_strip_header("Most over-resourced") + _rank_table(over, BLUE),
                   unsafe_allow_html=True)
    st.caption(
        "Mean model residual in log units: red countries received less adaptation aid than "
        "predicted, blue more. ⚠ marks a country scored on only one or two years. "
        "The map names the deepest red cases directly."
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

    st.markdown(country_md)
    st.markdown(region_md)

    st.caption(f"↗ For a full AI-generated policy brief on {choice}, "
               "open **Policy briefs** in the sidebar.")

    st.divider()
    st.caption("↗ For data sources, the model specification, and limitations, "
               "see **Methods & data** in the sidebar.")