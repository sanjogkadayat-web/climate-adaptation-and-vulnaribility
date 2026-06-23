import streamlit as st
from app.data import load_country_scored
from app.charts import build_choropleth
from app.details import country_summary


def render():
    st.title("Aid follows need on average, but not for everyone")
    st.markdown(
        "Once vulnerability, income, country size, and governance are accounted for, "
        "most countries receive close to what the model predicts. The darkest red "
        "countries are the systematic exceptions: they receive far less adaptation aid "
        "than their risk profile warrants."
    )
    df = load_country_scored()
    st.plotly_chart(build_choropleth(df), use_container_width=True)
    st.caption(
        "Red = under-allocated vs. model · Blue = over-allocated · Pale = on target. "
        "Hover any country for its profile, value, tier, and years observed."
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