import streamlit as st
from app.data import load_country_scored


def render():
    st.title("Climate adaptation aid vs. vulnerability")
    df = load_country_scored()
    st.success(f"Loaded country_scored.csv — {df.shape[0]} countries, {df.shape[1]} columns.")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption("Choropleth lands here in Step 3.")