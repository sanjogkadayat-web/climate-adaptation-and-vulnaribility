import pandas as pd
import streamlit as st

from app.data import PROCESSED


@st.cache_data(show_spinner=False)
def _load(name):
    return pd.read_csv(PROCESSED / name)


@st.cache_data(show_spinner=False)
def _csv(name):
    return _load(name).to_csv(index=False).encode("utf-8")


def render():
    st.title("Methods, data, and limitations")
    st.markdown(
        "This page documents how the misallocation scores are produced, what the analysis "
        "covers, and where to download the underlying tables."
    )

    st.subheader("Data sources")
    st.markdown(
        "The analysis merges five inputs into a country-year panel covering 138 countries from "
        "2010 to 2023:\n"
        "- **OECD Rio Markers**: bilateral and multilateral aid flagged as climate adaptation, "
        "the dependent variable.\n"
        "- **ND-GAIN**: each country's climate vulnerability score.\n"
        "- **World Bank**: GDP per capita, Government Effectiveness (WGI), income classification, "
        "and population.\n"
        "- **UN lists**: landlocked and small-island developing state (SIDS) status."
    )

    st.subheader("The model")
    st.markdown(
        "Adaptation aid is modeled with pooled OLS on a log(1 plus aid) scale:"
    )
    st.code("log(1 + aid) ~ vulnerability + log GDP per capita + log population\n"
            "               + governance + landlocked + SIDS + income tier\n"
            "               + year fixed effects", language="text")
    st.markdown(
        "Standard errors are clustered by country, and year fixed effects absorb the growing "
        "global aid pool. The residual from this model, how much more or less a country receives "
        "than predicted, is its misallocation score. Country-level scores on the map are the mean "
        "residual across a country's observed years. The full coefficient table is on the "
        "Regression page."
    )

    st.subheader("Coverage and exclusions")
    st.markdown(
        "138 countries are scored. Five appear in the raw data but are not scored because one or "
        "more model inputs (GDP per capita, governance, or population) were unavailable: North "
        "Korea, Palestine, Kosovo, South Sudan, and Saint Kitts and Nevis. Countries left blank "
        "on the map fall outside this recipient sample. Five scored countries rest on only one or "
        "two years of data and are flagged as thin data."
    )

    st.subheader("Key assumptions and caveats")
    st.markdown(
        "- A misallocation score is **model-relative**: a negative value means a country receives "
        "less than the model predicts, not a verdict that it is treated unfairly. About half of "
        "allocation is unexplained, and some of that may reflect legitimate unobserved factors.\n"
        "- The log scale means scores reflect **relative position, not dollar amounts**.\n"
        "- Income tier uses each country's **most common** classification over the period.\n"
        "- Adaptation aid is **donor-reported** through the Rio Markers, an imperfect proxy for "
        "true adaptation finance.\n"
        "- Projections assume past trends continue and are **extrapolations, not forecasts**."
    )

    st.subheader("The data")
    st.caption("The processed tables behind the app. Download to inspect or reuse.")
    t1, t2 = st.tabs(["country_scored (138 countries)", "model_scored (1,806 country-years)"])
    with t1:
        st.dataframe(_load("country_scored.csv"), use_container_width=True, height=360)
        st.download_button("Download country_scored.csv", _csv("country_scored.csv"),
                           file_name="country_scored.csv", mime="text/csv", key="dl_country")
    with t2:
        st.dataframe(_load("model_scored.csv"), use_container_width=True, height=360)
        st.download_button("Download model_scored.csv", _csv("model_scored.csv"),
                           file_name="model_scored.csv", mime="text/csv", key="dl_model")

    st.caption(
        "Raw source datasets (OECD, ND-GAIN, World Bank, UN) are available in the project "
        "repository, with credit to each provider."
    )