import pandas as pd
import streamlit as st

from app.data import PROCESSED

_EXCL_DISPLAY = {
    "Palestine, State of": "Palestine",
    "Korea, Democratic People's Republic of": "North Korea",
    "Saint Kitts and Nevis": "Saint Kitts and Nevis",
    "South Sudan": "South Sudan",
    "Kosovo": "Kosovo",
}
_INPUT_LABEL = {
    "vulnerability": "ND-GAIN vulnerability",
    "gdp_per_capita_usd": "World Bank GDP per capita",
    "gov_effectiveness": "WGI governance",
    "income_group": "World Bank income class",
}


@st.cache_data(show_spinner=False)
def _load(name):
    return pd.read_csv(PROCESSED / name)


@st.cache_data(show_spinner=False)
def _csv(name):
    return _load(name).to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def _exclusions():
    """Countries in the raw panel but not scored, with the exact input(s) they lack."""
    pan = _load("panel.csv")
    scored = set(_load("country_scored.csv")["country"])
    rows = []
    for c in sorted(set(pan["country"]) - scored):
        d = pan[pan["country"] == c]
        missing = "; ".join(lbl for col, lbl in _INPUT_LABEL.items()
                            if col in d.columns and d[col].isna().all())
        rows.append({
            "Country": _EXCL_DISPLAY.get(c, c),
            "Missing model input(s)": missing,
            "Years in raw data": f"{int(d['year'].min())} to {int(d['year'].max())}",
            "Adaptation aid received": f"${d['adaptation_aid_usd_m'].sum():,.0f}M",
            "_sort": d["adaptation_aid_usd_m"].sum(),
        })
    out = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    return out.reset_index(drop=True)


def render():
    st.title("Methods, data, and limitations")
    st.markdown(
        "This page documents every consequential choice behind the misallocation scores, why it "
        "was made, and how it shaped the result, along with exactly what the analysis covers and "
        "what it leaves out. The aim is that a reader can reconstruct and challenge the analysis, "
        "not just take its outputs on trust."
    )

    st.subheader("Data sources")
    st.markdown(
        "The analysis merges four sources into a country-year panel of 138 countries, 2010 to 2023:\n"
        "- **OECD Rio Markers**: bilateral and multilateral aid flagged as climate adaptation, the "
        "dependent variable.\n"
        "- **ND-GAIN**: each country's climate vulnerability score, the study's measure of need.\n"
        "- **World Bank**: GDP per capita, Government Effectiveness (WGI), income classification, "
        "and population.\n"
        "- **UN lists**: landlocked and small-island developing state (SIDS) status."
    )

    st.subheader("The model")
    st.markdown("Adaptation aid is modeled with pooled OLS on a log(1 plus aid) scale:")
    st.code("log(1 + aid) ~ vulnerability + log GDP per capita + log population\n"
            "               + governance + landlocked + SIDS + income tier\n"
            "               + year fixed effects", language="text")
    st.markdown(
        "Standard errors are clustered by country and year fixed effects are included. The residual, "
        "how much more or less a country receives than predicted, is its misallocation score; a "
        "country's map score is the mean residual across its observed years. The full coefficient "
        "table is on the Regression page."
    )

    st.subheader("Why each modeling choice, and what it changed")

    st.markdown("**Specification**")
    st.markdown(
        "- **Outcome on a log(1 + aid) scale.** Adaptation aid is heavily right-skewed across "
        "orders of magnitude, so an untransformed model would be dominated by a few large "
        "recipients. The log makes effects proportional and residuals readable as percentage over- "
        "or under-funding; the plus-one retains country-years with zero recorded aid rather than "
        "dropping them. *Effect:* every score in the app is a relative position in log units, not a "
        "dollar gap.\n"
        "- **Year fixed effects.** Global adaptation finance rose steadily over the period, so "
        "without year controls a country could look like it is catching up simply because the pool "
        "grew. Year effects hold each year's total constant and compare a country to its peers that "
        "year. *Effect:* misallocation is a within-year relative measure, which is exactly why "
        "comparing residuals across years is uninformative by design and why the persistence "
        "question is tested with an interaction rather than a raw year-on-year read.\n"
        "- **Standard errors clustered by country.** Each country appears up to 14 times and those "
        "observations are correlated; treating them as independent would overstate precision. "
        "*Effect:* several predictors with sizable point estimates (landlocked status, the "
        "income-tier dummies) fail the 95% test once clustered, which stops the analysis from "
        "claiming more certainty than the data supports."
    )

    st.markdown("**Identifying the need effect**")
    st.markdown(
        "- **Controlling for log population, the pivotal choice.** Large countries draw more aid "
        "almost mechanically, so size confounds need. *Effect:* this is the single most "
        "consequential decision in the study. Without population, vulnerability is statistically "
        "insignificant (p ≈ 0.5) and the headline would read 'aid ignores need.' Adding log "
        "population flips it: vulnerability turns positive and significant (p ≈ 0.007), changing the "
        "finding to 'aid tracks need once size is controlled, but specific countries deviate.'\n"
        "- **Dropping the ND-GAIN readiness score.** Readiness correlates with vulnerability at "
        "r ≈ −0.82, near-collinear, so including both would split the need signal across two "
        "unstable coefficients. *Effect:* a single, clean, interpretable need effect with no "
        "variance inflation.\n"
        "- **Income tier as the modal label, not the latest.** 54 of 138 countries changed World "
        "Bank income group at least once over the period, so the most recent class would "
        "misrepresent a country's typical standing; the modal tier is the stable representative. "
        "*Effect:* the tier control and the tier-relative profiles reflect where a country usually "
        "sat, not an end-of-period snapshot.\n"
        "- **Governance, GDP per capita, and geography as controls.** These are the main competing "
        "explanations for aid, absorptive capacity, wealth, and structural disadvantage, so "
        "including them makes the residual need-deviation net of them rather than a proxy for them. "
        "*Effect:* landlocked and small-island status come out null, which strengthens the claim "
        "that the residual is genuine misallocation rather than unmodeled geography."
    )

    st.markdown("**From residuals to scores**")
    st.markdown(
        "- **Country score as the mean residual across observed years.** A single stable score per "
        "country is needed for the map and rankings, and averaging smooths year-to-year noise. "
        "*Effect:* countries with only one or two years produce means that rest on thin evidence, so "
        "they are flagged as thin data and kept out of headline labels.\n"
        "- **Funding profiles built on tier-centered residuals.** Labeling profiles from raw "
        "residuals would load the underfunded end with large middle-income countries (China, Iran) "
        "for tier-structural reasons; centering within income tier first isolates deviation relative "
        "to peers. *Effect:* the profile labels are peer-relative and avoid income-tier artifacts "
        "that would otherwise distort the country lists."
    )

    st.markdown("**Projections**")
    st.markdown(
        "- **Full aid panel for projections, not the model sample.** The regression uses complete "
        "cases, which drops countries missing a control in some years; trends are instead fit on the "
        "full aid panel. *Effect:* this recovers the aid history of thin-but-real recipients such as "
        "Eritrea (14 years of aid) that the complete-case model would otherwise lose.\n"
        "- **Transparent trend extrapolation with a robustness check.** Each country's log-aid trend "
        "is fit and extended to 2030, cross-checked against a Theil-Sen slope to limit the pull of "
        "outlier years. *Effect:* projections are deliberately extrapolations, not forecasts; they "
        "assume the 2010 to 2023 path continues and carry wide uncertainty, stated on the "
        "Projections page."
    )

    st.subheader("Coverage and exclusions")
    st.markdown(
        "138 countries are scored. Five appear in the raw recipient data but cannot be scored "
        "because they are missing at least one model input. The dominant gap is the ND-GAIN "
        "vulnerability score, which is the study's measure of need: without it, a "
        "misallocation-relative-to-need score is undefined."
    )
    st.dataframe(_exclusions(), hide_index=True, use_container_width=True)
    st.markdown(
        "**Does this matter? For three of them, not much.** North Korea and Saint Kitts and Nevis "
        "are very small recipients (roughly $13M and $18M across the period), and Kosovo, which "
        "lacks four inputs because of its contested status, is a moderate recipient whose absence "
        "does not move the central results. **For two, it is a real limitation worth stating "
        "plainly.** Palestine (about $2.0B) and South Sudan (about $1.8B) are among the larger "
        "adaptation-aid recipients in the raw data, and South Sudan in particular is plausibly "
        "high-need. They are excluded not by choice but because ND-GAIN publishes no vulnerability "
        "score for them, so the analysis cannot place two of the bigger recipients, a gap a reader "
        "should keep in mind. Countries left blank on the map fall outside this recipient sample "
        "entirely."
    )
    thin = list(_load("country_scored.csv").query("thin_data == True")["country"])
    st.markdown(
        f"**Thin data.** {len(thin)} scored countries rest on only one or two observed years "
        f"({', '.join(thin)}). They are kept in the panel so the coverage is complete, but flagged "
        "everywhere and excluded from headline rankings and on-map labels, since a mean over one or "
        "two years is fragile."
    )

    st.subheader("Assumptions and caveats")
    st.markdown(
        "- A misallocation score is **model-relative**: a negative value means a country receives "
        "less than the model predicts, not a verdict that it is treated unfairly. About half of "
        "allocation is unexplained, and some of that may be legitimate unobserved factors.\n"
        "- Scores reflect **relative position, not dollar amounts**, a consequence of the log scale.\n"
        "- Adaptation aid is **donor-reported** through the Rio Markers, an imperfect and sometimes "
        "over-claimed proxy for true adaptation finance.\n"
        "- The model is **associational, not causal**: it describes how aid is distributed, not why "
        "any single allocation decision was made.\n"
        "- Projections are **extrapolations, not forecasts**, and assume past trends hold."
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