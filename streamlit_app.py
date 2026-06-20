import streamlit as st

st.set_page_config(page_title="Climate Adaptation Aid", layout="wide")

from app.views import overview, regression, projections, briefs

st.navigation([
    st.Page(overview.render,    title="Overview map",     icon="🗺️", default=True),
    st.Page(regression.render,  title="Regression",       icon="📈"),
    st.Page(projections.render, title="2030 projections", icon="🔮"),
    st.Page(briefs.render,      title="Policy briefs",    icon="📝"),
]).run()