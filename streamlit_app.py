import streamlit as st

st.set_page_config(page_title="Climate Adaptation Aid", layout="wide")

from app.views import overview, regression, projections, briefs

st.navigation([
    st.Page(overview.render,    title="Overview map",     icon="🗺️", url_path="overview",    default=True),
    st.Page(regression.render,  title="Regression",       icon="📈", url_path="regression"),
    st.Page(projections.render, title="2030 projections", icon="🔮", url_path="projections"),
    st.Page(briefs.render,      title="Policy briefs",    icon="📝", url_path="briefs"),
]).run()