from pathlib import Path
import streamlit as st

ASSETS = Path(__file__).parent / "app" / "assets"

st.set_page_config(
    page_title="TRACE",
    page_icon=str(ASSETS / "TRACE_icon.png"),
    layout="wide",
)

# Sidebar logo, top-left — globe only (the name lives in the top strip below)
st.logo(
    str(ASSETS / "TRACE_icon.png"),
    icon_image=str(ASSETS / "TRACE_icon.png"),
    size="large",
)

# --- Brand chrome: enlarge the logo + add the full-name top strip ---
st.markdown(
    """
    <style>
      /* KNOB 1: sidebar logo size. This is the dial — raise to 4.5rem / 5rem for bigger. */
      img[data-testid="stLogo"] { height: 4rem !important; width: auto !important; }
      /* let the header grow with the larger logo instead of clipping it */
      div[data-testid="stSidebarHeader"] { height: auto !important; padding-top: 0.4rem; }

      /* KNOB 2: top breathing room. Raise 3.6rem to push the name strip lower. */
      .block-container { padding-top: 3.6rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Persistent full-name strip across the top of every page
st.markdown(
    """
    <div style="display:flex; align-items:baseline; gap:0.65rem; flex-wrap:wrap;
                padding:0 0 0.55rem 0; margin-bottom:1.6rem;
                border-bottom:1px solid #ECECEC;">
      <span style="font-weight:800; letter-spacing:2.5px; color:#1F2A37;
                   font-size:1.05rem;">TRACE</span>
      <span style="color:#6B7280; font-size:0.95rem; font-weight:400;">
        Tracking Resource Allocation for Climate Equity
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

from app.views import overview, regression, projections, briefs, methods

st.navigation([
    st.Page(overview.render,    title="Overview map",     icon="🗺️", url_path="overview",    default=True),
    st.Page(regression.render,  title="Regression",       icon="📈", url_path="regression"),
    st.Page(projections.render, title="2030 projections", icon="🔮", url_path="projections"),
    st.Page(briefs.render,      title="Policy briefs",    icon="📝", url_path="briefs"),
    st.Page(methods.render,     title="Methods & data",   icon="📋", url_path="methods"),
]).run()