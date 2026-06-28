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

# --- Paper background: warm the canvas so cards and charts read as layered ---
# The main area gets a faint paper tone; the sidebar keeps its own grey (nav
# separation) and the header goes transparent so the paper shows through. All
# charts render with Streamlit's transparent plotly theme, so they blend in
# rather than floating as white rectangles. This is chrome only: no DATA color
# is touched. PAPER is the dial.
PAPER = "#FAFAF8"
st.markdown(
    f"""
    <style>
      [data-testid="stApp"] {{ background-color: {PAPER}; }}
      [data-testid="stHeader"] {{ background: transparent; }}

      /* White cards: any st.container(border=True, key="card-...") lifts off the
         paper canvas. The st-key class lands on the same element as the border,
         so we soften that border and let a soft shadow define the card. */
      [class*="st-key-card-"] {{
        background-color: #ffffff;
        border-color: rgba(16, 24, 40, 0.06) !important;
        border-radius: 0.6rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06), 0 1px 2px rgba(16, 24, 40, 0.04);
      }}

      /* Captions carry most of the explanatory load, so darken them from the
         faint default grey to a more legible secondary tone. */
      [data-testid="stCaptionContainer"],
      [data-testid="stCaptionContainer"] p {{ color: #565d66 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Per-page color accents
# ---------------------------------------------------------------------------
# Chrome-only palette. These tint the title underline and dividers so each page
# has its own identity. GUARDRAIL: these never touch DATA colors. The map RdBu
# scale, the closing/widening green-red, and the quartile profile colors are all
# set inside the views and are untouched here.
PAGE_ACCENTS = {
    "overview":    "#2c6e8f",  # teal
    "regression":  "#5b6cb8",  # indigo
    "projections": "#b5832e",  # amber
    "briefs":      "#7d5ba6",  # purple
    "methods":     "#5b6670",  # slate
}
_DEFAULT_ACCENT = PAGE_ACCENTS["overview"]


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex string to an rgba() string at the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
from app.views import overview, regression, projections, briefs, methods

nav_pages = {
    "overview":    st.Page(overview.render,    title="Home",             icon="🏠", url_path="overview", default=True),
    "regression":  st.Page(regression.render,  title="Regression",       icon="📈", url_path="regression"),
    "projections": st.Page(projections.render, title="2030 projections", icon="🔮", url_path="projections"),
    "briefs":      st.Page(briefs.render,      title="Policy briefs",    icon="📝", url_path="briefs"),
    "methods":     st.Page(methods.render,     title="Methods & data",   icon="📋", url_path="methods"),
}
pg = st.navigation(list(nav_pages.values()))

# Expose the page objects so a view can trigger cross-page navigation through
# st.switch_page (e.g. Overview -> Policy briefs for the selected country).
# Function-based pages can only be switched to via their StreamlitPage object,
# so the view reaches them here rather than by file path.
st.session_state["_nav_pages"] = nav_pages

# The default page reports url_path == "" (Streamlit serves it at the app root),
# so `or "overview"` maps that empty string back to the Overview accent. Every
# other page returns its own url_path. This is robust either way: if a future
# Streamlit returns "overview" for the default page, the lookup still resolves.
active_path = pg.url_path or "overview"
accent = PAGE_ACCENTS.get(active_path, _DEFAULT_ACCENT)

# Inject the accent CSS for THIS page: tinted dividers only (the title underline
# is applied inline on the strip below). Low alpha keeps dividers as a tint, not
# a bold rule. Scoped to the main content area so the sidebar is unaffected.
st.markdown(
    f"""
    <style>
      section[data-testid="stMain"] hr {{
        border-color: {_rgba(accent, 0.30)} !important;
        background-color: {_rgba(accent, 0.30)} !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Persistent full-name strip across the top of every page. The underline now
# carries the active page's accent (was a flat #ECECEC).
st.markdown(
    f"""
    <div style="display:flex; align-items:baseline; gap:0.65rem; flex-wrap:wrap;
                padding:0 0 0.55rem 0; margin-bottom:1.6rem;
                border-bottom:2px solid {accent};">
      <span style="font-weight:800; letter-spacing:2.5px; color:#1F2A37;
                   font-size:1.05rem;">TRACE</span>
      <span style="color:#6B7280; font-size:0.95rem; font-weight:400;">
        Tracking Resource Allocation for Climate Equity
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

pg.run()