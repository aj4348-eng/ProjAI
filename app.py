"""
NYC "Where Should I Live?" — Streamlit Dashboard
==================================================
Polis Technologies · Section 1

Run locally:  streamlit run app.py

Database modes:
  - Postgres (Neon): Set DATABASE_URL env var or st.secrets["DATABASE_URL"]
  - SQLite (local):  Falls back to nyc_livability.db in current directory

Required files:
  - nta_livability_scores.csv
  - nta_similarity_matrix.csv
  - nta_centroids.csv
  - nyc_livability.db  (only if DATABASE_URL not set)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os, time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CONNECTION (Postgres / Neon → SQLite fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _get_database_url():
    """Read DATABASE_URL from Streamlit secrets first, then env var."""
    try:
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    return os.environ.get("DATABASE_URL")

DATABASE_URL = _get_database_url()
USE_POSTGRES = bool(DATABASE_URL)

@st.cache_resource
def get_engine():
    """Return a SQLAlchemy engine for Postgres, or None if using SQLite."""
    if not USE_POSTGRES:
        return None
    try:
        from sqlalchemy import create_engine
        url = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1) \
            if DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL \
            else DATABASE_URL
        return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    except Exception as e:
        st.warning(f"Postgres connection failed, falling back to SQLite: {e}")
        return None

def db_query(sql):
    """Execute a SQL query against Postgres or SQLite. Returns DataFrame or empty."""
    if USE_POSTGRES:
        engine = get_engine()
        if engine is not None:
            try:
                return pd.read_sql(sql, engine)
            except Exception:
                return pd.DataFrame()
    if os.path.exists("nyc_livability.db"):
        c = sqlite3.connect("nyc_livability.db")
        try:
            return pd.read_sql(sql, c)
        except Exception:
            return pd.DataFrame()
        finally:
            c.close()
    return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NYC Livability", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="expanded")

# ── Plotly global template ──────────────────────────────────────────────────
nyc_template = go.layout.Template()
nyc_template.layout = go.Layout(
    font=dict(family="Fira Sans, sans-serif", color="#E2E8F0"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#3B82F6", "#60A5FA", "#34D399", "#FBBF24",
              "#F87171", "#A78BFA", "#F472B6", "#22D3EE"],
    title=dict(font=dict(size=16, color="#E2E8F0")),
    xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.15)"),
    margin=dict(t=48, b=32, l=48, r=16),
)
pio.templates["nyc"] = nyc_template
pio.templates.default = "plotly_dark+nyc"

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500;700&family=Playfair+Display:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #3B82F6;
    --primary-light: #60A5FA;
    --accent: #FBBF24;
    --bg: #0F172A;
    --surface: #1E293B;
    --border: rgba(255,255,255,0.1);
    --text: #E2E8F0;
    --text-muted: #94A3B8;
    --green: #34D399;
    --red: #F87171;
}

.stApp { font-family: 'Fira Sans', sans-serif; background: #0F172A !important; }
h1, h2, h3, h4 { font-family: 'Fira Sans', sans-serif !important; font-weight: 700 !important; color: var(--text) !important; }
/* Hero header bar */
header[data-testid="stHeader"] {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 45%, #1E40AF 100%) !important;
    height: auto !important;
    min-height: 5.2rem !important;
    padding: 1.6rem 2rem !important;
    display: flex !important;
    align-items: center !important;
    z-index: 999 !important;
}
header[data-testid="stHeader"] [data-testid="stToolbar"] {
    position: absolute !important;
    right: 1rem !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
}
header[data-testid="stHeader"] [data-testid="stToolbar"] button { color: #94A3B8 !important; }

/* Tighten Streamlit spacing — main content */
.block-container { padding-top: 3rem !important; padding-bottom: 0 !important; }
[data-testid="stMainBlockContainer"] { padding-top: 4.4rem !important; }
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0.75rem !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 0.75rem !important; }
h3 { margin-top: 0.2rem !important; margin-bottom: 0.5rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }
hr { margin: 0.5rem 0 !important; }
.stDivider { margin: 0.5rem 0 !important; }
.stDownloadButton { margin-top: 0.3rem !important; }
/* Sidebar stays tight */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0px !important; }
[data-testid="stSidebar"] .stElementContainer { margin: 2px 0 !important; }
/* Hide the blank " " label inside each slider */
[data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] {
    display: none !important;
}
[data-testid="stSidebar"] .stSlider {
    margin-top: 2px !important;
    margin-bottom: 4px !important;
    padding: 0 !important;
}
/* Crush inner slider padding */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    margin: 0 !important;
    padding: 4px 0 !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div {
    padding: 6px 0 !important;
}
[data-testid="stSidebar"] .stSlider > div { padding: 0 !important; }
/* Remove slider focus/click outline */
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stSlider *,
[data-testid="stSidebar"] .stSlider:focus-within {
    outline: none !important;
    box-shadow: none !important;
}
/* Hide pills label */
[data-testid="stSidebar"] [data-testid="stPills"] > [data-testid="stWidgetLabel"] {
    display: none !important;
}

/* Sidebar — fixed, uncollapsible, dark premium */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1437 0%, #111B45 40%, #0D1332 100%) !important;
    min-width: 285px !important;
    width: 285px !important;
    transform: none !important;
    transition: none !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
    top: 0 !important;
    z-index: 1000 !important;
    height: 100vh !important;
    border-right: 1px solid rgba(201,169,97,0.15) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.3) !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
    font-family: 'Outfit', sans-serif !important;
}
/* Kill ALL collapse/arrow controls */
[data-testid="stSidebar"] [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebar"] .stSidebarCollapsedControl,
[data-testid="stSidebar"] [data-testid="stSidebarNav"],
[data-testid="stSidebar"] header {
    display: none !important;
}
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stSidebar"] [data-testid="stSidebarResizeHandle"] { display: none !important; }
[data-testid="stSidebar"]::after { display: none !important; }
[data-testid="stSidebarHeader"] { display: none !important; }
[data-testid="stLogoSpacer"] { display: none !important; }
/* Smooth reruns — kill ALL flicker/dimming during reruns */
[data-testid="stStatusWidget"] { display: none !important; }
[data-stale="true"], [data-stale="true"] * { opacity: 1 !important; }
.stApp, .stApp *, [data-testid="stSidebar"], [data-testid="stSidebar"] * {
    transition-duration: 0s !important;
    transition-delay: 0s !important;
    animation-duration: 0s !important;
}
.stApp[data-test-script-state="running"] > div { opacity: 1 !important; }
/* Perf: skip rendering hidden tab panels — only active tab paints */
.stTabs [data-baseweb="tab-panel"] { content-visibility: auto; contain-intrinsic-size: auto 800px; }
/* Prevent layout shift on rerun — stable heights */
.stTabs { min-height: 700px; }
.hero-v2 { min-height: 60px; }
.metric-card, .metric-card-hero { height: 105px; }
/* Keep main content stable during script rerun */
.stApp[data-test-script-state="running"] section.main { overflow-anchor: auto !important; }
section.main .block-container { contain: layout style; }
[data-testid="stSidebar"] > div:first-child { overflow-y: auto !important; overflow-x: hidden !important; resize: none !important; padding-top: 0 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px !important;
    margin: 8px !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    contain: layout style paint !important;
}
[data-testid="stSidebar"] .stElementContainer:first-child { margin-top: 0 !important; }
/* Sidebar text — cream on dark */
[data-testid="stSidebar"] * { color: #F5F1E8 !important; }
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.82rem; font-weight: 600;
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] .stSlider label {
    font-size: 0 !important;
    line-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}
/* Unified gold slider — thumb cream, track gold gradient */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #C9A961 !important;
    border-color: #A89466 !important;
    width: 14px !important;
    height: 14px !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] ~ div {
    background: linear-gradient(90deg, #A89466, #C9A961) !important;
}
/* Hide default slider labels */
[data-testid="stSidebar"] .stSlider label p { font-size: 0 !important; line-height: 0 !important; height: 0 !important; margin: 0 !important; padding: 0 !important; overflow: hidden !important; }
/* Hide BaseWeb thumb value tooltip */
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSliderThumbValue"] { display: none !important; }
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] > div:first-child { visibility: hidden !important; width: 0 !important; height: 0 !important; overflow: hidden !important; }
/* Hide tick bar */
[data-testid="stSidebar"] [data-testid="stSliderTickBar"] { display: none !important; }
/* Drag-only tooltip */
[data-testid="stSidebar"] .stSlider:active [data-baseweb="slider"] [role="slider"]::after { content: attr(aria-valuenow)'%'; position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%); background: #8C7A4A; color: #F5F1E8; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; white-space: nowrap; pointer-events: none; }
/* Slider thumb glow — unified gold */
[data-testid="stSidebar"] .stSlider [role="slider"] { box-shadow: 0 0 10px rgba(201,169,97,0.5) !important; }
/* Dark track background — inset well */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="progressbar"] { background: rgba(0,0,0,0.28) !important; }
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 0.6rem 0 !important;
}
/* Dark scrollbar */
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar { width: 4px; }
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track { background: transparent; }
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
/* Budget input — hide spinners, add $ prefix and /mo suffix */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input[type="number"]::-webkit-inner-spin-button,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none !important; margin: 0;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input[type="number"] {
    -moz-appearance: textfield !important;
    padding-left: 20px !important;
    padding-right: 36px !important;
    background: rgba(0,0,0,0.28) !important;
    border: 0.5px solid rgba(201,169,97,0.22) !important;
    color: #F5F1E8 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input::placeholder {
    color: #6B7290 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"] {
    position: relative !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"]::before {
    content: '$';
    position: absolute; left: 8px; top: 50%; transform: translateY(-50%);
    font-size: 13px; font-weight: 500; color: #C9A961; z-index: 1; pointer-events: none;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"]::after {
    content: '/mo';
    position: absolute; right: 32px; top: 50%; transform: translateY(-50%);
    font-size: 11px; color: #6B7290; z-index: 1; pointer-events: none;
}
/* Hide Streamlit's +/- step buttons but keep clear (X) button */
[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"],
[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"] {
    display: none !important;
}
/* Hide "Press Enter to apply" tooltip */
[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-testid="InputInstructions"] {
    display: none !important;
}
/* Budget chip active state */
[class*="st-key-bchip_active"] .stButton button,
[class*="st-key-bchip_active"] .stButton button:hover,
[class*="st-key-bchip_active"] .stButton button:focus {
    background: rgba(201,169,97,0.25) !important;
    border: 1.5px solid #C9A961 !important;
    color: #F0E2B6 !important;
    box-shadow: 0 0 6px rgba(201,169,97,0.15) !important;
}

/* Weight slider labels */
.slider-label { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500; color: #F5F1E8; margin: 10px 0 2px 0; }
.slider-label .sl-dot { width: 6px; height: 6px; border-radius: 1px; flex-shrink: 0; }
.slider-label .sl-val { margin-left: auto; font-weight: 500; font-size: 12px; font-variant-numeric: tabular-nums; color: #C9A961; }
/* Section header — muted gold */
[data-testid="stSidebar"] .section-hdr {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #A89466 !important;
    margin: 14px 0 10px 0;
}
/* Budget quick-chips */
[class*="st-key-bchip"] .stButton button {
    height: 34px !important;
    min-height: 34px !important;
    max-height: 34px !important;
    border: 0.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: #A8B1C9 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 0 6px !important;
    box-shadow: none !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 34px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}
[class*="st-key-bchip"] .stButton button:hover {
    background: rgba(201,169,97,0.12) !important;
    border-color: rgba(201,169,97,0.4) !important;
    color: #E8D9B0 !important;
}
/* Logo shimmer bar */
.logo-shimmer-bar {
    height: 2px;
    margin-top: 8px;
    background: linear-gradient(90deg, transparent 0%, #C9A961 50%, transparent 100%);
    background-size: 200% 100%;
    border-radius: 1px;
    animation: logo-shimmer 3s ease-in-out infinite !important;
}
@keyframes logo-shimmer {
    0%, 100% { background-position: -200% 0; }
    50% { background-position: 200% 0; }
}
/* Logo card — Polis institutional */
.logo-card {
    background: #0B1437;
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 0;
    margin-bottom: 0;
    display: flex;
    flex-direction: column;
    gap: 0;
}
.logo-top {
    display: flex;
    align-items: center;
    gap: 10px;
}
.logo-divider {
    width: 1px;
    height: 28px;
    background: linear-gradient(180deg, transparent, #C9A961, transparent);
    flex-shrink: 0;
}
.logo-text-stack {
    display: flex;
    flex-direction: column;
    line-height: 1.15;
}
.logo-wordmark {
    font-size: 15px;
    font-weight: 500;
    letter-spacing: 0.28em;
    color: #F5F1E8 !important;
}
.logo-subtitle {
    font-size: 9px;
    font-weight: 400;
    letter-spacing: 0.22em;
    margin-top: 4px;
    color: #C9A961 !important;
    opacity: 0.85;
}
[data-testid="stSidebar"] .logo-card,
[data-testid="stSidebar"] .logo-card * {
    color: #fff !important;
}
[data-testid="stSidebar"] .stCheckbox label span { color: #F5F1E8 !important; }
/* Force boro pills full width (multi-select renders ~6px narrower) */
[data-testid="stSidebar"] .st-key-boro_pills {
    width: 100% !important;
    min-width: 100% !important;
    box-sizing: border-box !important;
}
[data-testid="stSidebar"] .st-key-boro_pills > div {
    width: 100% !important;
    min-width: 100% !important;
}
[data-testid="stSidebar"] .st-key-boro_pills [data-baseweb="button-group"] {
    width: 100% !important;
    min-width: 100% !important;
}
/* ── Pill groups shared: occupation + borough ── */
[data-testid="stSidebar"] .st-key-preset_pills [data-baseweb="button-group"],
[data-testid="stSidebar"] .st-key-boro_pills [data-baseweb="button-group"] {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 4px !important;
    background: rgba(0,0,0,0.28) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
/* Inactive buttons */
[data-testid="stSidebar"] .st-key-preset_pills button[data-testid="stBaseButton-pills"],
[data-testid="stSidebar"] .st-key-boro_pills button[data-testid="stBaseButton-pills"] {
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: #A8B1C9 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    height: 34px !important;
    min-height: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-shadow: none !important;
    padding: 6px 8px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
/* Prevent text truncation in pill buttons */
[data-testid="stSidebar"] .st-key-preset_pills button *,
[data-testid="stSidebar"] .st-key-boro_pills button * {
    overflow: visible !important;
    text-overflow: clip !important;
}
/* Hover */
[data-testid="stSidebar"] .st-key-preset_pills button[data-testid="stBaseButton-pills"]:hover,
[data-testid="stSidebar"] .st-key-boro_pills button[data-testid="stBaseButton-pills"]:hover {
    background: rgba(201,169,97,0.12) !important;
    border-color: rgba(201,169,97,0.4) !important;
    color: #F5F1E8 !important;
    box-shadow: none !important;
}
/* Last child span full width */
[data-testid="stSidebar"] .st-key-preset_pills button[data-testid="stBaseButton-pills"]:last-child,
[data-testid="stSidebar"] .st-key-preset_pills button[data-testid="stBaseButton-pillsActive"]:last-child,
[data-testid="stSidebar"] .st-key-boro_pills button[data-testid="stBaseButton-pills"]:last-child,
[data-testid="stSidebar"] .st-key-boro_pills button[data-testid="stBaseButton-pillsActive"]:last-child {
    grid-column: span 2 !important;
}
/* Active buttons */
[data-testid="stSidebar"] .st-key-preset_pills button[data-testid="stBaseButton-pillsActive"],
[data-testid="stSidebar"] .st-key-boro_pills button[data-testid="stBaseButton-pillsActive"] {
    border: 1px solid rgba(201,169,97,0.5) !important;
    border-radius: 8px !important;
    background: rgba(201,169,97,0.22) !important;
    box-shadow: 0 0 12px rgba(201,169,97,0.18), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    color: #C9A961 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    height: 34px !important;
    min-height: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    padding: 6px 8px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
/* Sidebar columns gap */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap: 4px !important;
    align-items: center !important;
}

/* Tab strip */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.06);
    padding: 4px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 500;
    font-size: 0.85rem;
    color: var(--text-muted);
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: #FFFFFF !important;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(30,64,175,0.25);
}

/* Hero — lives inside the fixed header bar */
.hero-v2 {
    background: transparent;
    color: #fff;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    min-height: 5.2rem;
    padding: 0 2rem;
    pointer-events: none;
}
.hero-v2::before { display: none; }
.hero-v2 h1 { color: #fff !important; font-size: 1.55rem; margin: 0 !important; position: relative; white-space: nowrap; font-weight: 700 !important; }
.hero-v2 p { color: #94A3B8; margin: 0; font-size: 0.8rem; position: relative; white-space: nowrap; }
.hero-v2 .accent { color: #F59E0B; font-weight: 600; }

/* Stat cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 0.3rem 0;
    display: flex; flex-direction: column; justify-content: center;
}
.metric-card::before { display: none; }
.metric-card-hero {
    background: var(--surface);
    border: 2px solid var(--accent);
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 0.3rem 0;
    box-shadow: 0 2px 12px rgba(245,158,11,0.10);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    cursor: default;
}
.metric-card-hero:hover {
    box-shadow: 0 6px 24px rgba(245,158,11,0.18);
    transform: translateY(-1px);
}
.metric-card-hero .metric-val { font-size: 1.25rem; }
.metric-card-hero { animation: fadeInUp 0.4s ease both; animation-delay: 0.05s; }
.metric-val {
    font-family: 'Fira Code', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
}
.metric-lbl {
    color: var(--text-muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* Pills */
.pill { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 0.73rem; font-weight: 600; }
.pill-green { background: rgba(16,185,129,0.18); color: #6EE7B7; }
.pill-blue { background: rgba(59,130,246,0.18); color: #93C5FD; }
.pill-amber { background: rgba(245,158,11,0.18); color: #FCD34D; }
.pill-red { background: rgba(239,68,68,0.18); color: #FCA5A5; }

/* Footer — matches header gradient */
.footer-v2 {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 45%, #1E40AF 100%);
    color: #CBD5E1;
    padding: 0.8rem 2rem;
    border-radius: 12px;
    margin-top: 0.5rem;
    font-size: 0.78rem;
    text-align: center;
    letter-spacing: 0.01em;
}
.footer-v2 span { color: #F59E0B; font-weight: 600; }


/* Metric card hover */
.metric-card {
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    cursor: default;
}
.metric-card:hover {
    box-shadow: 0 6px 20px rgba(30,64,175,0.12);
    transform: translateY(-1px);
}

/* Ranking card */
.rank-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.rank-card:hover {
    box-shadow: 0 4px 16px rgba(30,64,175,0.08);
    border-color: var(--primary-light);
}
.rank-num {
    font-family: 'Fira Code', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--primary);
    min-width: 2.5rem;
    text-align: center;
}
.rank-info { flex: 1; min-width: 0; }
.rank-name {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text);
    margin: 0;
}
.rank-borough {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0;
}
.rank-scores {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.35rem;
}
.rank-dim {
    font-size: 0.68rem;
    color: var(--text-muted);
    background: rgba(255,255,255,0.08);
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'Fira Code', monospace;
    white-space: nowrap;
}
.rank-right {
    text-align: right;
    min-width: 5rem;
    flex-shrink: 0;
}
.rank-score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 8px;
    font-family: 'Fira Code', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    color: #fff;
}
.badge-green { background: linear-gradient(135deg, #10B981, #059669); }
.badge-amber { background: linear-gradient(135deg, #F59E0B, #D97706); }
.badge-red { background: linear-gradient(135deg, #EF4444, #DC2626); }
.rank-rent {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
}
.rank-bar {
    width: 100%;
    height: 4px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    margin-top: 6px;
    overflow: hidden;
}
.rank-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--primary), var(--green));
    transition: width 0.6s ease;
}

/* Gold #1 card */
.rank-card-gold {
    background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(217,119,6,0.18) 100%);
    border: 2px solid rgba(245,158,11,0.5);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(245,158,11,0.15);
    transition: box-shadow 0.2s ease;
}
.rank-card-gold:hover { box-shadow: 0 8px 32px rgba(245,158,11,0.22); }
.rank-card-gold .rank-num { color: #FCD34D; font-size: 1.6rem; }
.rank-card-gold .rank-name { font-size: 1.05rem; }
.rank-card-gold .rank-explain {
    font-size: 0.78rem;
    color: #FCD34D;
    margin: 4px 0 0;
    font-style: italic;
}

/* Compact row for #6+ */
.rank-card-compact {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: background 0.15s ease;
}
.rank-card-compact:hover { background: rgba(255,255,255,0.06); }
.rank-card-compact .rank-num { font-size: 0.85rem; min-width: 2rem; }
.rank-card-compact .rank-name { font-size: 0.85rem; }
.rank-card-compact .rank-borough { font-size: 0.7rem; }
.rank-card-compact .rank-scores { display: none; }
.rank-card-compact .rank-bar { display: none; }

/* Smart tag */
.smart-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 0.68rem;
    font-weight: 600;
    margin-left: 6px;
}
.tag-safety { background: rgba(16,185,129,0.15); color: #6EE7B7; }
.tag-quiet { background: rgba(99,102,241,0.15); color: #A5B4FC; }
.tag-parks { background: rgba(34,197,94,0.15); color: #86EFAC; }
.tag-transit { background: rgba(59,130,246,0.15); color: #93C5FD; }
.tag-value { background: rgba(245,158,11,0.15); color: #FCD34D; }

/* Score context label */
.score-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* Sub-score bar row */
.subscore-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 3px;
}
.subscore-lbl {
    font-size: 0.68rem;
    color: var(--text-muted);
    min-width: 3.2rem;
    text-align: right;
}
.subscore-bar {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    overflow: hidden;
    max-width: 120px;
}
.subscore-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}
.subscore-val {
    font-size: 0.65rem;
    color: var(--text-muted);
    min-width: 2rem;
    font-family: 'Fira Code', monospace;
}

/* Info tooltip icon */
.info-tip {
    font-size: 0.85rem;
    color: #94A3B8;
    cursor: help;
    vertical-align: middle;
    margin-left: 6px;
    position: relative;
}
.info-tip:hover { color: var(--primary); }

/* Ranking row hover (fallback for st.columns rows) */
[data-testid="stHorizontalBlock"]:has(.pill) {
    transition: background 0.15s ease;
    border-radius: 8px;
    padding: 2px 4px;
    cursor: pointer;
}
[data-testid="stHorizontalBlock"]:has(.pill):hover {
    background: rgba(59,130,246,0.12);
}

/* Progress bar styling */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--primary), var(--primary-light), var(--green)) !important;
    border-radius: 4px;
}
[data-testid="stProgress"] {
    height: 6px !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

/* Download button */
.stDownloadButton button {
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: background 0.2s ease !important;
    cursor: pointer;
}
.stDownloadButton button:hover {
    background: var(--primary-light) !important;
}

/* Text input styling */
.stTextInput input {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus {
    border-color: var(--primary-light) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

/* Selectbox styling */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    border-radius: 8px;
}

/* Entrance animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.metric-card { animation: fadeInUp 0.4s ease both; }
.metric-card:nth-child(1) { animation-delay: 0s; }
.metric-card:nth-child(2) { animation-delay: 0.05s; }
.metric-card:nth-child(3) { animation-delay: 0.1s; }
.metric-card:nth-child(4) { animation-delay: 0.15s; }
.rank-card { animation: fadeInUp 0.35s ease both; }

/* Tab cursor */
.stTabs [data-baseweb="tab"] { cursor: pointer; }

/* Focus-visible states */
.stTabs [data-baseweb="tab"]:focus-visible {
    outline: 2px solid var(--primary-light);
    outline-offset: 2px;
    border-radius: 8px;
}
[data-testid="stSidebar"] .stSlider:focus-within {
    outline: 2px solid var(--primary-light);
    outline-offset: 2px;
    border-radius: 6px;
}
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible,
[data-testid="stSelectbox"]:focus-within {
    outline: 2px solid var(--primary-light) !important;
    outline-offset: 2px;
}

/* Tab strip mobile scroll */
.stTabs [data-baseweb="tab-list"] {
    overflow-x: auto;
    flex-wrap: nowrap;
    scrollbar-width: thin;
    -webkit-overflow-scrolling: touch;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 3px;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.2);
    border-radius: 3px;
}

/* Tab hover */
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(59,130,246,0.12);
    color: var(--primary) !important;
}

/* Responsive — 1440px+ (large desktops) */
@media (min-width: 1440px) {
    .metric-card { padding: 1.4rem 1.8rem; }
    .metric-val { font-size: 1.7rem; }
}

/* Responsive — tablet */
@media (max-width: 1024px) {
    .hero-v2 { gap: 0.8rem; padding: 1.2rem 1.5rem; }
    .hero-v2 h1 { font-size: 1.3rem; }
    .hero-v2 p { font-size: 0.75rem; }
}

/* Responsive — small tablet / landscape phone */
@media (max-width: 768px) {
    .metric-card { padding: 1rem 1rem; }
    .metric-val { font-size: 1.15rem; }
    .metric-lbl { font-size: 0.65rem; }
    .hero-v2 { padding: 0.8rem 1rem; flex-wrap: wrap; justify-content: center; gap: 0.5rem; }
    .hero-v2 h1 { font-size: 1.1rem; }
    .hero-v2 p { font-size: 0.65rem; }
    .footer-v2 { font-size: 0.7rem; padding: 0.6rem 1rem; }
}

/* Responsive — phone */
@media (max-width: 375px) {
    .metric-card { padding: 0.7rem 0.8rem; }
    .metric-val { font-size: 1rem; }
    .metric-lbl { font-size: 0.6rem; }
    .hero-v2 h1 { font-size: 0.95rem; }
    .hero-v2 p { display: none; }
    .stTabs [data-baseweb="tab"] { padding: 6px 10px; font-size: 0.75rem; }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

/* ── Anti-flash: smooth rerun transitions ──────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"] {
    transition: none !important;
    background-color: #0F172A !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #1E293B !important;
}
.main .block-container {
    animation: _rerunFade 0.25s ease-out both;
}
@keyframes _rerunFade {
    from { opacity: 0.88; }
    to   { opacity: 1; }
}
[data-testid="stSidebar"] [data-testid="stSlider"],
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] [data-testid="stPills"] {
    min-height: 0;
}
iframe[height="0"] { display: none !important; }

/* ── Cluster cards ────────────────────────────────────────────────── */
.cluster-hero {
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(168,85,247,0.15));
    border: 2px solid rgba(56,189,248,0.5);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 0 30px rgba(56,189,248,0.12), 0 8px 32px rgba(0,0,0,0.3);
    text-align: center;
}
.cluster-hero .hero-emoji { font-size: 3rem; margin-bottom: 0.3rem; }
.cluster-hero .hero-title { font-size: 1.5rem; font-weight: 700; color: #F1F5F9; margin: 0.2rem 0; }
.cluster-hero .hero-match { font-size: 2rem; font-weight: 800; background: linear-gradient(90deg,#38BDF8,#A855F7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.cluster-hero .hero-desc { color: #94A3B8; font-size: 0.92rem; margin-top: 0.4rem; }

.cluster-card {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(71,85,105,0.4);
    border-radius: 14px;
    padding: 1rem 0.9rem;
    transition: transform 0.2s, box-shadow 0.2s;
    text-align: center;
    height: 100%;
}
.cluster-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.cluster-card .card-emoji { font-size: 2rem; }
.cluster-card .card-title { font-size: 0.95rem; font-weight: 600; color: #E2E8F0; margin: 0.3rem 0 0.2rem; }
.cluster-card .card-count { font-size: 0.78rem; color: #64748B; margin-bottom: 0.5rem; }

.match-badge {
    display: inline-block;
    background: linear-gradient(90deg,#38BDF8,#818CF8);
    color: #FFF;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 10px;
    margin-bottom: 0.3rem;
    letter-spacing: 0.5px;
}

.cc-bar-wrap {
    display: flex; align-items: center; gap: 4px;
    margin: 3px 0; font-size: 0.7rem; color: #94A3B8;
}
.cc-bar-label { width: 48px; text-align: right; flex-shrink: 0; }
.cc-bar-track {
    flex: 1; height: 6px; border-radius: 3px;
    background: rgba(71,85,105,0.3); overflow: hidden;
}
.cc-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.4s ease;
}
.cc-bar-val { width: 30px; text-align: left; flex-shrink: 0; }

/* ── COMMUTE TAB ── */
.commute-placeholder {
    text-align: center; padding: 3rem 2rem; border-radius: 16px;
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(168,85,247,0.08));
    border: 1.5px dashed rgba(148,163,184,0.25);
}
.commute-placeholder .cp-icon { font-size: 2.8rem; margin-bottom: 0.6rem; }
.commute-placeholder .cp-title { font-size: 1.15rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.3rem; }
.commute-placeholder .cp-sub { font-size: 0.85rem; color: #94A3B8; }

.commute-row {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 0.8rem 1rem; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.8rem;
    transition: box-shadow 0.2s, border-color 0.2s; animation: fadeInUp 0.35s ease both;
}
.commute-row:hover { box-shadow: 0 4px 16px rgba(30,64,175,0.08); border-color: var(--primary-light); }
.commute-row .cr-rank, .commute-row-gold .cr-rank {
    font-family: 'Fira Code', monospace; font-size: 1.1rem; font-weight: 700;
    color: var(--primary); min-width: 28px; text-align: center;
}
.commute-row .cr-info, .commute-row-gold .cr-info { flex: 1; min-width: 0; }
.commute-row .cr-name, .commute-row-gold .cr-name { font-size: 0.9rem; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.commute-row .cr-boro, .commute-row-gold .cr-boro { font-size: 0.72rem; color: var(--text-muted); }
.commute-row .cr-bar, .commute-row-gold .cr-bar { height: 5px; border-radius: 3px; background: rgba(71,85,105,0.3); margin-top: 4px; overflow: hidden; }
.commute-row .cr-bar-fill, .commute-row-gold .cr-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--primary), var(--green)); }
.commute-row .cr-right, .commute-row-gold .cr-right { display: flex; flex-direction: column; align-items: flex-end; gap: 3px; }
.commute-row .cr-score, .commute-row-gold .cr-score { font-family: 'Fira Code', monospace; font-size: 0.82rem; font-weight: 600; color: var(--text); }
.commute-row .cr-rent, .commute-row-gold .cr-rent { font-size: 0.72rem; color: var(--text-muted); }

.commute-badge {
    display: inline-block; padding: 3px 10px; border-radius: 99px;
    font-size: 0.75rem; font-weight: 700; font-family: 'Fira Code', monospace;
}
.cb-green { background: rgba(16,185,129,0.18); color: #6EE7B7; }
.cb-amber { background: rgba(245,158,11,0.18); color: #FCD34D; }
.cb-red   { background: rgba(239,68,68,0.18); color: #FCA5A5; }

.commute-row-gold {
    background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(217,119,6,0.18));
    border: 2px solid rgba(245,158,11,0.5); border-radius: 12px;
    padding: 0.8rem 1rem; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.8rem;
    box-shadow: 0 4px 20px rgba(245,158,11,0.15); animation: fadeInUp 0.35s ease both;
}
.commute-row-gold:hover { box-shadow: 0 8px 32px rgba(245,158,11,0.22); }
.commute-row-gold .cr-rank { color: #FCD34D; font-size: 1.1rem; }
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_master():
    if not os.path.exists("nta_livability_scores.csv"):
        st.error("Missing nta_livability_scores.csv — run the notebook first.")
        st.stop()
    return pd.read_csv("nta_livability_scores.csv")

@st.cache_data(ttl=3600)
def load_listings():
    return db_query("SELECT * FROM rent_listings")

@st.cache_data(ttl=3600)
def load_redfin():
    return db_query("SELECT * FROM redfin_listings")

@st.cache_data(ttl=3600)
def load_311():
    return db_query("SELECT created_date,complaint_type,nta_name,borough FROM complaints_311 WHERE created_date IS NOT NULL")

@st.cache_data
def load_sim():
    if os.path.exists("nta_similarity_matrix.csv"):
        return pd.read_csv("nta_similarity_matrix.csv", index_col=0)
    return None

@st.cache_data(ttl=86400)
def load_geojson():
    import requests
    for url in ["https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?$limit=500",
                "https://raw.githubusercontent.com/nycehs/NYC_geography/master/NTA.geo.json"]:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue
    return None

df = load_master()
listings = load_listings()
redfin = load_redfin()
ts_data = load_311()
sim_mx = load_sim()
geo = load_geojson()

RC = next((c for c in ['redfin_median_rent', 'census_median_rent', 'predicted_rent'] if c in df.columns and df[c].notna().sum() > 20), None)
TC = 'transit_plus_score' if 'transit_plus_score' in df.columns else 'transit_score'

SLIDER_COLORS = {"Safety": "#10B981", "Quiet": "#6366F1", "Parks": "#22C55E", "Transit": "#3B82F6"}

def render_logo_card(match_count=0, avg_rent=None):
    st.markdown('''<div class="logo-card">
<div class="logo-top">
<svg viewBox="0 0 64 64" width="32" height="32" aria-label="Polis">
  <rect x="1.5" y="1.5" width="61" height="61" fill="none" stroke="#C9A961" stroke-width="0.75"/>
  <rect x="4" y="4" width="56" height="56" fill="none" stroke="#C9A961" stroke-width="0.4" opacity="0.5"/>
  <text x="32" y="48" text-anchor="middle" font-family="'Playfair Display', Georgia, serif" font-size="44" font-weight="400" fill="#C9A961">P</text>
</svg>
<div class="logo-divider"></div>
<div class="logo-text-stack">
<div class="logo-wordmark">POLIS</div>
<div class="logo-subtitle">TECHNOLOGIES</div>
</div>
</div>
<div class="logo-shimmer-bar"></div>
</div>''', unsafe_allow_html=True)

PRESETS = {
    "Professional": (25, 10, 10, 55),
    "Family": (40, 30, 25, 5),
    "Student": (15, 10, 5, 70),
    "Commuter": (20, 10, 10, 60),
    "Custom":       (25, 25, 25, 25),
}

if 'preset' not in st.session_state:
    st.session_state.preset = 'Custom'

def _reset_weights():
    st.session_state.preset = "Custom"
    st.session_state.preset_pills = "Custom"
    st.session_state.sw = 25
    st.session_state.nw = 25
    st.session_state.pw = 25
    st.session_state.tw = 25
    st.session_state.budget_input = None

def _set_budget(v):
    st.session_state.budget_input = v if v else None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_slot = st.empty()
    sv = st.session_state.get
    active_preset = st.session_state.preset or "Custom"
    d = PRESETS.get(active_preset, (25, 25, 25, 25))
    _sw = sv("sw", d[0]); _nw = sv("nw", d[1]); _pw = sv("pw", d[2]); _tw = sv("tw", d[3])
    _total = max(_sw + _nw + _pw + _tw, 1)

    # OCCUPATION
    st.markdown('<div class="section-hdr" style="margin-top:18px">OCCUPATION</div>', unsafe_allow_html=True)
    persona = st.pills("Occupation", list(PRESETS.keys()), default="Custom", key="preset_pills", label_visibility="collapsed")
    if persona and persona != st.session_state.preset:
        st.session_state.preset = persona
        d = PRESETS[persona]
        st.session_state.sw = d[0]; st.session_state.nw = d[1]
        st.session_state.pw = d[2]; st.session_state.tw = d[3]

    # Sliders with color dots
    st.markdown(f'<div class="slider-label"><span class="sl-dot" style="background:#10B981"></span>Safety <span class="sl-val" style="color:#10B981">{sv("sw", d[0])}%</span></div>', unsafe_allow_html=True)
    sw = st.slider(" ", 0, 100, d[0], 5, key="sw")
    st.markdown(f'<div class="slider-label"><span class="sl-dot" style="background:#6366F1"></span>Quiet <span class="sl-val" style="color:#6366F1">{sv("nw", d[1])}%</span></div>', unsafe_allow_html=True)
    nw = st.slider(" ", 0, 100, d[1], 5, key="nw")
    st.markdown(f'<div class="slider-label"><span class="sl-dot" style="background:#22C55E"></span>Parks <span class="sl-val" style="color:#22C55E">{sv("pw", d[2])}%</span></div>', unsafe_allow_html=True)
    pw = st.slider(" ", 0, 100, d[2], 5, key="pw")
    st.markdown(f'<div class="slider-label"><span class="sl-dot" style="background:#3B82F6"></span>Transit <span class="sl-val" style="color:#3B82F6">{sv("tw", d[3])}%</span></div>', unsafe_allow_html=True)
    tw = st.slider(" ", 0, 100, d[3], 5, key="tw")

    # BUDGET
    st.markdown('<div class="section-hdr" style="margin-top:18px">BUDGET</div>', unsafe_allow_html=True)
    budget_val = st.number_input(
        "Budget", min_value=0, max_value=20000, value=None, step=100,
        key="budget_input", label_visibility="collapsed",
        placeholder="No limit",
    )
    budget = budget_val if budget_val is not None else None
    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
    with bc1:
        st.button("$2k", key=f"bchip_{'active_' if budget_val == 2000 else ''}2k", type="tertiary", on_click=_set_budget, args=(2000,))
    with bc2:
        st.button("$3k", key=f"bchip_{'active_' if budget_val == 3000 else ''}3k", type="tertiary", on_click=_set_budget, args=(3000,))
    with bc3:
        st.button("$4k", key=f"bchip_{'active_' if budget_val == 4000 else ''}4k", type="tertiary", on_click=_set_budget, args=(4000,))
    with bc4:
        st.button("$5k", key=f"bchip_{'active_' if budget_val == 5000 else ''}5k", type="tertiary", on_click=_set_budget, args=(5000,))
    with bc5:
        st.button("∞", key=f"bchip_{'active_' if budget_val is None else ''}inf", type="tertiary", on_click=_set_budget, args=(0,))

    # BOROUGHS
    all_boros = sorted(df['borough'].dropna().unique().tolist())
    boro_labels = list(all_boros)
    boro_map = dict(zip(boro_labels, all_boros))
    boro_count = len(st.session_state.get("boro_pills", []))
    st.markdown(f'<div class="section-hdr" style="margin-top:18px">BOROUGHS <span style="float:right;font-variant-numeric:tabular-nums">{boro_count} / {len(all_boros)}</span></div>', unsafe_allow_html=True)
    sb_raw = st.pills("Borough", boro_labels, selection_mode="multi", default=[], key="boro_pills", label_visibility="collapsed")
    sb = [boro_map.get(b, b) for b in (sb_raw or [])]

    # DB status indicator
    db_color = "#34D399" if USE_POSTGRES else "#FBBF24"
    db_label = "Postgres" if USE_POSTGRES else "SQLite"
    st.markdown(
        f'<div style="margin-top:24px;padding:8px 12px;background:rgba(255,255,255,0.04);'
        f'border-radius:8px;border:1px solid rgba(255,255,255,0.08);font-size:0.7rem;'
        f'color:#94A3B8;display:flex;align-items:center;gap:8px;">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{db_color};"></span>'
        f'<span>Database: <strong style="color:{db_color}">{db_label}</strong></span></div>',
        unsafe_allow_html=True
    )

    topn = 10

# Compute user score
W = max(sw + nw + pw + tw, 1)
df['user_score'] = (sw / W * df['safety_score'] + nw / W * df['noise_score'] +
                    pw / W * df['parks_score'] + tw / W * df[TC])

ft = df.copy()
budget_empty = False
if budget and RC:
    ft = ft[ft[RC].fillna(99999) <= budget]
    if len(ft) == 0:
        budget_empty = True
if sb:
    ft = ft[ft['borough'].isin(sb)]
ft = ft.sort_values('user_score', ascending=False)
top = ft.head(topn)

top_boroughs = top['borough'].tolist() if len(top) > 0 else []
with logo_slot.container():
    render_logo_card(match_count=len(ft), avg_rent=ft[RC].median() if RC and len(ft) > 0 else None)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<div class="hero-v2">
<h1>NYC — Where Should I Live?</h1>
<p>Find the right neighborhood for how you actually live.</p>
</div>""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1.5, 1])
def mcard(n, l, hero=False):
    cls = "metric-card-hero" if hero else "metric-card"
    return f'<div class="{cls}"><div class="metric-val">{n}</div><div class="metric-lbl">{l}</div></div>'
with c1:
    st.markdown(mcard(len(df), "Neighborhoods"), unsafe_allow_html=True)
with c2:
    nm = top.iloc[0]['nta_name'].split('-')[0] if len(top) > 0 else "—"
    st.markdown(mcard(f'<span style="font-size:1.05rem">{nm}</span>', "#1 Pick", hero=True), unsafe_allow_html=True)
with c3:
    rv = f"${top.iloc[0][RC]:,.0f}" if len(top) > 0 and RC and pd.notna(top.iloc[0].get(RC)) else "—"
    st.markdown(mcard(rv, "Est. Rent"), unsafe_allow_html=True)

st.markdown('<div style="margin-bottom:1.5rem"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Rankings", "Map", "Commute", "Similar",
                "Clusters", "Listings", "Trends", "Demographics", "Data"])

# ═══════ TAB 1: RANKINGS ════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('### Top Neighborhoods <span class="info-tip" title="Ranked by your weighted priorities from the sidebar.">&#9432;</span>', unsafe_allow_html=True)
    if len(top) == 0:
        if budget_empty:
            st.warning(f"No neighborhoods under ${budget:,}/mo. Try raising your budget or loosening filters.")
        else:
            st.warning("No results. Adjust filters.")
    else:
        score_map = {'safety_score': ('Safe', 'tag-safety', '#10B981'),
                     'noise_score': ('Quiet', 'tag-quiet', '#6366F1'),
                     'parks_score': ('Parks', 'tag-parks', '#22C55E'),
                     TC: ('Transit', 'tag-transit', '#3B82F6')}
        weight_names = {'safety_score': 'Safety', 'noise_score': 'Quiet',
                        'parks_score': 'Parks', TC: 'Transit'}
        weights = {'safety_score': sw, 'noise_score': nw, 'parks_score': pw, TC: tw}
        top_weight_key = max(weights, key=weights.get)

        for rk, (_, r) in enumerate(top.iterrows(), 1):
            s = r['user_score']
            bcss = "badge-green" if s > .70 else ("badge-amber" if s > .50 else "badge-red")
            score_lbl = "Excellent match" if s > .70 else ("Strong match" if s > .60 else ("Good match" if s > .50 else "Fair match"))

            best_key = max(score_map, key=lambda k: r.get(k, 0) if pd.notna(r.get(k)) else 0)
            best_lbl, best_cls, _ = score_map[best_key]
            tag_labels = {'Safe': 'Safest', 'Quiet': 'Most Quiet', 'Parks': 'Most Parks', 'Transit': 'Top Transit'}
            smart_tag = f'<span class="smart-tag {best_cls}">{tag_labels.get(best_lbl, best_lbl)}</span>'

            bars_html = ""
            for sk, (sl, _, sc) in score_map.items():
                v = r.get(sk, 0) if pd.notna(r.get(sk)) else 0
                bw = min(v * 100, 100)
                bars_html += f'''<div class="subscore-row">
                    <span class="subscore-lbl">{sl}</span>
                    <div class="subscore-bar"><div class="subscore-bar-fill" style="width:{bw}%;background:{sc}"></div></div>
                    <span class="subscore-val">{v:.2f}</span>
                </div>'''

            rent_html = ""
            if RC and pd.notna(r.get(RC)):
                rent_html = f'<div class="rank-rent">~${r[RC]:,.0f}/mo</div>'
            bar_w = min(s * 100, 100)

            if rk == 1:
                top_prio = weight_names.get(top_weight_key, 'your priorities')
                explain = f'Top pick for {top_prio.lower()} — your highest-weighted factor.'
                st.markdown(f'''<div class="rank-card-gold" style="animation-delay:0.04s">
  <div class="rank-num">#{rk}</div>
  <div class="rank-info">
    <p class="rank-name">{r["nta_name"]}{smart_tag}</p>
    <p class="rank-borough">{r["borough"]}</p>
    <p class="rank-explain">{explain}</p>
    {bars_html}
    <div class="rank-bar"><div class="rank-bar-fill" style="width:{bar_w}%"></div></div>
  </div>
  <div class="rank-right">
    <span class="rank-score-badge {bcss}">{s:.3f}</span>
    <div class="score-label">{score_lbl}</div>
    {rent_html}
  </div>
</div>''', unsafe_allow_html=True)
            elif rk <= 5:
                st.markdown(f'''<div class="rank-card" style="animation-delay:{rk * 0.04}s">
  <div class="rank-num">#{rk}</div>
  <div class="rank-info">
    <p class="rank-name">{r["nta_name"]}{smart_tag}</p>
    <p class="rank-borough">{r["borough"]}</p>
    {bars_html}
    <div class="rank-bar"><div class="rank-bar-fill" style="width:{bar_w}%"></div></div>
  </div>
  <div class="rank-right">
    <span class="rank-score-badge {bcss}">{s:.3f}</span>
    <div class="score-label">{score_lbl}</div>
    {rent_html}
  </div>
</div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="rank-card-compact" style="animation-delay:{rk * 0.04}s">
  <div class="rank-num">#{rk}</div>
  <div class="rank-info">
    <p class="rank-name">{r["nta_name"]}{smart_tag}</p>
    <p class="rank-borough">{r["borough"]}</p>
  </div>
  <div class="rank-right">
    <span class="rank-score-badge {bcss}">{s:.3f}</span>
    <div class="score-label">{score_lbl}</div>
    {rent_html}
  </div>
</div>''', unsafe_allow_html=True)

# ═══════ TAB 2: MAP ═════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Livability Map")
    st.caption("Color = your weighted score. Borough/budget filters grey out non-matching areas.")
    if geo and 'nta_code' in df.columns:
        with st.spinner("Rendering map..."):
            props = geo['features'][0]['properties']
            gk = next((k for k in props if 'nta' in k.lower() and ('code' in k.lower() or '2020' in k.lower())), None)
            if gk:
                ft_codes = set(ft['nta_code'].values)
                sm = df.set_index('nta_code')['user_score'].to_dict()
                nm = df.set_index('nta_code')['nta_name'].to_dict()
                bm = df.set_index('nta_code')['borough'].to_dict()
                gd = pd.DataFrame([{'nta_code': f['properties'].get(gk, ''),
                                    'name': nm.get(f['properties'].get(gk, ''), ''),
                                    'borough': bm.get(f['properties'].get(gk, ''), ''),
                                    'score': sm.get(f['properties'].get(gk, ''), 0)
                                            if f['properties'].get(gk, '') in ft_codes else None}
                                   for f in geo['features']])
                valid_scores = gd['score'].dropna()
                sc_min = valid_scores.min() if len(valid_scores) > 0 else 0
                sc_max = valid_scores.max() if len(valid_scores) > 0 else 1
                if sc_max - sc_min < 0.01:
                    sc_min, sc_max = max(0, sc_min - 0.1), min(1, sc_max + 0.1)
                fig = px.choropleth_mapbox(gd, geojson=geo, locations='nta_code',
                                           featureidkey=f'properties.{gk}', color='score',
                                           color_continuous_scale=[[0,'#EF4444'],[0.25,'#F97316'],[0.5,'#EAB308'],[0.75,'#84CC16'],[1,'#10B981']],
                                           range_color=[sc_min, sc_max],
                                           mapbox_style='carto-darkmatter', zoom=10,
                                           center={"lat": 40.7128, "lon": -73.95}, opacity=.75,
                                           hover_name='name',
                                           hover_data={'score': ':.3f', 'borough': True, 'nta_code': False})
                if 'centroid_lat' in top.columns and 'centroid_lon' in top.columns:
                    top_map = top.head(10).copy()
                    top_map['rank'] = range(1, len(top_map) + 1)
                    top_map['label'] = top_map.apply(lambda r: f"#{r['rank']} {r['nta_name']}", axis=1)
                    fig.add_trace(go.Scattermapbox(
                        lat=top_map['centroid_lat'], lon=top_map['centroid_lon'],
                        mode='markers+text', text=top_map['rank'].astype(str),
                        textposition='middle center',
                        marker=dict(size=22, color='#3B82F6', opacity=0.9),
                        textfont=dict(size=10, color='white', family='Arial Black'),
                        hovertext=top_map['label'], hoverinfo='text',
                        name='Top 10'))
                fig.update_layout(margin=dict(r=0, t=0, l=0, b=0), height=620, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not match GeoJSON keys.")
    else:
        st.warning("Map data unavailable.")

# ═══════ TAB 3: COMMUTE ═════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Commute Estimator")

    addr = st.text_input("Work / school address", placeholder="e.g. 1 World Trade Center, New York",
                         key="commute_addr")

    if not addr or 'centroid_lat' not in df.columns:
        st.markdown("""<div class="commute-placeholder">
            <div class="cp-icon">🚗</div>
            <div class="cp-title">Where do you commute?</div>
            <div class="cp-sub">Type your work or school address above to see drive-time estimates<br>
            for your top-scored neighborhoods.</div>
        </div>""", unsafe_allow_html=True)
    else:
        import requests as req
        with st.spinner("Geocoding..."):
            try:
                g = req.get("https://nominatim.openstreetmap.org/search",
                            params={"q": addr, "format": "json", "limit": 1},
                            headers={"User-Agent": "NYCLivability/1.0"}, timeout=10).json()
                if not g:
                    st.error("Address not found — try adding 'New York' to the end.")
                else:
                    dlat, dlon = float(g[0]['lat']), float(g[0]['lon'])
                    found_name = g[0].get('display_name', '')[:80]
                    st.caption(f"📍 {found_name}")

                    _cols = ['nta_name', 'borough', 'user_score', 'centroid_lat', 'centroid_lon']
                    if RC:
                        _cols.append(RC)
                    cdf = ft[_cols].dropna().copy()
                    R = 6371
                    cdf['dist'] = cdf.apply(lambda r: R * 2 * np.arctan2(
                        np.sqrt(np.sin(np.radians(dlat - r['centroid_lat']) / 2) ** 2 +
                                np.cos(np.radians(r['centroid_lat'])) * np.cos(np.radians(dlat)) *
                                np.sin(np.radians(dlon - r['centroid_lon']) / 2) ** 2),
                        np.sqrt(1 - (np.sin(np.radians(dlat - r['centroid_lat']) / 2) ** 2 +
                                     np.cos(np.radians(r['centroid_lat'])) * np.cos(np.radians(dlat)) *
                                     np.sin(np.radians(dlon - r['centroid_lon']) / 2) ** 2))), axis=1)
                    cdf['est_min'] = (cdf['dist'] / 25 * 60).round(0).astype(int)

                    top_c = cdf.nlargest(20, 'user_score').copy()
                    osrm_t = []
                    for _, row in top_c.iterrows():
                        try:
                            u = f"http://router.project-osrm.org/route/v1/driving/{row['centroid_lon']},{row['centroid_lat']};{dlon},{dlat}?overview=false"
                            osrm_t.append(round(req.get(u, timeout=5).json()['routes'][0]['duration'] / 60))
                        except Exception:
                            osrm_t.append(row['est_min'])
                    top_c['commute_min'] = osrm_t
                    top_c = top_c.sort_values('user_score', ascending=False).reset_index(drop=True)

                    # ── METRIC CARDS ──
                    fastest_row = top_c.loc[top_c['commute_min'].idxmin()]
                    avg_commute = int(top_c['commute_min'].mean())
                    sweet = top_c[top_c['commute_min'] <= 30]
                    best_pick = sweet.iloc[0] if len(sweet) > 0 else top_c.iloc[0]

                    mc1, mc2, mc3 = st.columns(3)
                    bp_name = best_pick['nta_name'].split('-')[0].strip()
                    bp_min = int(best_pick['commute_min'])
                    bp_rent = f" · ${best_pick[RC]:,.0f}/mo" if RC and pd.notna(best_pick.get(RC)) else ""
                    with mc1:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-val">{int(fastest_row['commute_min'])} min</div>
                            <div class="metric-lbl">Fastest — {fastest_row['nta_name'].split('-')[0].strip()}</div>
                        </div>""", unsafe_allow_html=True)
                    with mc2:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-val">{avg_commute} min</div>
                            <div class="metric-lbl">Avg across top 20</div>
                        </div>""", unsafe_allow_html=True)
                    with mc3:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-val">{bp_min} min</div>
                            <div class="metric-lbl">Best pick — {bp_name}{bp_rent}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("")

                    # ── STYLED NEIGHBORHOOD ROWS ──
                    st.markdown("##### Top Neighborhoods")
                    for rk, (_, r) in enumerate(top_c.head(10).iterrows(), 1):
                        cm = int(r['commute_min'])
                        cb = "cb-green" if cm <= 20 else ("cb-amber" if cm <= 40 else "cb-red")
                        bar_w = min(r['user_score'] * 100, 100)
                        rent_html = ""
                        if RC and pd.notna(r.get(RC)):
                            rent_html = f'<div class="cr-rent">${r[RC]:,.0f}/mo</div>'
                        card_cls = "commute-row-gold" if rk == 1 else "commute-row"
                        st.markdown(f"""<div class="{card_cls}" style="animation-delay:{rk*0.04}s">
                            <div class="cr-rank">#{rk}</div>
                            <div class="cr-info">
                                <div class="cr-name">{r['nta_name']}</div>
                                <div class="cr-boro">{r['borough']}</div>
                                <div class="cr-bar"><div class="cr-bar-fill" style="width:{bar_w:.0f}%"></div></div>
                            </div>
                            <div class="cr-right">
                                <span class="commute-badge {cb}">{cm} min</span>
                                <div class="cr-score">{r['user_score']:.2f}</div>
                                {rent_html}
                            </div>
                        </div>""", unsafe_allow_html=True)

                    # ── COMMUTE DISTRIBUTION ──
                    st.markdown("##### Commute Breakdown")
                    bins = [0, 15, 30, 45, 999]
                    labels_b = ['Under 15 min', '15–30 min', '30–45 min', '45+ min']
                    band_colors = ['#34D399', '#38BDF8', '#FBBF24', '#F87171']
                    top_c['band'] = pd.cut(top_c['commute_min'], bins=bins, labels=labels_b, right=True)
                    band_counts = top_c['band'].value_counts().reindex(labels_b, fill_value=0)

                    fig_dist = go.Figure()
                    for lbl, clr in zip(labels_b, band_colors):
                        fig_dist.add_trace(go.Bar(
                            y=[lbl], x=[band_counts[lbl]], orientation='h', name=lbl,
                            marker=dict(color=clr, line=dict(color='rgba(255,255,255,0.12)', width=0.5)),
                            text=[f"{band_counts[lbl]} neighborhoods"],
                            textposition='inside', textfont=dict(size=11, color='#FFF'),
                        ))
                    fig_dist.update_layout(
                        barmode='stack', showlegend=False,
                        plot_bgcolor='rgba(15,23,42,0)', paper_bgcolor='rgba(15,23,42,0)',
                        xaxis=dict(visible=False),
                        yaxis=dict(tickfont=dict(size=11, color='#E2E8F0'), autorange='reversed'),
                        height=180, margin=dict(l=10, r=20, t=5, b=5),
                        hoverlabel=dict(bgcolor='#1E293B', font_size=12, font_color='#E2E8F0'),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True, key="commute_dist")

                    # ── MAP ──
                    st.markdown("##### Map")
                    fig_map = go.Figure()
                    for i, (_, r) in enumerate(top_c.head(10).iterrows()):
                        fig_map.add_trace(go.Scattermapbox(
                            lat=[r['centroid_lat']], lon=[r['centroid_lon']],
                            mode='markers+text', text=[str(i + 1)],
                            marker=dict(size=20, color='#3B82F6', opacity=0.9),
                            textfont=dict(size=9, color='white', family='Arial Black'),
                            name=r['nta_name'],
                            hovertemplate=f"<b>{r['nta_name']}</b><br>{int(r['commute_min'])} min<extra></extra>",
                        ))
                        fig_map.add_trace(go.Scattermapbox(
                            lat=[r['centroid_lat'], dlat], lon=[r['centroid_lon'], dlon],
                            mode='lines',
                            line=dict(width=1.2, color='rgba(59,130,246,0.3)'),
                            showlegend=False, hoverinfo='skip',
                        ))
                    fig_map.add_trace(go.Scattermapbox(
                        lat=[dlat], lon=[dlon], mode='markers+text',
                        marker=dict(size=16, color='#FCD34D', symbol='star'),
                        text=['Work'], textfont=dict(size=10, color='#FCD34D'),
                        name='Work',
                        hovertemplate=f"<b>Work</b><br>{found_name}<extra></extra>",
                    ))
                    fig_map.update_layout(
                        mapbox=dict(style='carto-darkmatter',
                                    center=dict(lat=(dlat + top_c['centroid_lat'].mean()) / 2,
                                                lon=(dlon + top_c['centroid_lon'].mean()) / 2),
                                    zoom=10),
                        height=450, margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False,
                        paper_bgcolor='rgba(15,23,42,0)',
                    )
                    st.plotly_chart(fig_map, use_container_width=True, key="commute_map")

            except Exception as e:
                st.error(f"Geocoding failed: {e}")

# ═══════ TAB 4: SIMILAR ═════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Find Similar Neighborhoods")
    st.caption("Weighted by your preferences — adjust sliders to find neighborhoods that match what matters to you.")

    ref = st.selectbox("I like...", sorted(df['nta_name'].unique()), key="sim_ref")

    feats = ['safety_score', 'noise_score', 'parks_score', TC]
    sim_weights = np.array([sw, nw, pw, tw], dtype=float)
    X_sim = df[feats].fillna(0).values * sim_weights
    sm = cosine_similarity(X_sim)
    idx = df.index[df['nta_name'] == ref]
    if len(idx) > 0:
        local = df.index.get_loc(idx[0])
        scores = pd.Series(sm[local], index=df['nta_name']).drop(ref, errors='ignore').nlargest(5)
    else:
        scores = pd.Series(dtype=float)

    if len(scores) > 0:
        sres = df[df['nta_name'].isin(scores.index)].copy()
        sres['similarity'] = sres['nta_name'].map(scores.to_dict())
        sres = sres.sort_values('similarity', ascending=False)

        sim_floor = max(sres['similarity'].min() * 100 - 2, 0)
        fig = go.Figure(go.Bar(
            y=sres['nta_name'][::-1],
            x=(sres['similarity'] * 100).values[::-1],
            orientation='h',
            marker_color=['#22C55E' if s > 0.98 else '#3B82F6' if s > 0.95 else '#6366F1'
                          for s in sres['similarity'].values[::-1]],
            text=[f"{s*100:.1f}%" for s in sres['similarity'].values[::-1]],
            textposition='inside',
            textfont=dict(color='white', size=14),
        ))
        fig.update_layout(
            title=f"Top 5 matches for {ref}",
            xaxis_title="Match %",
            xaxis=dict(range=[sim_floor, 100]),
            height=300,
            margin=dict(l=10, r=10, t=40, b=40),
            yaxis=dict(automargin=True),
        )
        st.plotly_chart(fig, use_container_width=True)

        dims = ['Safety', 'Quiet', 'Green Space', 'Transit']
        dcols = ['safety_score', 'noise_score', 'parks_score', TC]
        ref_row = df[df['nta_name'] == ref].iloc[0]
        top_match = sres.iloc[0]
        with st.expander("How do they compare?"):
            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Bar(
                name=ref, x=dims,
                y=[ref_row[c] for c in dcols],
                marker_color='#EF4444',
            ))
            cmp_fig.add_trace(go.Bar(
                name=top_match['nta_name'], x=dims,
                y=[top_match[c] for c in dcols],
                marker_color='#3B82F6',
            ))
            cmp_fig.update_layout(
                barmode='group', yaxis=dict(range=[0, 1], title="Score"),
                height=300, margin=dict(l=10, r=10, t=30, b=30),
            )
            st.plotly_chart(cmp_fig, use_container_width=True)

        scols = ['nta_name', 'borough', 'similarity', 'user_score']
        if RC:
            scols.append(RC)
        if 'cluster_label' in sres.columns:
            scols.append('cluster_label')
        st.dataframe(sres[[c for c in scols if c in sres.columns]].round(3),
                     hide_index=True, use_container_width=True)

# ═══════ TAB 5: CLUSTERS ════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Neighborhood Groups")
    st.caption("We split all neighborhoods into 5 groups using machine learning, based on Safety, Quiet, Parks & Transit scores. Your sidebar sliders pick the best group for you!")

    cl_src = ft if len(ft) > 0 else df
    FEATS_CL = ['safety_score', 'noise_score', 'parks_score', TC]
    X_cl = cl_src[FEATS_CL].fillna(0).values

    K = min(5, len(cl_src))
    if K < 2:
        st.info("Not enough neighborhoods to cluster. Adjust your filters.")
    else:
        km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X_cl)
        cl_src = cl_src.copy()
        cl_src['cluster_id'] = km.labels_

        dim_names = ['Safety', 'Quiet', 'Parks', 'Transit']
        dim_colors = ['#38BDF8', '#A855F7', '#34D399', '#FB923C']
        cluster_emojis_map = {
            'Safe & Steady': '\U0001f6e1️',
            'Peace & Quiet': '\U0001f319',
            'Green Oasis': '\U0001f333',
            'Transit Hub': '\U0001f687',
            'All-Rounder': '⭐',
        }
        cluster_card_colors = ['#38BDF8', '#A855F7', '#FB923C', '#34D399', '#FB7185']

        global_means = X_cl.mean(axis=0)
        cluster_labels = {}
        cluster_descriptions = {}
        used_labels = set()
        label_options = [
            ('Safe & Steady', 0, True, "High safety scores — feel secure walking around"),
            ('Peace & Quiet', 1, False, "Low noise levels — great for relaxation"),
            ('Green Oasis', 2, True, "Lots of parks and green spaces nearby"),
            ('Transit Hub', 3, True, "Easy access to subways and buses"),
            ('All-Rounder', -1, True, "A good mix of everything"),
        ]
        centroids = km.cluster_centers_
        for ci in range(K):
            c = centroids[ci]
            diff = c - global_means
            best_label = None
            for lname, dim_idx, high, desc in label_options:
                if lname in used_labels:
                    continue
                if dim_idx == -1:
                    continue
                threshold = 0.05 if high else -0.05
                if (high and diff[dim_idx] > threshold) or (not high and diff[dim_idx] < threshold):
                    best_label = (lname, desc)
                    break
            if best_label is None:
                for lname, dim_idx, high, desc in label_options:
                    if lname not in used_labels:
                        best_label = (lname, desc)
                        break
            if best_label is None:
                best_label = ('All-Rounder', 'A good mix of everything')
            cluster_labels[ci] = best_label[0]
            cluster_descriptions[ci] = best_label[1]
            used_labels.add(best_label[0])

        cl_src['cluster_name'] = cl_src['cluster_id'].map(cluster_labels)

        user_w = np.array([sw, nw, pw, tw], dtype=float)
        if user_w.sum() > 0:
            user_w_norm = user_w / np.linalg.norm(user_w)
        else:
            user_w_norm = np.ones(4) / 2.0
        cent_norms = np.array([c / (np.linalg.norm(c) + 1e-9) for c in centroids])
        match_scores = cent_norms @ user_w_norm
        best_ci = int(np.argmax(match_scores))
        best_pct = float(match_scores[best_ci]) * 100

        best_name = cluster_labels[best_ci]
        best_emoji = cluster_emojis_map.get(best_name, '⭐')
        best_desc = cluster_descriptions[best_ci]
        n_in_best = int((cl_src['cluster_id'] == best_ci).sum())

        st.markdown(f"""<div class="cluster-hero">
<div class="hero-emoji">{best_emoji}</div>
<div class="hero-title">Your Best Fit: {best_name}</div>
<div class="hero-match">{best_pct:.0f}% match</div>
<div class="hero-desc">{best_desc} &mdash; {n_in_best} neighborhoods in this group</div>
</div>""", unsafe_allow_html=True)

        cols = st.columns(K)
        for ci in range(K):
            with cols[ci]:
                lbl = cluster_labels[ci]
                emoji = cluster_emojis_map.get(lbl, '⭐')
                desc = cluster_descriptions[ci]
                n_count = int((cl_src['cluster_id'] == ci).sum())
                c = centroids[ci]
                badge = '<div class="match-badge">YOUR MATCH</div>' if ci == best_ci else ''
                bars_html = ""
                for di, (dname, dcolor) in enumerate(zip(dim_names, dim_colors)):
                    pct = min(c[di] * 100, 100)
                    bars_html += f"""<div class="cc-bar-wrap">
<span class="cc-bar-label">{dname}</span>
<div class="cc-bar-track"><div class="cc-bar-fill" style="width:{pct:.0f}%;background:{dcolor}"></div></div>
<span class="cc-bar-val">{pct:.0f}%</span>
</div>"""
                st.markdown(f"""<div class="cluster-card">
{badge}
<div class="card-emoji">{emoji}</div>
<div class="card-title">{lbl}</div>
<div class="card-count">{n_count} neighborhoods</div>
{bars_html}
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        dims_display = ['Safety', 'Quiet', 'Parks', 'Transit']
        fig = go.Figure()
        for ci in range(K):
            lbl = cluster_labels[ci]
            vals = centroids[ci].tolist()
            clr = cluster_card_colors[ci % len(cluster_card_colors)]
            fig.add_trace(go.Bar(
                name=f"{cluster_emojis_map.get(lbl, '')} {lbl}",
                x=dims_display, y=vals,
                marker=dict(color=clr, line=dict(color='rgba(255,255,255,0.2)', width=1)),
                text=[f"{v:.0%}" for v in vals],
                textposition='outside', textfont=dict(size=11, color='#CBD5E1'),
            ))
        wv = [sw / W, nw / W, pw / W, tw / W]
        fig.add_trace(go.Scatter(
            name='⭐ Your Priorities', x=dims_display, y=wv,
            mode='lines+markers',
            line=dict(color='#FACC15', width=3, dash='dot'),
            marker=dict(size=10, symbol='diamond', color='#FACC15',
                        line=dict(color='#FEF9C3', width=2)),
        ))
        fig.update_layout(
            barmode='group',
            plot_bgcolor='rgba(15,23,42,0)', paper_bgcolor='rgba(15,23,42,0)',
            yaxis=dict(range=[0, 1.15], title='Average Score', tickformat='.0%',
                       gridcolor='rgba(148,163,184,0.08)', zeroline=False,
                       title_font=dict(size=13, color='#94A3B8'),
                       tickfont=dict(size=11, color='#64748B')),
            xaxis=dict(tickfont=dict(size=13, color='#E2E8F0')),
            height=400, margin=dict(l=50, r=20, t=30, b=10),
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center',
                        font=dict(size=11, color='#CBD5E1'),
                        bgcolor='rgba(30,41,59,0.6)', bordercolor='rgba(71,85,105,0.4)',
                        borderwidth=1),
            bargap=0.15, bargroupgap=0.06,
            hoverlabel=dict(bgcolor='#1E293B', font_size=13,
                            font_color='#E2E8F0', bordercolor='#475569'),
        )
        st.plotly_chart(fig, use_container_width=True, key="cluster_comparison_bar")

        with st.expander("Explore Neighborhoods by Group"):
            sel_cluster = st.selectbox("Pick a group", [cluster_labels[i] for i in range(K)],
                                       key="cluster_explorer_select")
            sel_ci = [k for k, v in cluster_labels.items() if v == sel_cluster][0]
            sub = cl_src[cl_src['cluster_id'] == sel_ci][['nta_name', 'borough', 'safety_score',
                'noise_score', 'parks_score', TC, 'user_score']].sort_values('user_score', ascending=False).copy()
            sub.rename(columns={'nta_name': 'Neighborhood', 'borough': 'Borough',
                                'safety_score': 'Safety', 'noise_score': 'Quiet',
                                'parks_score': 'Parks', TC: 'Transit', 'user_score': 'Your Score'}, inplace=True)
            for c in ['Safety', 'Quiet', 'Parks', 'Transit', 'Your Score']:
                sub[c] = (sub[c] * 100).round(0)
            col_cfg = {
                'Safety': st.column_config.ProgressColumn("Safety", min_value=0, max_value=100, format="%d%%"),
                'Quiet': st.column_config.ProgressColumn("Quiet", min_value=0, max_value=100, format="%d%%"),
                'Parks': st.column_config.ProgressColumn("Parks", min_value=0, max_value=100, format="%d%%"),
                'Transit': st.column_config.ProgressColumn("Transit", min_value=0, max_value=100, format="%d%%"),
                'Your Score': st.column_config.ProgressColumn("Your Score", min_value=0, max_value=100, format="%d%%"),
            }
            st.dataframe(sub, column_config=col_cfg, use_container_width=True, hide_index=True)

            top5 = sub.head(5)
            if len(top5) > 0:
                st.caption(f"Top {len(top5)} in this group by your score:")
                fig_nb = go.Figure()
                score_cols = ['Safety', 'Quiet', 'Parks', 'Transit']
                score_colors = ['#38BDF8', '#A855F7', '#34D399', '#FB923C']
                for sc, scolor in zip(score_cols, score_colors):
                    fig_nb.add_trace(go.Bar(
                        name=sc, y=top5['Neighborhood'], x=top5[sc],
                        orientation='h',
                        marker=dict(color=scolor, line=dict(color='rgba(255,255,255,0.15)', width=0.5)),
                        text=top5[sc].astype(int).astype(str) + '%',
                        textposition='inside', textfont=dict(size=10, color='#FFF'),
                    ))
                fig_nb.update_layout(
                    barmode='group',
                    plot_bgcolor='rgba(15,23,42,0)', paper_bgcolor='rgba(15,23,42,0)',
                    xaxis=dict(title='Score %', range=[0, 105], gridcolor='rgba(148,163,184,0.08)',
                               tickfont=dict(size=11, color='#64748B')),
                    yaxis=dict(tickfont=dict(size=11, color='#E2E8F0'), autorange='reversed'),
                    height=max(200, len(top5) * 55 + 80),
                    margin=dict(l=10, r=20, t=10, b=10),
                    legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center',
                                font=dict(size=11, color='#CBD5E1')),
                    bargap=0.2, bargroupgap=0.05,
                    hoverlabel=dict(bgcolor='#1E293B', font_size=12, font_color='#E2E8F0'),
                )
                st.plotly_chart(fig_nb, use_container_width=True, key="cluster_nb_bars")

        with st.expander("Full Statistics"):
            agg = cl_src.groupby('cluster_name').agg(
                neighborhoods=('nta_name', 'count'),
                safety=('safety_score', 'mean'),
                quiet=('noise_score', 'mean'),
                parks=('parks_score', 'mean'),
                transit=(TC, 'mean'),
                your_score=('user_score', 'mean'),
            )
            agg[['safety', 'quiet', 'parks', 'transit', 'your_score']] = (agg[['safety', 'quiet', 'parks', 'transit', 'your_score']] * 100).round(0)
            agg.rename(columns={'safety': 'Safety %', 'quiet': 'Quiet %', 'parks': 'Parks %',
                                'transit': 'Transit %', 'your_score': 'Your Score %',
                                'neighborhoods': 'Count'}, inplace=True)
            if RC:
                agg['Median Rent'] = cl_src.groupby('cluster_name')[RC].median().round(0).astype(int)
            if 'median_income' in cl_src.columns:
                agg['Median Income'] = cl_src.groupby('cluster_name')['median_income'].median().round(0).astype(int)
            agg = agg.sort_values('Your Score %', ascending=False)
            st.dataframe(agg, use_container_width=True)

# ═══════ TAB 6: LISTINGS ════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Rental Listings")
    st.caption("Filtered by your sidebar borough and budget settings.")

    all_ls = listings.copy()
    if len(redfin) > 0:
        rf_show = redfin.rename(columns={'price': 'price'}).copy()
        if 'title' not in rf_show.columns:
            rf_show['title'] = rf_show.apply(lambda r: f"{int(r['bedrooms'])}BR" if pd.notna(r.get('bedrooms')) else "Studio", axis=1)
        if 'borough' not in rf_show.columns:
            rf_show['borough'] = 'NYC'
        rf_show['source'] = 'Redfin'
        all_ls['source'] = 'Craigslist'
        all_ls = pd.concat([all_ls, rf_show], ignore_index=True)

    if len(all_ls) > 0 and len(top) > 0:
        bl = top['borough'].unique().tolist()
        if 'borough' in all_ls.columns:
            ls = all_ls[all_ls['borough'].isin(bl)].copy()
        else:
            ls = all_ls.copy()
        if budget and 'price' in ls.columns:
            ls = ls[ls['price'] <= budget]
        ls = ls.sort_values('price')
        st.caption(f"{len(ls)} listings in {', '.join(bl)}" + (f" under ${budget:,}/mo" if budget else ""))

        if 'sentiment_polarity' in ls.columns:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(ls, x='sentiment_polarity', nbins=20)
                fig.update_layout(height=260, title='Listing Sentiment', margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(ls, x='sentiment_polarity', y='price', color='borough', opacity=.7)
                fig.update_layout(height=260, title='Sentiment vs Price', margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        show = [c for c in ['title', 'price', 'borough', 'bedrooms', 'sqft', 'no_fee', 'sentiment_label', 'source'] if c in ls.columns]
        st.dataframe(ls[show].head(50).rename(columns={'title': 'location'}), hide_index=True, use_container_width=True)
    else:
        st.info("No listing data.")

# ═══════ TAB 7: TRENDS ══════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 311 Complaint Trends")
    st.caption("City-wide 311 data — independent of your sidebar priority settings.")
    if len(ts_data) > 100:
        ts = ts_data.copy()
        ts['date'] = pd.to_datetime(ts['created_date'], errors='coerce')
        ts = ts.dropna(subset=['date'])

        c1, c2 = st.columns(2)
        with c1:
            daily = ts.set_index('date').resample('D').size().reset_index(name='n')
            daily['avg'] = daily['n'].rolling(7, center=True).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily['date'], y=daily['n'], mode='lines', name='Daily', opacity=.3))
            fig.add_trace(go.Scatter(x=daily['date'], y=daily['avg'], mode='lines', name='7-day avg',
                                     line=dict(color='#EF4444', width=2)))
            fig.update_layout(height=300, title='Daily Volume', margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            dow = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ts['d'] = ts['date'].dt.day_name().str[:3]
            dc = ts['d'].value_counts().reindex(dow).fillna(0)
            fig = px.bar(x=dc.index, y=dc.values,
                          labels={'x': 'Day', 'y': 'Complaints'})
            fig.update_layout(height=300, title='By Day of Week', margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        sel_nta = st.selectbox("Zoom into a neighborhood",
                                ["All NYC"] + sorted(ts['nta_name'].dropna().unique().tolist()), key="ts_nta")
        ts_filt = ts if sel_nta == "All NYC" else ts[ts['nta_name'] == sel_nta]

        t5 = ts_filt['complaint_type'].value_counts().head(5).index
        wk = ts_filt[ts_filt['complaint_type'].isin(t5)].set_index('date').groupby(
            'complaint_type').resample('W').size().reset_index(name='n')
        fig = px.line(wk, x='date', y='n', color='complaint_type')
        fig.update_layout(height=350, title=f'Weekly by Type — {sel_nta}', margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series data.")

# ═══════ TAB 8: DEMOGRAPHICS ════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### Neighborhood Demographics")
    st.caption("Census data filtered by your sidebar settings, plotted against your weighted score.")
    demo_src = ft if len(ft) > 0 else df
    demo_cols = [c for c in ['median_income', 'total_population', 'median_age', 'college_rate',
                              'census_median_rent'] if c in demo_src.columns]
    if demo_cols:
        metric = st.selectbox("Metric", demo_cols,
                               format_func=lambda x: x.replace('_', ' ').replace('census ', '').title())

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(demo_src, x=metric, nbins=30, color='borough', barmode='overlay', opacity=.7)
            fig.update_layout(height=380, title=f'{metric.replace("_", " ").title()} Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ba = demo_src.groupby('borough')[metric].median().sort_values(ascending=False)
            fig = px.bar(x=ba.values, y=ba.index, orientation='h')
            fig.update_layout(height=380, title='Median by Borough', yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(demo_src, x=metric, y='user_score', color='borough', hover_name='nta_name',
                          opacity=.4,
                          labels={metric: metric.replace('_', ' ').title(), 'user_score': 'Your Score'})
        top_demo = top[top[metric].notna()].head(10) if metric in top.columns else pd.DataFrame()
        if len(top_demo) > 0:
            fig.add_trace(go.Scatter(
                x=top_demo[metric], y=top_demo['user_score'], mode='markers+text',
                text=[f"#{i+1}" for i in range(len(top_demo))],
                textposition='top center', textfont=dict(size=9, color='#3B82F6'),
                marker=dict(size=14, color='#3B82F6', symbol='star', line=dict(width=1, color='white')),
                hovertext=top_demo['nta_name'], hoverinfo='text+x+y',
                name='Your Top 10', showlegend=True))
        fig.update_layout(height=400, title=f'{metric.replace("_", " ").title()} vs Livability')
        st.plotly_chart(fig, use_container_width=True)

        num = ['user_score', 'safety_score', 'noise_score', 'parks_score', TC] + demo_cols
        corr = demo_src[num].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto')
        fig.update_layout(height=500, title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run Census cells in the notebook.")

# ═══════ TAB 9: DATA ════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("### Full Dataset")
    st.caption("Filtered by your sidebar borough and budget — sorted by your weighted score.")
    defaults = ['nta_name', 'borough', 'user_score', 'safety_score', 'noise_score', 'parks_score', TC]
    if RC:
        defaults.append(RC)
    if 'cluster_label' in df.columns:
        defaults.append('cluster_label')
    for d in ['median_income', 'total_population', 'median_age', 'college_rate']:
        if d in df.columns:
            defaults.append(d)

    sel = st.multiselect("Columns", ft.columns.tolist(),
                          default=[c for c in defaults if c in ft.columns])
    if sel:
        st.dataframe(ft[sel].sort_values('user_score', ascending=False).round(3),
                     height=500, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", ft[sel].to_csv(index=False), "livability.csv", "text/csv")

    st.markdown("#### Score Distributions")
    wpct = lambda v: f"{v/W*100:.0f}%" if W > 0 else "0%"
    st.caption(f"Current weights: Safety {wpct(sw)} · Quiet {wpct(nw)} · Parks {wpct(pw)} · Transit {wpct(tw)}")
    box = ['safety_score', 'noise_score', 'parks_score', TC, 'user_score']
    fig = go.Figure()
    for d in box:
        fig.add_trace(go.Box(y=ft[d], name=d.replace('_score', '').replace('_', ' ').title(), boxpoints='outliers'))
    fig.update_layout(height=380, yaxis_title='Score (0-1)')
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SCROLL PRESERVATION — restore position after Streamlit reruns
# ─────────────────────────────────────────────────────────────────────────────
components.html("""
<script>
(function() {
    const KEY = '__nyc_scroll_y';
    const SKEY = '__nyc_sidebar_y';
    const p = window.parent.document;

    const el = p.querySelector('[data-testid="stMain"]')
             || p.querySelector('section.main');
    const sb = p.querySelector('[data-testid="stSidebar"]');
    const sbInner = sb && (sb.querySelector('[data-testid="stSidebarContent"]') || sb.querySelector('.st-emotion-cache-1gwvy71') || sb.firstElementChild);

    // Restore main scroll
    if (el) {
        const saved = sessionStorage.getItem(KEY);
        if (saved !== null) {
            const y = parseInt(saved, 10);
            requestAnimationFrame(() => {
                el.scrollTop = y;
                setTimeout(() => { el.scrollTop = y; }, 50);
                setTimeout(() => { el.scrollTop = y; }, 150);
            });
        }
        let tid;
        el.addEventListener('scroll', function() {
            clearTimeout(tid);
            tid = setTimeout(() => sessionStorage.setItem(KEY, el.scrollTop), 80);
        }, {passive: true});
    }

    // Restore sidebar scroll
    if (sbInner) {
        const sSaved = sessionStorage.getItem(SKEY);
        if (sSaved !== null) {
            const sy = parseInt(sSaved, 10);
            requestAnimationFrame(() => {
                sbInner.scrollTop = sy;
                setTimeout(() => { sbInner.scrollTop = sy; }, 50);
            });
        }
        let stid;
        sbInner.addEventListener('scroll', function() {
            clearTimeout(stid);
            stid = setTimeout(() => sessionStorage.setItem(SKEY, sbInner.scrollTop), 80);
        }, {passive: true});
    }
})();
</script>
""", height=0)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="footer-v2">
<span>Polis Technologies</span> · Section 1 · {len(df)} NTAs · 9 data sources · ML-powered · interactive analysis<br>
NYC Open Data · Census ACS · Citi Bike · MTA · Redfin · Craigslist
</div>""", unsafe_allow_html=True)
