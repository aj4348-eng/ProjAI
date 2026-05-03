"""
NYC "Where Should I Live?" — Streamlit Dashboard
=================================================
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
import sqlite3, os, time, json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

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
        # Neon connection strings work with psycopg2 driver
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
# CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NYC Livability", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="expanded")

# ── Custom Plotly template ──────────────────────────────────────────────
import plotly.io as pio
pio.templates["polis"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", color="#e8e6e1", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=["#d4a574", "#8a9a5b", "#a87c5f", "#5b7b8a",
                  "#b8956a", "#7a8b6f", "#9d6b4f", "#4d6b78"],
        xaxis=dict(gridcolor="rgba(232,230,225,0.06)", zerolinecolor="rgba(232,230,225,0.1)",
                   linecolor="rgba(232,230,225,0.2)", tickcolor="rgba(232,230,225,0.2)"),
        yaxis=dict(gridcolor="rgba(232,230,225,0.06)", zerolinecolor="rgba(232,230,225,0.1)",
                   linecolor="rgba(232,230,225,0.2)", tickcolor="rgba(232,230,225,0.2)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(232,230,225,0.1)"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
)
pio.templates.default = "polis"

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg: #0a0a0a;
  --surface: #141414;
  --surface-2: #1a1a1a;
  --border: rgba(232,230,225,0.08);
  --border-strong: rgba(232,230,225,0.16);
  --text: #e8e6e1;
  --text-muted: #807c75;
  --text-faint: #4a4845;
  --accent: #d4a574;
  --accent-soft: rgba(212,165,116,0.12);
  --good: #8a9a5b;
  --warn: #c9a961;
  --bad: #a87c5f;
  --primary: #3B82F6;
  --primary-light: rgba(59,130,246,0.35);
  --green: #10B981;
}

.stApp { background: var(--bg) !important; font-family: 'Inter', system-ui, sans-serif; color: var(--text); }
/* Hide Streamlit chrome BUT keep the sidebar collapse control accessible */
#MainMenu, footer { visibility: hidden; height: 0; }
header[data-testid="stHeader"] {
  background: transparent !important;
  height: auto !important;
  visibility: visible !important;
}
/* Hide everything inside the header EXCEPT the sidebar toggle */
header[data-testid="stHeader"] > div:not(:has([data-testid="stSidebarCollapsedControl"])) { display: none; }
header[data-testid="stHeader"] [data-testid="stToolbar"] { display: none !important; }
/* Make sure the sidebar collapse/expand button is always visible and styled */
[data-testid="stSidebarCollapsedControl"],
button[kind="header"][data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 999 !important;
  color: var(--accent) !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
  color: var(--accent) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 4px !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
  border-color: var(--accent) !important;
}
.block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; max-width: 1400px; }

h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; }

.section-h {
  font-family: 'Fraunces', serif; font-size: 1.6rem; font-weight: 400;
  letter-spacing: -0.025em; color: var(--text); margin: 0.5rem 0 0.25rem;
}
.section-sub { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.5rem; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] h2 {
  font-family: 'Fraunces', serif !important; font-weight: 400 !important;
  font-size: 1.2rem !important; letter-spacing: -0.02em; margin-bottom: 1rem; color: var(--text) !important;
}
[data-testid="stSidebar"] label {
  font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--text-muted) !important; font-weight: 500;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background: var(--surface-2) !important; border: 1px solid var(--border) !important;
  border-radius: 6px !important; color: var(--text) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div { background: var(--border-strong) !important; }
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div { background: var(--accent) !important; }

.hero {
  padding: 3rem 0 2rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; position: relative;
}
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.18em; color: var(--accent); margin-bottom: 1rem;
}
.hero-title {
  font-family: 'Fraunces', serif; font-weight: 400; font-size: clamp(2.2rem, 5vw, 4rem);
  line-height: 1.02; letter-spacing: -0.04em; color: var(--text); margin: 0 0 1rem; max-width: 900px;
}
.hero-title em { font-style: italic; color: var(--accent); font-weight: 300; }
.hero-meta {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted);
  letter-spacing: 0.06em; text-transform: uppercase; display: flex; gap: 1.5rem; flex-wrap: wrap;
}
.hero-meta span::before { content: "—"; color: var(--text-faint); margin-right: 0.5rem; }
.hero-meta span:first-child::before { content: ""; margin: 0; }

.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2.5rem; }
.stat {
  padding: 1.25rem 1.4rem; background: var(--surface); border: 1px solid var(--border);
  border-left: 2px solid var(--accent); transition: border-color 0.25s ease, transform 0.25s ease;
}
.stat:hover { border-left-color: var(--text); transform: translateY(-2px); }
.stat-num {
  font-family: 'Fraunces', serif; font-size: 1.9rem; font-weight: 400;
  letter-spacing: -0.025em; color: var(--text); line-height: 1;
}
.stat-num.small { font-size: 1.25rem; }
.stat-lbl {
  font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; text-transform: uppercase;
  letter-spacing: 0.14em; color: var(--text-muted); margin-top: 0.6rem;
}

.stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border-bottom: 1px solid var(--border);
  gap: 0; padding: 0; margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border: none !important;
  border-bottom: 1px solid transparent !important; border-radius: 0 !important;
  padding: 0.85rem 1.4rem !important; font-family: 'Inter', sans-serif !important;
  font-size: 0.82rem !important; font-weight: 500 !important; color: var(--text-muted) !important;
  letter-spacing: 0.01em; transition: color 0.2s ease, border-color 0.2s ease; margin-right: 0.25rem;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; }
.stTabs [aria-selected="true"] { color: var(--text) !important; border-bottom-color: var(--accent) !important; }
.stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p { font-size: 0.82rem !important; margin: 0 !important; }

.rank {
  display: grid; grid-template-columns: 50px 1fr 220px 110px; gap: 1.5rem;
  align-items: center; padding: 1.5rem 0; border-bottom: 1px solid var(--border);
  transition: background 0.2s ease, padding 0.2s ease;
}
.rank:hover { background: var(--surface); padding-left: 0.8rem; padding-right: 0.8rem; }
.rank:last-child { border-bottom: none; }
.rank-num {
  font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 300;
  color: var(--text-faint); line-height: 1; font-feature-settings: 'tnum';
}
.rank-1 .rank-num { color: var(--accent); font-size: 2.4rem; }
.rank-name {
  font-family: 'Fraunces', serif; font-size: 1.1rem; font-weight: 500;
  color: var(--text); letter-spacing: -0.015em; margin-bottom: 0.25rem;
}
.rank-1 .rank-name { font-size: 1.3rem; }
.rank-meta {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.08em;
}
.rank-tag {
  display: inline-block; margin-left: 0.6rem; padding: 1px 8px;
  font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--accent); border: 1px solid var(--accent); border-radius: 2px;
  font-family: 'JetBrains Mono', monospace;
}
.rank-dims {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 14px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; color: var(--text-muted);
}
.rank-dims .v { color: var(--text); margin-left: 4px; }
.rank-bar-wrap {
  margin-top: 0.5rem; height: 2px; background: var(--border); width: 100%; position: relative;
}
.rank-bar-fill {
  position: absolute; height: 100%; background: var(--accent);
  transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.rank-score { text-align: right; }
.rank-score-num {
  font-family: 'Fraunces', serif; font-size: 1.6rem; font-weight: 400;
  color: var(--text); letter-spacing: -0.02em; font-feature-settings: 'tnum';
}
.rank-1 .rank-score-num { color: var(--accent); }
.rank-score-rent {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  color: var(--text-muted); margin-top: 0.25rem; letter-spacing: 0.04em;
}

.stTextInput input, .stSelectbox > div > div, .stNumberInput input {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: 4px !important;
}
.stTextInput input:focus { border-color: var(--accent) !important; }
.stToggle [data-baseweb="checkbox"] [role="switch"][aria-checked="true"] { background: var(--accent) !important; }
.stProgress > div > div > div > div { background: var(--accent) !important; }
.stDataFrame { border: 1px solid var(--border); border-radius: 4px; }

.stButton > button, .stDownloadButton > button {
  background: transparent !important; border: 1px solid var(--border-strong) !important;
  color: var(--text) !important; border-radius: 4px !important;
  font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
  text-transform: uppercase; letter-spacing: 0.1em; padding: 0.6rem 1.2rem !important;
  transition: all 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
}

.pill {
  display: inline-block; padding: 2px 10px; font-size: 0.65rem;
  letter-spacing: 0.1em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
  border-radius: 2px; border: 1px solid;
}
.pg { color: var(--good); border-color: var(--good); }
.pb { color: var(--accent); border-color: var(--accent); }
.pa { color: var(--warn); border-color: var(--warn); }
.pr { color: var(--bad); border-color: var(--bad); }

.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; font-size: 0.78rem !important; }
[data-testid="chatMessage"] { background: var(--surface) !important; border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.25rem; }
hr { border-color: var(--border) !important; }

.footer {
  margin-top: 4rem; padding-top: 2rem; border-top: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  color: var(--text-muted); letter-spacing: 0.05em; text-align: center;
}
.footer .accent { color: var(--accent); }

@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.rank, .stat { animation: fadeUp 0.4s ease both; }
.rank:nth-child(2) { animation-delay: 0.05s; }
.rank:nth-child(3) { animation-delay: 0.1s; }
.rank:nth-child(4) { animation-delay: 0.15s; }
.rank:nth-child(5) { animation-delay: 0.2s; }

@media (max-width: 768px) {
  .stat-row { grid-template-columns: repeat(2, 1fr); }
  .rank { grid-template-columns: 40px 1fr; gap: 1rem; }
  .rank > div:nth-child(3), .rank > div:nth-child(4) { grid-column: 2; padding-top: 0.5rem; }
}

/* ── Profile tab ── */
.profile-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    padding: 1.4rem 0 1rem; border-bottom: 1px solid rgba(212,165,116,0.15);
    margin-bottom: 1.5rem;
}
.profile-name {
    font-family: 'Fraunces', serif; font-size: 2.2rem; font-weight: 400;
    color: #e8e6e1; line-height: 1.1; letter-spacing: -0.025em;
}
.profile-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    color: #807c75; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem;
}
.profile-rank-badge {
    background: rgba(212,165,116,0.1); border: 1px solid rgba(212,165,116,0.4);
    border-radius: 10px; padding: 0.5rem 1rem; text-align: center; flex-shrink: 0;
}
.profile-rank-num {
    font-family: 'Fira Code', monospace; font-size: 1.8rem; font-weight: 700;
    color: var(--accent); line-height: 1;
}
.profile-rank-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    color: #807c75; text-transform: uppercase; letter-spacing: 0.1em;
}
.profile-score-card {
    background: #141414; border: 1px solid rgba(232,230,225,0.08);
    border-left: 2px solid var(--accent); border-radius: 8px;
    padding: 0.9rem 1rem;
}
.profile-score-val {
    font-family: 'Fraunces', serif; font-size: 1.6rem; color: #e8e6e1;
    line-height: 1; font-weight: 400;
}
.profile-score-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #807c75;
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.35rem;
}
.profile-score-bar { height: 2px; background: rgba(232,230,225,0.08); margin-top: 0.55rem; position: relative; }
.profile-score-pct {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    margin-top: 0.35rem; letter-spacing: 0.06em;
}
.profile-section-title {
    font-family: 'Fraunces', serif; font-size: 1rem; font-weight: 500;
    color: #e8e6e1; margin: 1.5rem 0 0.6rem;
}
.profile-demo-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0; border-bottom: 1px solid rgba(232,230,225,0.06);
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
}
.profile-demo-lbl { color: #807c75; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.67rem; }
.profile-demo-val { color: #e8e6e1; }
.profile-listing {
    padding: 0.8rem 0.9rem; margin-bottom: 0.5rem;
    background: rgba(30,41,59,0.5); border: 1px solid rgba(232,230,225,0.06);
    border-radius: 8px;
}
.profile-listing-title { font-family: 'Fraunces', serif; color: #e8e6e1; font-size: 0.92rem; }
.profile-listing-price { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--accent); margin-top: 0.2rem; }
.profile-listing-src { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #807c75; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.12rem; }

/* ── METRIC CARDS ── */
/* ── AI Advisor ── */
.advisor-bubble {
    background: linear-gradient(135deg, rgba(212,165,116,0.07), rgba(30,41,59,0.9));
    border: 1px solid rgba(212,165,116,0.3);
    border-left: 3px solid var(--accent);
    border-radius: 0 14px 14px 14px;
    padding: 1.1rem 1.3rem;
    margin: 0.5rem 0 1rem;
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.7;
}
.advisor-bubble strong { color: var(--accent); }
.advisor-bubble ul { padding-left: 1.2rem; margin: 0.4rem 0; }
.advisor-bubble li { margin-bottom: 0.25rem; }
.advisor-bubble h3, .advisor-bubble h4 {
    color: var(--accent);
    font-size: 0.95rem;
    margin: 0.6rem 0 0.2rem;
    letter-spacing: 0.02em;
}
.advisor-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--accent);
    opacity: 0.7;
    margin-bottom: 0.2rem;
}
/* ── Profile sections fancy ── */
.demo-stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px,1fr)); gap: 0.6rem; margin-bottom: 1.2rem; }
.demo-stat {
    background: rgba(20,20,20,0.8); border: 1px solid rgba(232,230,225,0.07);
    border-top: 2px solid rgba(212,165,116,0.4); border-radius: 8px;
    padding: 0.8rem 0.9rem;
}
.demo-stat-num { font-family: 'Fraunces', serif; font-size: 1.4rem; color: #e8e6e1; font-weight: 400; line-height: 1; }
.demo-stat-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #807c75; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.35rem; }
.demo-stat-vs { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; margin-top: 0.3rem; }

.vibe-grid { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.vibe-pill {
    background: linear-gradient(135deg, rgba(212,165,116,0.1), rgba(30,41,59,0.8));
    border: 1px solid rgba(212,165,116,0.25); border-radius: 12px;
    padding: 0.7rem 1rem; display: flex; align-items: center; gap: 0.6rem; flex: 1; min-width: 110px;
}
.vibe-icon { font-size: 1.4rem; }
.vibe-num { font-family: 'Fraunces', serif; font-size: 1.3rem; color: #d4a574; font-weight: 400; line-height: 1; }
.vibe-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #807c75; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

.sim-card {
    background: rgba(20,20,20,0.7); border: 1px solid rgba(232,230,225,0.07);
    border-radius: 10px; padding: 0.7rem 0.9rem; margin-bottom: 0.45rem;
    display: flex; align-items: center; justify-content: space-between;
    transition: border-color 0.2s;
}
.sim-card:hover { border-color: rgba(212,165,116,0.3); }
.sim-name { font-family: 'Fraunces', serif; font-size: 0.95rem; color: #e8e6e1; }
.sim-boro { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #807c75; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }
.sim-score-wrap { text-align: right; flex-shrink: 0; }
.sim-score-num { font-family: 'Fira Code', monospace; font-size: 0.85rem; color: #d4a574; font-weight: 600; }
.sim-bar-track { width: 60px; height: 3px; background: rgba(232,230,225,0.08); border-radius: 2px; margin-top: 0.3rem; overflow: hidden; }
.sim-bar-fill { height: 100%; background: linear-gradient(90deg,#d4a574,#c9a961); border-radius: 2px; }

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

/* ── COMMUTE TAB ── */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
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

/* ── Cluster cards ────────────────────────────────────────────────── */
.cluster-hero {
    background: linear-gradient(135deg, rgba(212,165,116,0.12), rgba(201,169,97,0.18));
    border: 2px solid rgba(212,165,116,0.5);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    max-width: 600px;
    margin: 0 auto 1.2rem;
    box-shadow: 0 0 30px rgba(212,165,116,0.12), 0 8px 32px rgba(0,0,0,0.3);
    text-align: center;
}
.cluster-hero .hero-emoji { font-size: 3rem; margin-bottom: 0.3rem; }
.cluster-hero .hero-title { font-size: 1.5rem; font-weight: 700; color: #F1F5F9; margin: 0.2rem 0; }
.cluster-hero .hero-match { font-size: 2rem; font-weight: 800; background: linear-gradient(90deg,#d4a574,#C9A961); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.cluster-hero .hero-desc { color: #94A3B8; font-size: 0.92rem; margin-top: 0.4rem; }

.cluster-card {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(71,85,105,0.4);
    border-radius: 14px;
    padding: 1rem 0.9rem;
    transition: transform 0.2s, box-shadow 0.2s;
    text-align: center;
    height: 220px;
    box-sizing: border-box;
}
.cluster-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.cluster-card.best-match {
    border: 2px solid rgba(212,165,116,0.8);
    box-shadow: 0 0 20px rgba(212,165,116,0.25), 0 4px 16px rgba(0,0,0,0.3);
    background: rgba(212,165,116,0.08);
}
.cluster-card .card-emoji { font-size: 2rem; }
.cluster-card .card-title { font-size: 0.95rem; font-weight: 600; color: #E2E8F0; margin: 0.3rem 0 0.2rem; }
.cluster-card .card-count { font-size: 0.78rem; color: #64748B; margin-bottom: 0.5rem; }


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
</style>""", unsafe_allow_html=True)

# JS safety net: ensure the sidebar collapse button stays visible no matter what.
# Streamlit version differences sometimes break our CSS selectors, so we also
# unhide the control via JavaScript on every render.
st.markdown("""<script>
(function() {
  const ensureToggleVisible = () => {
    document.querySelectorAll(
      '[data-testid="stSidebarCollapsedControl"], ' +
      '[data-testid="stSidebarCollapseButton"], ' +
      'button[kind="header"]'
    ).forEach(el => {
      el.style.visibility = 'visible';
      el.style.opacity = '1';
      el.style.zIndex = '9999';
      el.style.pointerEvents = 'auto';
    });
  };
  ensureToggleVisible();
  // Re-apply on DOM mutations (Streamlit re-renders frequently)
  const obs = new MutationObserver(ensureToggleVisible);
  obs.observe(document.body, { childList: true, subtree: true });
})();
</script>""", unsafe_allow_html=True)

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
sim_mx = load_sim()
geo = load_geojson()

# Detect best rent column
RC = next((c for c in ['redfin_median_rent', 'census_median_rent', 'predicted_rent']
           if c in df.columns and df[c].notna().sum() > 20), None)
TC = 'transit_plus_score' if 'transit_plus_score' in df.columns else 'transit_score'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Priorities")
    has_vib = 'vibrancy_score' in df.columns
    persona = st.selectbox("Persona", ["Custom", "Professional", "Family",
                                        "Student", "Retiree", "Creative"])
    # Tuples are (safety, quiet, parks, transit, vibrancy)
    P = {"Custom":       (25, 20, 15, 25, 15),
         "Professional": (20,  8,  8, 50, 14),
         "Family":       (35, 28, 24,  5,  8),
         "Student":      (12,  8,  4, 60, 16),
         "Retiree":      (28, 38, 24,  5,  5),
         "Creative":     (12, 18, 18, 28, 24)}
    d = P.get(persona, (25, 20, 15, 25, 15))
    # If vibrancy data isn't available, redistribute its weight to the others proportionally
    if not has_vib:
        rest = sum(d[:4])
        if rest > 0:
            d = (d[0] + d[4] * d[0] // rest,
                 d[1] + d[4] * d[1] // rest,
                 d[2] + d[4] * d[2] // rest,
                 d[3] + d[4] * d[3] // rest,
                 0)
    sw = st.slider("Safety", 0, 100, d[0], 5)
    nw = st.slider("Quiet", 0, 100, d[1], 5)
    pw = st.slider("Parks", 0, 100, d[2], 5)
    tw = st.slider("Transit", 0, 100, d[3], 5)
    if has_vib:
        vw = st.slider("Vibrancy", 0, 100, d[4], 5,
                       help="Restaurants, bars, cafes, gyms, shops")
    else:
        vw = 0
    st.divider()
    budget_mode = st.selectbox("Budget", ["No filter", "Manual budget", "Income-based"])
    if budget_mode == "Manual budget":
        mb = st.slider("Max $/mo", 1500, 10000, 3500, 100)
    elif budget_mode == "Income-based":
        income = st.number_input("Annual income ($)", min_value=10000, max_value=500000,
                                  value=80000, step=5000)
        rent_pct = st.slider("Max % on rent", 25, 40, 30)
        mb = int(round(income / 12 * rent_pct / 100, -2))
        st.caption(f"= ${mb:,.0f}/mo (30% rule)")
    else:
        mb = None
    st.divider()
    boros = ["All"] + sorted(df['borough'].dropna().unique().tolist())
    sb = st.selectbox("Borough", boros)
    sb = None if sb == "All" else sb
    topn = st.slider("Results", 5, 30, 10)
    st.divider()
    _db = "Postgres" if USE_POSTGRES else "SQLite"
    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
        f'color:#807c75;letter-spacing:0.1em;text-transform:uppercase;">'
        f'Database  ·  <span style="color:#d4a574">{_db}</span></div>',
        unsafe_allow_html=True
    )

# Compute user score
W = max(sw + nw + pw + tw + vw, 1)
df['user_score'] = (sw / W * df['safety_score'] + nw / W * df['noise_score'] +
                    pw / W * df['parks_score'] + tw / W * df[TC])
if has_vib and vw > 0:
    df['user_score'] = df['user_score'] + (vw / W * df['vibrancy_score'].fillna(0))

ft = df.copy()
if mb and RC:
    ft = ft[ft[RC].fillna(99999) <= mb]
if sb:
    ft = ft[ft['borough'] == sb]
ft = ft.sort_values('user_score', ascending=False)
top = ft.head(topn)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<div class="hero">
<div class="hero-eyebrow">Polis Technologies — Section 01</div>
<h1 class="hero-title">Where in New York<br>should you actually <em>live</em>?</h1>
<div class="hero-meta">
<span>197 neighborhoods</span>
<span>100K+ records</span>
<span>9 data sources</span>
<span>ML-powered</span>
</div>
</div>""", unsafe_allow_html=True)

# Stat row
nm = top.iloc[0]['nta_name'].split('-')[0].strip() if len(top) > 0 else "—"
rv = f"${top.iloc[0][RC]:,.0f}" if len(top) > 0 and RC and pd.notna(top.iloc[0].get(RC)) else "—"
top_score = f"{top.iloc[0]['user_score']:.3f}" if len(top) > 0 else "—"

st.markdown(f"""<div class="stat-row">
<div class="stat"><div class="stat-num">{len(df)}</div><div class="stat-lbl">Neighborhoods</div></div>
<div class="stat"><div class="stat-num">{len(ft)}</div><div class="stat-lbl">Matching filters</div></div>
<div class="stat"><div class="stat-num small">{nm}</div><div class="stat-lbl">Top pick &middot; {top_score}</div></div>
<div class="stat"><div class="stat-num">{rv}</div><div class="stat-lbl">Est. monthly rent</div></div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Rankings", "Profile", "Map", "Advisor", "Commute", "Similar",
                "Compare", "Clusters", "Listings", "Demographics", "Data"])

# ═══════ TAB 1: PICKS ═════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-h">Top neighborhoods for you</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ranked by your priority weights. Drag the sidebar sliders to re-rank.</div>', unsafe_allow_html=True)
    if len(top) == 0:
        st.warning("No results. Adjust filters.")
    else:
        rank_html = ""
        for rk, (_, r) in enumerate(top.iterrows(), 1):
            s = r['user_score']
            cluster_tag = ""
            if 'cluster_label' in r.index and pd.notna(r.get('cluster_label')):
                cluster_tag = f'<span class="rank-tag">{r["cluster_label"]}</span>'
            rent_str = ""
            if RC and pd.notna(r.get(RC)):
                rent_str = f'<div class="rank-score-rent">${r[RC]:,.0f}/mo</div>'
            dims_html = (
                f'<div><span>Safety</span><span class="v">{r["safety_score"]:.2f}</span></div>'
                f'<div><span>Quiet</span><span class="v">{r["noise_score"]:.2f}</span></div>'
                f'<div><span>Parks</span><span class="v">{r["parks_score"]:.2f}</span></div>'
                f'<div><span>Transit</span><span class="v">{r[TC]:.2f}</span></div>'
            )
            # Add Vibrancy if present (from restaurants + OSM amenities)
            if 'vibrancy_score' in r.index and pd.notna(r.get('vibrancy_score')):
                dims_html += f'<div><span>Vibrancy</span><span class="v">{r["vibrancy_score"]:.2f}</span></div>'
            row_class = "rank rank-1" if rk == 1 else "rank"
            rk_str = str(rk).zfill(2)
            bar_w = min(s * 100, 100)
            rank_html += (
                f'<div class="{row_class}">'
                f'<div class="rank-num">{rk_str}</div>'
                f'<div>'
                f'<div class="rank-name">{r["nta_name"]}{cluster_tag}</div>'
                f'<div class="rank-meta">{r["borough"]}</div>'
                f'<div class="rank-bar-wrap"><div class="rank-bar-fill" style="width:{bar_w:.1f}%"></div></div>'
                f'</div>'
                f'<div class="rank-dims">{dims_html}</div>'
                f'<div class="rank-score">'
                f'<div class="rank-score-num">{s:.3f}</div>'
                f'{rent_str}'
                f'</div>'
                f'</div>'
            )
        st.markdown(rank_html, unsafe_allow_html=True)

# ═══════ TAB 1: PROFILE ══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-h">Neighborhood profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pick any neighborhood to see its full breakdown.</div>', unsafe_allow_html=True)

    _nta_opts = sorted(df['nta_name'].unique())
    _top_nta = ft.iloc[0]['nta_name'] if len(ft) > 0 else df.nlargest(1, 'user_score').iloc[0]['nta_name']
    _default_idx = _nta_opts.index(_top_nta) if _top_nta in _nta_opts else 0
    _profile_key = f"profile_nta_{sw}_{nw}_{pw}_{tw}_{sb}_{mb}"
    sel_nta = st.selectbox("Neighborhood", _nta_opts, index=_default_idx, key=_profile_key)
    n = df[df['nta_name'] == sel_nta].iloc[0]

    # rank among all neighborhoods by composite score
    rank_col = 'composite_score' if 'composite_score' in df.columns else 'user_score'
    nta_rank = int((df[rank_col].fillna(0) > float(n.get(rank_col, 0))).sum()) + 1

    cluster_str = ""
    if 'cluster_label' in n.index and pd.notna(n.get('cluster_label')):
        cluster_str = f' &nbsp;·&nbsp; {n["cluster_label"]}'

    overall_score = n.get('user_score', n.get('composite_score', 0))

    st.markdown(f"""
<div class="profile-header">
  <div>
    <div class="profile-name">{n["nta_name"]}</div>
    <div class="profile-meta">{n["borough"]}{cluster_str}</div>
  </div>
  <div class="profile-rank-badge">
    <div class="profile-rank-num">#{nta_rank}</div>
    <div class="profile-rank-lbl">of {len(df)}</div>
    <div style="font-family:'Fira Code',monospace;font-size:0.75rem;color:#d4a574;margin-top:0.3rem">{float(overall_score):.3f}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Score cards + radar ──
    score_dims = [('Safety', 'safety_score'), ('Quiet', 'noise_score'),
                  ('Parks', 'parks_score'), ('Transit', TC)]
    if 'rent_score' in df.columns: score_dims.append(('Afford.', 'rent_score'))
    if 'vibrancy_score' in df.columns: score_dims.append(('Vibrancy', 'vibrancy_score'))

    col_cards, col_radar = st.columns([3, 2])

    with col_cards:
        valid_dims = [(lbl, col) for lbl, col in score_dims if col in df.columns and pd.notna(n.get(col))]
        n_cols = min(3, len(valid_dims))
        card_cols = st.columns(n_cols)
        for i, (label, col) in enumerate(valid_dims):
            val = float(n[col])
            pctile = int((df[col].fillna(0) < val).sum() / len(df) * 100)
            pc = "#8a9a5b" if pctile >= 70 else "#c9a961" if pctile >= 40 else "#a87c5f"
            bar_w = val * 100
            with card_cols[i % n_cols]:
                st.markdown(f"""<div class="profile-score-card">
<div class="profile-score-val">{val:.2f}</div>
<div class="profile-score-lbl">{label}</div>
<div class="profile-score-bar"><div style="position:absolute;height:100%;width:{bar_w:.1f}%;background:#d4a574"></div></div>
<div class="profile-score-pct" style="color:{pc}">{pctile}th pct</div>
</div>""", unsafe_allow_html=True)

    with col_radar:
        radar_dims = [(lbl, col) for lbl, col in score_dims if col in df.columns and pd.notna(n.get(col))]
        if len(radar_dims) >= 3:
            cats = [l for l, _ in radar_dims]
            vals = [float(n[c]) for _, c in radar_dims]
            fig_r = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill='toself',
                fillcolor='rgba(212,165,116,0.15)',
                line=dict(color='#d4a574', width=2),
                marker=dict(color='#d4a574', size=5),
            ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 1], tickfont=dict(size=8, color='#807c75'),
                                    gridcolor='rgba(232,230,225,0.08)', tickvals=[0.25,0.5,0.75,1.0]),
                    angularaxis=dict(tickfont=dict(size=10, color='#e8e6e1')),
                    bgcolor='rgba(0,0,0,0)',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=30, r=30, t=20, b=20),
                height=240,
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True, key="profile_radar")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Demographics — stat cards with vs-NYC indicator
        demo_keys = [
            ('Median income', 'median_income', '${:,.0f}'),
            ('Population', 'total_population', '{:,.0f}'),
            ('Median age', 'median_age', '{:.0f}'),
            ('College rate', 'college_rate', '{:.0%}'),
        ]
        if RC: demo_keys.append(('Median rent', RC, '${:,.0f}'))
        rows_present = [(l, k, f) for l, k, f in demo_keys
                        if k and k in df.columns and pd.notna(n.get(k))]
        if rows_present:
            st.markdown('<div class="profile-section-title">Demographics</div>', unsafe_allow_html=True)
            cards = ''
            for label, col, fmt in rows_present:
                val = n[col]
                city_med = df[col].median()
                delta = val - city_med
                if label in ('Median income', 'Median rent'):
                    vs_str = f'{"▲" if delta > 0 else "▼"} vs NYC median'
                    vs_color = '#8a9a5b' if delta > 0 else '#a87c5f'
                elif label == 'College rate':
                    vs_str = f'{"▲" if delta > 0 else "▼"} vs NYC avg'
                    vs_color = '#8a9a5b' if delta > 0 else '#a87c5f'
                else:
                    vs_str = ''
                    vs_color = '#807c75'
                cards += (f'<div class="demo-stat">'
                          f'<div class="demo-stat-num">{fmt.format(val)}</div>'
                          f'<div class="demo-stat-lbl">{label}</div>'
                          f'{"<div class=demo-stat-vs style=color:" + vs_color + ">" + vs_str + "</div>" if vs_str else ""}'
                          f'</div>')
            st.markdown(f'<div class="demo-stat-grid">{cards}</div>', unsafe_allow_html=True)

        # Vibrancy — emoji pills
        vibe_items = []
        if 'restaurant_count' in df.columns and pd.notna(n.get('restaurant_count')):
            vibe_items.append(('🍽️', int(n['restaurant_count']), 'Restaurants'))
        if 'cuisine_diversity' in df.columns and pd.notna(n.get('cuisine_diversity')):
            vibe_items.append(('🌮', int(n['cuisine_diversity']), 'Cuisines'))
        if 'osm_amenity_count' in df.columns and pd.notna(n.get('osm_amenity_count')):
            vibe_items.append(('🍺', int(n['osm_amenity_count']), 'Bars & Cafes'))
        if vibe_items:
            st.markdown('<div class="profile-section-title">Vibrancy</div>', unsafe_allow_html=True)
            pills = ''
            for icon, num, lbl in vibe_items:
                pills += (f'<div class="vibe-pill"><div class="vibe-icon">{icon}</div>'
                          f'<div><div class="vibe-num">{num:,}</div><div class="vibe-lbl">{lbl}</div></div></div>')
            st.markdown(f'<div class="vibe-grid">{pills}</div>', unsafe_allow_html=True)

        # Similar neighborhoods — cards with similarity bar
        if sim_mx is not None and sel_nta in sim_mx.index:
            st.markdown('<div class="profile-section-title">Similar neighborhoods</div>', unsafe_allow_html=True)
            similars = sim_mx[sel_nta].drop(sel_nta, errors='ignore').nlargest(5)
            sim_html = ''
            for snm, sc in similars.items():
                srow = df[df['nta_name'] == snm]
                if len(srow) == 0: continue
                srow = srow.iloc[0]
                bar_w = sc * 100
                sim_html += (
                    f'<div class="sim-card">'
                    f'<div><div class="sim-name">{snm}</div><div class="sim-boro">{srow["borough"]}</div></div>'
                    f'<div class="sim-score-wrap"><div class="sim-score-num">{sc:.3f}</div>'
                    f'<div class="sim-bar-track"><div class="sim-bar-fill" style="width:{bar_w:.0f}%"></div></div></div>'
                    f'</div>'
                )
            st.markdown(sim_html, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="profile-section-title">Sample listings</div>', unsafe_allow_html=True)
        rf_match = redfin[redfin['nta_name'] == sel_nta] if 'nta_name' in redfin.columns and len(redfin) > 0 else pd.DataFrame()
        rendered = 0
        if len(rf_match) > 0:
            for _, l in rf_match.head(4).iterrows():
                rent = l.get('price') or l.get('rent')
                if pd.isna(rent): continue
                bd = l.get('bedrooms')
                title = f"{int(bd)}BR" if pd.notna(bd) else "Unit"
                sqft = l.get('sqft')
                sqft_str = f" · {int(sqft)} sqft" if pd.notna(sqft) else ""
                st.markdown(f'<div class="profile-listing"><div class="profile-listing-title">{title}{sqft_str}</div><div class="profile-listing-price">${rent:,.0f}/mo</div><div class="profile-listing-src">Redfin</div></div>', unsafe_allow_html=True)
                rendered += 1
        if rendered < 3 and len(listings) > 0 and 'borough' in listings.columns:
            cl_match = listings[listings['borough'] == n['borough']]
            for _, l in cl_match.head(4 - rendered).iterrows():
                title = str(l.get('title', 'Listing'))[:50]
                if pd.isna(l.get('price')): continue
                st.markdown(f'<div class="profile-listing"><div class="profile-listing-title">{title}</div><div class="profile-listing-price">${l["price"]:,.0f}/mo</div><div class="profile-listing-src">Craigslist · {n["borough"]}</div></div>', unsafe_allow_html=True)
                rendered += 1
        if rendered == 0:
            st.caption("No listings found for this neighborhood.")

# ═══════ TAB 2: MAP ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-h">Livability across the city</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Color reflects your weighted score. Hover any neighborhood for details.</div>', unsafe_allow_html=True)
    if geo and 'nta_code' in df.columns:
        props = geo['features'][0]['properties']
        gk = next((k for k in props if 'nta' in k.lower() and ('code' in k.lower() or '2020' in k.lower())), None)
        if gk:
            sm = df.set_index('nta_code')['user_score'].to_dict()
            nm = df.set_index('nta_code')['nta_name'].to_dict()
            bm = df.set_index('nta_code')['borough'].to_dict()
            gd = pd.DataFrame([{'nta_code': f['properties'].get(gk, ''),
                                'name': nm.get(f['properties'].get(gk, ''), ''),
                                'borough': bm.get(f['properties'].get(gk, ''), ''),
                                'score': sm.get(f['properties'].get(gk, ''), 0)}
                               for f in geo['features']])
            fig = px.choropleth_mapbox(gd, geojson=geo, locations='nta_code',
                                       featureidkey=f'properties.{gk}', color='score',
                                       color_continuous_scale=[[0,'#d32f2f'],[0.1,'#ef5350'],[0.2,'#ff7043'],[0.3,'#ff9800'],[0.4,'#ffc107'],[0.5,'#ffeb3b'],[0.6,'#a5d6a7'],[0.7,'#66bb6a'],[0.85,'#43a047'],[1,'#2e7d32']], range_color=[0, 1],
                                       mapbox_style='carto-darkmatter', zoom=10,
                                       center={"lat": 40.7128, "lon": -73.95}, opacity=.75,
                                       hover_name='name',
                                       hover_data={'score': ':.3f', 'borough': True, 'nta_code': False})
            top5 = df.nlargest(5, 'user_score').dropna(subset=['centroid_lat', 'centroid_lon'])
            fig.add_trace(go.Scattermapbox(
                lat=top5['centroid_lat'], lon=top5['centroid_lon'],
                mode='markers',
                marker=dict(size=30, color='#4caf50'),
                hovertext=[f"#{i+1} {n} — {s:.3f}" for i, (n, s) in enumerate(zip(top5['nta_name'], top5['user_score']))],
                hoverinfo='text',
                showlegend=False))
            fig.add_trace(go.Scattermapbox(
                lat=top5['centroid_lat'].tolist(), lon=top5['centroid_lon'].tolist(),
                mode='text',
                text=[str(i+1) for i in range(len(top5))],
                textfont=dict(size=14, color='white'),
                hoverinfo='skip',
                showlegend=False))
            fig.add_trace(go.Scattermapbox(
                lat=top5['centroid_lat'].tolist(), lon=top5['centroid_lon'].tolist(),
                mode='text',
                text=top5['nta_name'].tolist(),
                textposition='top center',
                textfont=dict(size=12, color='white', family='Inter'),
                hoverinfo='skip',
                showlegend=False))
            fig.update_layout(margin=dict(r=0, t=0, l=0, b=0), height=620)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not match GeoJSON keys.")
    else:
        st.warning("Map data unavailable.")

# ═══════ TAB 3: AI ADVISOR (Groq — free) ════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-h">AI advisor</div>', unsafe_allow_html=True)

    groq_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not groq_key:
        groq_key = st.text_input("Groq API Key (free at console.groq.com)", type="password", key="groq_key_input")

    if "advisor_chat" not in st.session_state:
        st.session_state.advisor_chat = []

    has_msgs = len(st.session_state.advisor_chat) > 0
    msg_container = st.container(height=500) if has_msgs else None
    prompt = st.chat_input("Ask about NYC neighborhoods...")

    def _render_history(container):
        target = container if container else st
        for m in st.session_state.advisor_chat:
            if m["role"] == "user":
                with target.chat_message("user"):
                    st.markdown(m["content"])
            else:
                target.markdown(f'<div class="advisor-label">🤖 ADVISOR</div><div class="advisor-bubble">{m["content"]}</div>', unsafe_allow_html=True)

    if has_msgs:
        with msg_container:
            _render_history(None)

    if prompt and groq_key:
        st.session_state.advisor_chat.append({"role": "user", "content": prompt})
        if msg_container:
            with msg_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            msg_container = st.container(height=500)

        import requests as req

        top_rows = ft.head(20) if len(ft) > 0 else df.nlargest(20, 'user_score')
        cols_ctx = ['nta_name', 'borough', 'user_score', 'safety_score', 'noise_score', 'parks_score', TC]
        if RC and RC in top_rows.columns:
            cols_ctx.append(RC)
        ctx_data = top_rows[cols_ctx].round(3).to_dict(orient='records')

        vib_part = f", Vibrancy {vw}%" if has_vib and vw > 0 else ""
        sys_prompt = f"""You are a concise NYC neighborhood advisor. Answer only based on the data below.
User priorities: Safety {sw}%, Quiet {nw}%, Parks {pw}%, Transit {tw}%{vib_part}.
Budget: {"$"+str(mb)+"/mo" if mb else "no limit"}. Borough filter: {sb or "all boroughs"}.

Top 20 neighborhoods for this user (already ranked by their score):
{json.dumps(ctx_data, default=str)}

Be helpful and specific. Use neighborhood names from the data. Keep answers under 200 words."""

        messages = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state.advisor_chat]

        _target = msg_container if msg_container else st
        with _target:
          with st.spinner("Thinking..."):
                try:
                    raw = req.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json={"model": "llama-3.1-8b-instant", "messages": [{"role": "system", "content": sys_prompt}] + messages, "max_tokens": 512, "temperature": 0.5},
                        timeout=20
                    )
                    resp = raw.json()
                    if "choices" not in resp:
                        st.error(f"Groq API error: {resp.get('error', resp)}")
                    else:
                        reply = resp["choices"][0]["message"]["content"]
                        import re
                        html_reply = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', reply)
                        html_reply = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_reply)
                        html_reply = re.sub(r'^#{1,3}\s+(.+)$', r'<h4>\1</h4>', html_reply, flags=re.MULTILINE)
                        html_reply = re.sub(r'^[-•]\s+(.+)$', r'<li>\1</li>', html_reply, flags=re.MULTILINE)
                        html_reply = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', html_reply, flags=re.DOTALL)
                        html_reply = re.sub(r'\n\n+', '<br><br>', html_reply)
                        st.markdown(f'<div class="advisor-label">🤖 ADVISOR</div><div class="advisor-bubble">{html_reply}</div>', unsafe_allow_html=True)
                        st.session_state.advisor_chat.append({"role": "assistant", "content": reply})
                        st.rerun()
                except Exception as e:
                    st.error(f"Groq error: {e}")

    elif prompt:
        st.warning("Paste your free Groq API key above.")

# ═══════ TAB 4: COMMUTE ═══════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Commute Estimator")
    st.caption("🚇 Subway + walking estimates (walk to station ~7 min + ride + exit ~5 min)")

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
                    cdf = df[_cols].dropna().copy()
                    R = 6371
                    cdf['dist'] = cdf.apply(lambda r: R * 2 * np.arctan2(
                        np.sqrt(np.sin(np.radians(dlat - r['centroid_lat']) / 2) ** 2 +
                                np.cos(np.radians(r['centroid_lat'])) * np.cos(np.radians(dlat)) *
                                np.sin(np.radians(dlon - r['centroid_lon']) / 2) ** 2),
                        np.sqrt(1 - (np.sin(np.radians(dlat - r['centroid_lat']) / 2) ** 2 +
                                     np.cos(np.radians(r['centroid_lat'])) * np.cos(np.radians(dlat)) *
                                     np.sin(np.radians(dlon - r['centroid_lon']) / 2) ** 2))), axis=1)
                    # Transit estimate: walk to subway (7min) + subway ride + exit walk (5min)
                    # Subway avg speed 25 km/h with 1.4x route factor for non-straight lines
                    walk_only = (cdf['dist'] / 5 * 60)
                    subway_ride = 12 + (cdf['dist'] * 1.4 / 25 * 60)
                    cdf['est_min'] = np.minimum(walk_only, subway_ride).round(0).astype(int).clip(lower=3)

                    top_c = cdf.nlargest(20, 'user_score').copy()
                    top_c['commute_min'] = top_c['est_min']
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
                    for rk, (_, r) in enumerate(top_c.head(5).iterrows(), 1):
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
                    for i, (_, r) in enumerate(top_c.head(5).iterrows()):
                        cm = int(r['commute_min'])
                        fig_map.add_trace(go.Scattermapbox(
                            lat=[r['centroid_lat']], lon=[r['centroid_lon']],
                            mode='markers+text', text=[f"{cm}m"],
                            marker=dict(size=26, color='#d4a574', opacity=0.92),
                            textfont=dict(size=9, color='#1a1208', family='Arial Black'),
                            name=r['nta_name'],
                            hovertemplate=f"<b>{r['nta_name']}</b><br>#{i+1} — ~{cm} min transit<extra></extra>",
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

# ═══════ TAB 5: SIMILAR ═══════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-h">Find similar neighborhoods</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">"I like Williamsburg — what else would I like?" Cosine similarity on score vectors.</div>', unsafe_allow_html=True)

    sim_options = sorted(df['nta_name'].unique())
    top1 = ft.iloc[0]['nta_name'] if len(ft) > 0 else sim_options[0]
    sim_default = sim_options.index(top1) if top1 in sim_options else 0
    _sim_key = f"sim_ref_{sw}_{nw}_{pw}_{tw}_{sb}_{mb}"
    ref = st.selectbox("Reference neighborhood", sim_options, index=sim_default, key=_sim_key)
    nsim = 10

    if sim_mx is not None and ref in sim_mx.index:
        scores = sim_mx[ref].drop(ref, errors='ignore').nlargest(nsim)
    else:
        feats = ['safety_score', 'noise_score', 'parks_score', TC]
        if 'rent_score' in df.columns:
            feats.append('rent_score')
        X = df[feats].fillna(0).values
        sm = cosine_similarity(X)
        idx = df.index[df['nta_name'] == ref]
        if len(idx) > 0:
            local = df.index.get_loc(idx[0])
            scores = pd.Series(sm[local], index=df['nta_name']).drop(ref, errors='ignore').nlargest(nsim)
        else:
            scores = pd.Series(dtype=float)

    if len(scores) > 0:
        sres = df[df['nta_name'].isin(scores.index)].copy()
        sres['similarity'] = sres['nta_name'].map(scores.to_dict())
        sres = sres.sort_values('similarity', ascending=False)

        dims = ['Safety', 'Noise', 'Parks', 'Transit']
        dcols = ['safety_score', 'noise_score', 'parks_score', TC]
        ref_row = df[df['nta_name'] == ref].iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[ref_row[c] for c in dcols] + [ref_row[dcols[0]]],
                                       theta=dims + [dims[0]], fill='toself', name=ref,
                                       line=dict(color='#ef4444', width=3), opacity=.8))
        for i, (_, row) in enumerate(sres.head(5).iterrows()):
            fig.add_trace(go.Scatterpolar(r=[row[c] for c in dcols] + [row[dcols[0]]],
                                           theta=dims + [dims[0]], fill='toself',
                                           name=f"{row['nta_name']} ({row['similarity']:.3f})", opacity=.4))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=450, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        scols = ['nta_name', 'borough', 'similarity', 'user_score']
        if RC:
            scols.append(RC)
        if 'cluster_label' in sres.columns:
            scols.append('cluster_label')
        st.dataframe(sres[[c for c in scols if c in sres.columns]].round(3),
                     hide_index=True, use_container_width=True)


# ═══════ TAB 6: COMPARE ══════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-h">Side-by-side comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pick 2 or 3 neighborhoods to compare directly. Best value per row in green, worst in red.</div>', unsafe_allow_html=True)

    selected = st.multiselect("Pick up to 3 neighborhoods",
                               sorted(df['nta_name'].unique()),
                               max_selections=3, default=[])

    if len(selected) >= 2:
        comp_df = df[df['nta_name'].isin(selected)].copy()
        comp_df = comp_df.set_index('nta_name').loc[selected].reset_index()

        # Radar chart overlay
        radar_dims = ['safety_score', 'noise_score', 'parks_score', TC]
        if 'rent_score' in comp_df.columns: radar_dims.append('rent_score')
        if 'vibrancy_score' in comp_df.columns: radar_dims.append('vibrancy_score')
        radar_labels = {'safety_score': 'Safety', 'noise_score': 'Quiet',
                         'parks_score': 'Parks', TC: 'Transit',
                         'rent_score': 'Affordability', 'vibrancy_score': 'Vibrancy'}
        cats = [radar_labels.get(c, c) for c in radar_dims]

        fig = go.Figure()
        radar_colors = ['#d4a574', '#8a9a5b', '#5b7b8a']
        for i, (_, r) in enumerate(comp_df.iterrows()):
            vals = [r[c] for c in radar_dims] + [r[radar_dims[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]], fill='toself',
                name=r['nta_name'],
                line=dict(color=radar_colors[i % 3], width=2),
                opacity=0.45
            ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                           height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        rows = [
            ('Borough', 'borough', '{}'),
            ('Composite score', 'composite_score', '{:.3f}'),
            ('Safety', 'safety_score', '{:.2f}'),
            ('Quiet', 'noise_score', '{:.2f}'),
            ('Parks', 'parks_score', '{:.2f}'),
            ('Transit', TC, '{:.2f}'),
        ]
        if 'vibrancy_score' in comp_df.columns:
            rows.append(('Vibrancy', 'vibrancy_score', '{:.2f}'))
        if RC:
            rows.append(('Median rent', RC, '${:,.0f}'))
        if 'cluster_label' in comp_df.columns:
            rows.append(('Archetype', 'cluster_label', '{}'))
        for c, lbl, fmt in [('median_income', 'Median income', '${:,.0f}'),
                             ('total_population', 'Population', '{:,.0f}'),
                             ('median_age', 'Median age', '{:.0f}'),
                             ('college_rate', 'College rate', '{:.0%}')]:
            if c in comp_df.columns:
                rows.append((lbl, c, fmt))

        # Build HTML table
        tbl = ('<table style="font-family:JetBrains Mono,monospace;font-size:0.8rem;'
               'width:100%;border-collapse:collapse;margin-top:1.5rem">')
        tbl += ('<thead><tr><th style="text-align:left;padding:0.7rem 0.5rem;'
                'border-bottom:1px solid rgba(232,230,225,0.16);color:#807c75;'
                'text-transform:uppercase;letter-spacing:0.08em;font-size:0.66rem;'
                'font-weight:500">Metric</th>')
        for nta in selected:
            tbl += (f'<th style="text-align:right;padding:0.7rem 0.5rem;'
                    f'border-bottom:1px solid rgba(232,230,225,0.16);'
                    f'font-family:Fraunces,serif;font-size:0.95rem;color:#e8e6e1;'
                    f'font-weight:500">{nta}</th>')
        tbl += '</tr></thead><tbody>'

        for label, col, fmt in rows:
            if col not in comp_df.columns: continue
            tbl += (f'<tr><td style="padding:0.6rem 0.5rem;'
                    f'border-bottom:1px solid rgba(232,230,225,0.06);'
                    f'color:#807c75;text-transform:uppercase;letter-spacing:0.08em;'
                    f'font-size:0.66rem">{label}</td>')
            vals = [comp_df.iloc[i][col] for i in range(len(selected))]

            # Find best/worst for numeric cols
            best_idx, worst_idx = -1, -1
            if label not in ('Borough', 'Archetype'):
                try:
                    arr = np.array([float(v) if pd.notna(v) else np.nan for v in vals])
                    if not np.all(np.isnan(arr)):
                        higher_better = col != RC  # for rent, lower is better
                        best_idx = int(np.nanargmax(arr) if higher_better else np.nanargmin(arr))
                        worst_idx = int(np.nanargmin(arr) if higher_better else np.nanargmax(arr))
                except (ValueError, TypeError):
                    pass

            for i, v in enumerate(vals):
                if pd.isna(v):
                    s, style = '—', ''
                else:
                    try:
                        s = fmt.format(v)
                    except Exception:
                        s = str(v)
                    if i == best_idx and best_idx != worst_idx and len(selected) > 1:
                        style = 'color:#8a9a5b;font-weight:600'
                    elif i == worst_idx and best_idx != worst_idx and len(selected) > 1:
                        style = 'color:#a87c5f'
                    else:
                        style = 'color:#e8e6e1'
                tbl += (f'<td style="padding:0.6rem 0.5rem;'
                        f'border-bottom:1px solid rgba(232,230,225,0.06);'
                        f'text-align:right;{style}">{s}</td>')
            tbl += '</tr>'
        tbl += '</tbody></table>'
        st.markdown(tbl, unsafe_allow_html=True)


    elif len(selected) == 1:
        st.info("Pick one more neighborhood to compare.")
    else:
        st.caption("Select 2 or 3 neighborhoods above to start comparing.")

# ═══════ TAB 7: CLUSTERS ══════════════════════════════════════════════════════
with tabs[7]:
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
                card_class = "cluster-card best-match" if ci == best_ci else "cluster-card"
                bars_html = ""
                for di, (dname, dcolor) in enumerate(zip(dim_names, dim_colors)):
                    pct = min(c[di] * 100, 100)
                    bars_html += f"""<div class="cc-bar-wrap">
<span class="cc-bar-label">{dname}</span>
<div class="cc-bar-track"><div class="cc-bar-fill" style="width:{pct:.0f}%;background:{dcolor}"></div></div>
<span class="cc-bar-val">{pct:.0f}%</span>
</div>"""
                st.markdown(f"""<div class="{card_class}">
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

# ═══════ TAB 7: LISTINGS ══════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="section-h">Live rental listings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Combined Craigslist + Redfin data, filtered to your top-ranked boroughs.</div>', unsafe_allow_html=True)

    all_ls = listings.copy()
    if len(redfin) > 0:
        rf_show = redfin.copy()
        if 'title' not in rf_show.columns:
            rf_show['title'] = rf_show.apply(
                lambda r: f"{int(r['bedrooms'])}BR" if pd.notna(r.get('bedrooms')) else "Studio", axis=1)
        if 'borough' not in rf_show.columns and 'nta_name' in rf_show.columns:
            # Map NTA -> borough from master df
            nta_boro = df.set_index('nta_name')['borough'].to_dict()
            rf_show['borough'] = rf_show['nta_name'].map(nta_boro).fillna('NYC')
        elif 'borough' not in rf_show.columns:
            rf_show['borough'] = 'NYC'
        rf_show['source'] = 'Redfin'
        if len(all_ls) > 0:
            all_ls['source'] = 'Craigslist'
        all_ls = pd.concat([all_ls, rf_show], ignore_index=True)
    elif len(all_ls) > 0:
        all_ls['source'] = 'Craigslist'

    if len(all_ls) > 0 and len(top) > 0:
        bl = top['borough'].unique().tolist()
        if 'borough' in all_ls.columns:
            ls = all_ls[all_ls['borough'].isin(bl)].copy()
        else:
            ls = all_ls.copy()
        if mb and 'price' in ls.columns:
            ls = ls[ls['price'] <= mb]
        ls = ls.sort_values('price') if 'price' in ls.columns else ls
        st.caption(f"{len(ls)} listings in {', '.join(bl)}" + (f" under ${mb:,}/mo" if mb else ""))

        if 'sentiment_polarity' in ls.columns:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(ls, x='sentiment_polarity', nbins=20, color_discrete_sequence=['#3b82f6'])
                fig.update_layout(height=260, title='Listing Sentiment', margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(ls, x='sentiment_polarity', y='price', color='borough', opacity=.7,
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=260, title='Sentiment vs Price', margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        show = [c for c in ['title', 'price', 'borough', 'bedrooms', 'sqft', 'no_fee', 'sentiment_label', 'source'] if c in ls.columns]
        st.dataframe(ls[show].head(50), hide_index=True, use_container_width=True)
    else:
        st.info("No listing data.")

# ═══════ TAB 9: DEMOGRAPHICS ══════════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="section-h">Who lives where</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Census ACS data joined to NTAs via the official NYC Planning crosswalk.</div>', unsafe_allow_html=True)
    demo_cols = [c for c in ['median_income', 'total_population', 'median_age', 'college_rate',
                              'census_median_rent'] if c in df.columns]
    if demo_cols:
        metric = st.selectbox("Metric", demo_cols,
                               format_func=lambda x: x.replace('_', ' ').replace('census ', '').title())

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(df, x=metric, nbins=30, color='borough', barmode='overlay', opacity=.7,
                                color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=380, title=f'{metric.replace("_", " ").title()} Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ba = df.groupby('borough')[metric].median().sort_values(ascending=False)
            fig = px.bar(x=ba.values, y=ba.index, orientation='h', color_discrete_sequence=['#3b82f6'])
            fig.update_layout(height=380, title='Median by Borough', yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(df, x=metric, y='user_score', color='borough', hover_name='nta_name',
                          opacity=.7, color_discrete_sequence=px.colors.qualitative.Set2,
                          labels={metric: metric.replace('_', ' ').title(), 'user_score': 'Your Score'})
        fig.update_layout(height=400, title=f'{metric.replace("_", " ").title()} vs Livability')
        st.plotly_chart(fig, use_container_width=True)

        num = ['user_score', 'safety_score', 'noise_score', 'parks_score', TC] + demo_cols
        if 'vibrancy_score' in df.columns:
            num.insert(5, 'vibrancy_score')
        corr = df[num].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto')
        fig.update_layout(height=500, title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run Census cells in the notebook.")

# ═══════ TAB 10: DATA ═════════════════════════════════════════════════════════
with tabs[10]:
    st.markdown('<div class="section-h">The full dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All 197 neighborhoods, all 35 columns. Filter, sort, download.</div>', unsafe_allow_html=True)
    defaults = ['nta_name', 'borough', 'user_score', 'safety_score', 'noise_score', 'parks_score', TC]
    if 'vibrancy_score' in df.columns:
        defaults.append('vibrancy_score')
    if 'rent_score' in df.columns:
        defaults.append('rent_score')
    if RC:
        defaults.append(RC)
    if 'cluster_label' in df.columns:
        defaults.append('cluster_label')
    for d in ['restaurant_count', 'osm_amenity_count', 'median_income', 'total_population', 'median_age', 'college_rate']:
        if d in df.columns:
            defaults.append(d)

    sel = st.multiselect("Columns", ft.columns.tolist(),
                          default=[c for c in defaults if c in ft.columns])
    if sel:
        st.dataframe(ft[sel].sort_values('user_score', ascending=False).round(3),
                     height=500, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", ft[sel].to_csv(index=False), "livability.csv", "text/csv")

    st.markdown("#### Score Distributions")
    box = ['safety_score', 'noise_score', 'parks_score', TC, 'user_score']
    if 'rent_score' in df.columns:
        box.append('rent_score')
    fig = go.Figure()
    for d in box:
        fig.add_trace(go.Box(y=df[d], name=d.replace('_score', '').replace('_', ' ').title(), boxpoints='outliers'))
    fig.update_layout(height=380, yaxis_title='Score (0–1)')
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
_db = "Postgres" if USE_POSTGRES else "SQLite"
st.markdown(f"""<div class="footer">
<span class="accent">Polis Technologies</span>  &middot;  Section 01  &middot;  {len(df)} NTAs  &middot;  9 data sources  &middot;  ML-powered  &middot;  MCP tool calling  &middot;  {_db}<br>
<span style="opacity:0.6">NYC Open Data  &middot;  Census ACS  &middot;  Citi Bike  &middot;  MTA  &middot;  Redfin  &middot;  Craigslist</span>
</div>""", unsafe_allow_html=True)
