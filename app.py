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
                "Compare", "Clusters", "Listings", "Trends", "Demographics", "Data"])

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
    st.markdown('<div class="section-sub">Pick a neighborhood to see its full breakdown — scores, demographics, listings, similar areas.</div>', unsafe_allow_html=True)

    sel_nta = st.selectbox("Neighborhood", sorted(df['nta_name'].unique()), key='profile_nta')
    n = df[df['nta_name'] == sel_nta].iloc[0]

    cluster_str = ""
    if 'cluster_label' in n.index and pd.notna(n.get('cluster_label')):
        cluster_str = f' &nbsp;·&nbsp; {n["cluster_label"]}'

    st.markdown(
        f'<div style="font-family:Fraunces,serif;font-size:2rem;font-weight:400;'
        f'letter-spacing:-0.025em;color:#e8e6e1;margin:1rem 0 0.25rem">{n["nta_name"]}</div>'
        f'<div style="font-family:JetBrains Mono,monospace;color:#807c75;'
        f'text-transform:uppercase;letter-spacing:0.1em;font-size:0.72rem;'
        f'margin-bottom:1.5rem">{n["borough"]}{cluster_str}</div>',
        unsafe_allow_html=True
    )

    # Score cards with percentile rank
    score_dims = [('Safety', 'safety_score'), ('Quiet', 'noise_score'),
                  ('Parks', 'parks_score'), ('Transit', TC)]
    if 'rent_score' in df.columns: score_dims.append(('Affordability', 'rent_score'))
    if 'vibrancy_score' in df.columns: score_dims.append(('Vibrancy', 'vibrancy_score'))

    cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(170px, 1fr));gap:0.7rem;margin-bottom:2rem">'
    for label, col in score_dims:
        if col not in df.columns: continue
        val = n[col]
        if pd.isna(val): continue
        pctile = int((df[col].fillna(0) < val).sum() / len(df) * 100)
        pc = "#8a9a5b" if pctile >= 70 else "#c9a961" if pctile >= 40 else "#a87c5f"
        bar_w = val * 100
        cards_html += (
            f'<div style="padding:1rem 1.1rem;background:#141414;border:1px solid rgba(232,230,225,0.08);'
            f'border-left:2px solid #d4a574">'
            f'<div style="font-family:Fraunces,serif;font-size:1.55rem;color:#e8e6e1;line-height:1;font-weight:400">{val:.2f}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#807c75;'
            f'text-transform:uppercase;letter-spacing:0.12em;margin-top:0.4rem">{label}</div>'
            f'<div style="height:2px;background:rgba(232,230,225,0.08);margin-top:0.6rem;position:relative">'
            f'<div style="position:absolute;height:100%;width:{bar_w:.1f}%;background:#d4a574"></div></div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:{pc};'
            f'margin-top:0.4rem;letter-spacing:0.06em">{pctile}th percentile</div>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Demographics
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
            st.markdown(
                '<div style="font-family:Fraunces,serif;font-size:1.05rem;font-weight:500;'
                'margin:1.5rem 0 0.6rem;color:#e8e6e1">Demographics</div>',
                unsafe_allow_html=True
            )
            tbl = '<table style="font-family:JetBrains Mono,monospace;font-size:0.78rem;width:100%;border-collapse:collapse">'
            for label, col, fmt in rows_present:
                tbl += (
                    f'<tr style="border-bottom:1px solid rgba(232,230,225,0.06)">'
                    f'<td style="padding:0.55rem 0;color:#807c75;text-transform:uppercase;'
                    f'letter-spacing:0.08em;font-size:0.68rem">{label}</td>'
                    f'<td style="padding:0.55rem 0;text-align:right;color:#e8e6e1">{fmt.format(n[col])}</td></tr>'
                )
            tbl += '</table>'
            st.markdown(tbl, unsafe_allow_html=True)

        # Food / amenity scene if present
        food_lines = []
        if 'restaurant_count' in df.columns and pd.notna(n.get('restaurant_count')):
            food_lines.append(f"<strong style='color:#e8e6e1'>{int(n['restaurant_count'])}</strong> restaurants")
        if 'cuisine_diversity' in df.columns and pd.notna(n.get('cuisine_diversity')):
            food_lines.append(f"<strong style='color:#e8e6e1'>{int(n['cuisine_diversity'])}</strong> distinct cuisines")
        if 'osm_amenity_count' in df.columns and pd.notna(n.get('osm_amenity_count')):
            food_lines.append(f"<strong style='color:#e8e6e1'>{int(n['osm_amenity_count'])}</strong> bars/cafes/shops")
        if food_lines:
            st.markdown(
                '<div style="font-family:Fraunces,serif;font-size:1.05rem;font-weight:500;'
                'margin:1.5rem 0 0.6rem;color:#e8e6e1">Vibrancy</div>',
                unsafe_allow_html=True
            )
            blk = ''.join(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;'
                f'color:#807c75;padding:0.35rem 0">{l}</div>' for l in food_lines
            )
            st.markdown(blk, unsafe_allow_html=True)

        # Similar neighborhoods
        if sim_mx is not None and sel_nta in sim_mx.index:
            st.markdown(
                '<div style="font-family:Fraunces,serif;font-size:1.05rem;font-weight:500;'
                'margin:1.5rem 0 0.6rem;color:#e8e6e1">Similar neighborhoods</div>',
                unsafe_allow_html=True
            )
            similars = sim_mx[sel_nta].drop(sel_nta, errors='ignore').nlargest(5)
            sim_html = ''
            for nm, sc in similars.items():
                row = df[df['nta_name'] == nm]
                if len(row) == 0: continue
                row = row.iloc[0]
                sim_html += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:0.7rem 0;border-bottom:1px solid rgba(232,230,225,0.06)">'
                    f'<div><span style="font-family:Fraunces,serif;font-size:0.95rem;color:#e8e6e1">{nm}</span>'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
                    f'color:#807c75;margin-left:0.6rem;text-transform:uppercase;letter-spacing:0.08em">'
                    f'{row["borough"]}</span></div>'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#d4a574">'
                    f'{sc:.3f}</span></div>'
                )
            st.markdown(sim_html, unsafe_allow_html=True)

    with col_r:
        st.markdown(
            '<div style="font-family:Fraunces,serif;font-size:1.05rem;font-weight:500;'
            'margin:1.5rem 0 0.6rem;color:#e8e6e1">Sample listings</div>',
            unsafe_allow_html=True
        )
        # Try Redfin first (precise NTA match)
        rf_match = redfin[redfin['nta_name'] == sel_nta] if 'nta_name' in redfin.columns and len(redfin) > 0 else pd.DataFrame()
        rendered = 0
        if len(rf_match) > 0:
            for _, l in rf_match.head(5).iterrows():
                rent = l.get('price') or l.get('rent')
                if pd.isna(rent): continue
                bd = l.get('bedrooms')
                title = f"{int(bd)}BR" if pd.notna(bd) else "Unit"
                sqft = l.get('sqft')
                sqft_str = f" · {int(sqft)} sqft" if pd.notna(sqft) else ""
                st.markdown(
                    f'<div style="padding:0.75rem 0;border-bottom:1px solid rgba(232,230,225,0.06)">'
                    f'<div style="font-family:Fraunces,serif;color:#e8e6e1;font-size:0.95rem">{title}{sqft_str}</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#d4a574;margin-top:0.2rem">'
                    f'${rent:,.0f}/mo</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#807c75;'
                    f'margin-top:0.15rem;letter-spacing:0.06em;text-transform:uppercase">Redfin</div></div>',
                    unsafe_allow_html=True
                )
                rendered += 1
        # Fallback to borough-level Craigslist
        if rendered < 3 and len(listings) > 0 and 'borough' in listings.columns:
            cl_match = listings[listings['borough'] == n['borough']]
            for _, l in cl_match.head(5 - rendered).iterrows():
                title = str(l.get('title', 'Listing'))[:55]
                if pd.isna(l.get('price')): continue
                st.markdown(
                    f'<div style="padding:0.75rem 0;border-bottom:1px solid rgba(232,230,225,0.06)">'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#e8e6e1;font-size:0.78rem">{title}</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#d4a574;margin-top:0.2rem">'
                    f'${l["price"]:,.0f}/mo</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#807c75;'
                    f'margin-top:0.15rem;letter-spacing:0.06em;text-transform:uppercase">'
                    f'Craigslist · {n["borough"]}</div></div>',
                    unsafe_allow_html=True
                )
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
                                       color_continuous_scale=[[0,'#3a2818'],[0.3,'#7a5a3a'],[0.6,'#b8956a'],[1,'#d4a574']], range_color=[0, 1],
                                       mapbox_style='carto-darkmatter', zoom=10,
                                       center={"lat": 40.7128, "lon": -73.95}, opacity=.75,
                                       hover_name='name',
                                       hover_data={'score': ':.3f', 'borough': True, 'nta_code': False})
            fig.update_layout(margin=dict(r=0, t=0, l=0, b=0), height=620)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not match GeoJSON keys.")
    else:
        st.warning("Map data unavailable.")

# ═══════ TAB 3: AI CHAT (MCP Tool Calling) ════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-h">AI advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Powered by Claude via MCP tool calling. Queries the database, finds similar neighborhoods, searches listings, estimates commutes in real time.</div>', unsafe_allow_html=True)

    # Prefer secrets, fall back to user input
    secret_key = None
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            secret_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    if secret_key:
        api_key = secret_key
        st.caption("Using configured API key")
    else:
        api_key = st.text_input("Anthropic API Key", type="password",
                                 help="Free at console.anthropic.com. Never stored.")

    # Tool implementations
    def _tool_search(min_score=0.0, max_rent=99999, borough=None, limit=10, sort_by='composite_score'):
        r = df.copy()
        r = r[r['composite_score'] >= min_score]
        if RC:
            r = r[r[RC].fillna(99999) <= max_rent]
        if borough:
            r = r[r['borough'].str.lower() == borough.lower()]
        if sort_by in r.columns:
            r = r.sort_values(sort_by, ascending=False)
        cols = ['nta_name', 'borough', 'composite_score', 'safety_score', 'noise_score', 'parks_score', TC]
        if 'cluster_label' in r.columns:
            cols.append('cluster_label')
        if RC:
            cols.append(RC)
        return r[cols].head(limit).to_dict(orient='records')

    def _tool_details(neighborhood_name):
        m = df[df['nta_name'].str.lower().str.contains(neighborhood_name.lower())]
        if len(m) == 0:
            return {"error": f"'{neighborhood_name}' not found"}
        r = m.iloc[0]
        return {c: (round(float(r[c]), 3) if isinstance(r[c], (float, np.floating)) else
                     int(r[c]) if isinstance(r[c], (int, np.integer)) else str(r[c]))
                for c in r.index if pd.notna(r[c]) and c not in ['geometry', 'centroid_lat', 'centroid_lon']}

    def _tool_similar(neighborhood_name, n=5):
        if sim_mx is None:
            return {"error": "Similarity matrix not loaded"}
        nm = neighborhood_name
        if nm not in sim_mx.index:
            matches = [x for x in sim_mx.index if neighborhood_name.lower() in x.lower()]
            if matches:
                nm = matches[0]
            else:
                return {"error": f"'{neighborhood_name}' not found"}
        s = sim_mx[nm].drop(nm, errors='ignore').nlargest(n)
        return [{"nta_name": name, "similarity": round(sc, 4),
                 "borough": df.loc[df['nta_name'] == name, 'borough'].values[0] if len(df[df['nta_name'] == name]) > 0 else ''}
                for name, sc in s.items()]

    def _tool_listings(borough=None, max_price=None, min_bedrooms=None, limit=10):
        sql = "SELECT * FROM rent_listings WHERE 1=1"
        if borough:
            sql += f" AND LOWER(borough) = '{borough.lower()}'"
        if max_price:
            sql += f" AND price <= {max_price}"
        if min_bedrooms is not None:
            sql += f" AND bedrooms >= {min_bedrooms}"
        sql += f" ORDER BY price ASC LIMIT {limit}"
        ls = db_query(sql)
        if len(ls) == 0:
            return {"error": "No listings found"}
        cols = [c for c in ['title', 'price', 'borough', 'bedrooms', 'sqft', 'no_fee', 'sentiment_label'] if c in ls.columns]
        return ls[cols].to_dict(orient='records')

    def _tool_commute(neighborhood_name, destination_lat, destination_lon):
        m = df[df['nta_name'].str.lower().str.contains(neighborhood_name.lower())]
        if len(m) == 0:
            return {"error": f"'{neighborhood_name}' not found"}
        r = m.iloc[0]
        if pd.isna(r.get('centroid_lat')):
            return {"error": "No centroid data"}
        R = 6371
        dlat = np.radians(destination_lat - r['centroid_lat'])
        dlon = np.radians(destination_lon - r['centroid_lon'])
        a = (np.sin(dlat / 2) ** 2 + np.cos(np.radians(r['centroid_lat'])) *
             np.cos(np.radians(destination_lat)) * np.sin(dlon / 2) ** 2)
        dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return {"neighborhood": r['nta_name'], "distance_km": round(dist, 1),
                "est_minutes": round(dist / 25 * 60)}

    MCP_TOOLS = [
        {"name": "search_neighborhoods", "description": "Search neighborhoods by score, rent, borough.",
         "input_schema": {"type": "object", "properties": {
             "min_score": {"type": "number"}, "max_rent": {"type": "number"},
             "borough": {"type": "string"}, "limit": {"type": "integer"}, "sort_by": {"type": "string"}}}},
        {"name": "get_neighborhood_details", "description": "Full profile for a neighborhood.",
         "input_schema": {"type": "object", "properties": {"neighborhood_name": {"type": "string"}}, "required": ["neighborhood_name"]}},
        {"name": "find_similar", "description": "Find similar neighborhoods via cosine similarity.",
         "input_schema": {"type": "object", "properties": {"neighborhood_name": {"type": "string"}, "n": {"type": "integer"}}, "required": ["neighborhood_name"]}},
        {"name": "search_listings", "description": "Search rental listings by borough, price, bedrooms.",
         "input_schema": {"type": "object", "properties": {"borough": {"type": "string"}, "max_price": {"type": "number"}, "min_bedrooms": {"type": "integer"}, "limit": {"type": "integer"}}}},
        {"name": "estimate_commute", "description": "Estimate commute from neighborhood to a lat/lon.",
         "input_schema": {"type": "object", "properties": {"neighborhood_name": {"type": "string"}, "destination_lat": {"type": "number"}, "destination_lon": {"type": "number"}}, "required": ["neighborhood_name", "destination_lat", "destination_lon"]}},
    ]
    DISPATCH = {"search_neighborhoods": _tool_search, "get_neighborhood_details": _tool_details,
                "find_similar": _tool_similar, "search_listings": _tool_listings, "estimate_commute": _tool_commute}

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "msgs_raw" not in st.session_state:
        st.session_state.msgs_raw = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about NYC neighborhoods...")
    if prompt and api_key:
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.session_state.msgs_raw.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        import requests as req

        vib_part = f", Vibrancy {vw}%" if has_vib and vw > 0 else ""
        sys_prompt = f"""You are an NYC neighborhood advisor. You have 5 tools to query a database of 197 scored neighborhoods.
ALWAYS use tools to answer — never guess. User priorities: Safety {sw}%, Noise {nw}%, Parks {pw}%, Transit {tw}%{vib_part}.
Budget: {"$" + str(mb) + "/mo" if mb else "None"}. Borough: {sb or "All"}. Be concise and helpful."""

        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = req.post("https://api.anthropic.com/v1/messages", headers=headers,
                                    json={"model": "claude-sonnet-4-20250514", "max_tokens": 2048, "system": sys_prompt,
                                          "tools": MCP_TOOLS, "messages": st.session_state.msgs_raw}, timeout=30).json()

                    iterations = 0
                    tool_log = []
                    while resp.get('stop_reason') == 'tool_use' and iterations < 5:
                        iterations += 1
                        content = resp['content']
                        results = []
                        for blk in content:
                            if blk['type'] == 'tool_use':
                                fn = DISPATCH.get(blk['name'])
                                tool_log.append(f"`{blk['name']}({json.dumps(blk['input'], default=str)[:80]})`")
                                try:
                                    res = fn(**blk['input']) if fn else {"error": "unknown tool"}
                                except Exception as e:
                                    res = {"error": str(e)}
                                results.append({"type": "tool_result", "tool_use_id": blk['id'],
                                                "content": json.dumps(res, default=str)})

                        st.session_state.msgs_raw.append({"role": "assistant", "content": content})
                        st.session_state.msgs_raw.append({"role": "user", "content": results})

                        resp = req.post("https://api.anthropic.com/v1/messages", headers=headers,
                                        json={"model": "claude-sonnet-4-20250514", "max_tokens": 2048, "system": sys_prompt,
                                              "tools": MCP_TOOLS, "messages": st.session_state.msgs_raw}, timeout=30).json()

                    reply = "".join(b.get('text', '') for b in resp.get('content', []) if b.get('type') == 'text')

                    if tool_log:
                        with st.expander(f"{len(tool_log)} tool call(s)", expanded=False):
                            for tl in tool_log:
                                st.markdown(tl)

                    st.markdown(reply)
                    st.session_state.chat.append({"role": "assistant", "content": reply})
                    st.session_state.msgs_raw.append({"role": "assistant", "content": resp.get('content', [])})

                except Exception as e:
                    st.error(f"API error: {e}")
    elif prompt:
        st.warning("Enter your Anthropic API key above.")

# ═══════ TAB 4: COMMUTE ═══════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-h">How long to get to work?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter your destination and we will estimate driving time from each top-ranked neighborhood.</div>', unsafe_allow_html=True)

    addr = st.text_input("Work address", placeholder="e.g. 1 World Trade Center, New York")
    if addr and 'centroid_lat' in df.columns:
        import requests as req
        with st.spinner("Geocoding..."):
            try:
                g = req.get("https://nominatim.openstreetmap.org/search",
                            params={"q": addr, "format": "json", "limit": 1},
                            headers={"User-Agent": "NYCLivability/1.0"}, timeout=10).json()
                if g:
                    dlat, dlon = float(g[0]['lat']), float(g[0]['lon'])
                    st.success(f"Found: {g[0].get('display_name', '')[:80]}")

                    cdf = df[['nta_name', 'borough', 'user_score', 'centroid_lat', 'centroid_lon']].dropna().copy()
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
                        except:
                            osrm_t.append(row['est_min'])
                        time.sleep(0.1)
                    top_c['commute_min'] = osrm_t

                    show = ['nta_name', 'borough', 'user_score', 'commute_min']
                    if RC:
                        show.append(RC)
                    disp = top_c[show].sort_values('user_score', ascending=False)
                    disp.columns = [c.replace('_', ' ').title() for c in disp.columns]
                    st.dataframe(disp, hide_index=True, use_container_width=True)

                    fig = px.scatter(top_c, x='commute_min', y='user_score', hover_name='nta_name',
                                     color='borough', size='user_score', size_max=15,
                                     labels={'commute_min': 'Commute (min)', 'user_score': 'Your Score'})
                    fig.update_layout(height=400, title='Score vs Commute')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Address not found.")
            except Exception as e:
                st.error(f"Geocoding failed: {e}")

# ═══════ TAB 5: SIMILAR ═══════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-h">Find similar neighborhoods</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">"I like Williamsburg — what else would I like?" Cosine similarity on score vectors.</div>', unsafe_allow_html=True)

    ref = st.selectbox("Reference neighborhood", sorted(df['nta_name'].unique()), key="sim_ref")
    nsim = st.slider("Show top", 3, 15, 7, key="sim_n")

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
    st.markdown('<div class="section-h">Neighborhood archetypes</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">K-Means clustering revealed two distinct profiles across NYC.</div>', unsafe_allow_html=True)
    if 'cluster_label' in df.columns:
        agg = df.groupby('cluster_label').agg(
            n=('nta_name', 'count'), safety=('safety_score', 'mean'),
            noise=('noise_score', 'mean'), parks=('parks_score', 'mean'),
            transit=(TC, 'mean'), score=('user_score', 'mean')).round(3)
        if RC:
            agg['median_rent'] = df.groupby('cluster_label')[RC].median().round(0)
        if 'median_income' in df.columns:
            agg['median_income'] = df.groupby('cluster_label')['median_income'].median().round(0)
        agg = agg.sort_values('score', ascending=False)
        st.dataframe(agg, use_container_width=True)

        cats = ['Safety', 'Noise', 'Parks', 'Transit']
        fig = go.Figure()
        for lbl, row in agg.iterrows():
            v = [row['safety'], row['noise'], row['parks'], row['transit'], row['safety']]
            fig.add_trace(go.Scatterpolar(r=v, theta=cats + [cats[0]], fill='toself',
                                           name=f"{lbl} ({int(row['n'])})", opacity=.6))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=450)
        st.plotly_chart(fig, use_container_width=True)

        bd = df.groupby(['cluster_label', 'borough']).size().reset_index(name='count')
        fig2 = px.bar(bd, x='cluster_label', y='count', color='borough', barmode='stack',
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=350, xaxis_tickangle=-25)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run K-Means in the notebook first.")

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

# ═══════ TAB 8: TRENDS ════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="section-h">When New Yorkers complain</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">311 complaint patterns across time and across the city.</div>', unsafe_allow_html=True)
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
                                     line=dict(color='red', width=2)))
            fig.update_layout(height=300, title='Daily Volume', margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            dow = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ts['d'] = ts['date'].dt.day_name().str[:3]
            dc = ts['d'].value_counts().reindex(dow).fillna(0)
            fig = px.bar(x=dc.index, y=dc.values, color_discrete_sequence=['#10b981'],
                          labels={'x': 'Day', 'y': 'Complaints'})
            fig.update_layout(height=300, title='By Day of Week', margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        sel_nta = st.selectbox("Zoom into a neighborhood",
                                ["All NYC"] + sorted(ts['nta_name'].dropna().unique().tolist()), key="ts_nta")
        ts_filt = ts if sel_nta == "All NYC" else ts[ts['nta_name'] == sel_nta]

        t5 = ts_filt['complaint_type'].value_counts().head(5).index
        wk = ts_filt[ts_filt['complaint_type'].isin(t5)].set_index('date').groupby(
            'complaint_type').resample('W').size().reset_index(name='n')
        fig = px.line(wk, x='date', y='n', color='complaint_type',
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, title=f'Weekly by Type — {sel_nta}', margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series data.")

# ═══════ TAB 9: DEMOGRAPHICS ══════════════════════════════════════════════════
with tabs[10]:
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
with tabs[11]:
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
