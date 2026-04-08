"""
NYC "Where Should I Live?" — Streamlit Dashboard
==================================================
Polis Technologies · Section 1

Run:  streamlit run app.py
Requires: nta_livability_scores.csv, nyc_livability.db,
          nta_similarity_matrix.csv, nta_centroids.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os, time, json
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NYC Livability", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
.stApp{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'JetBrains Mono',monospace!important}
[data-testid="stSidebar"]{background:#0f172a}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#059669 100%);
      padding:2.2rem 2rem;border-radius:16px;color:#fff;margin-bottom:1.2rem}
.hero h1{color:#fff!important;font-size:2rem;margin-bottom:4px}
.hero p{color:#cbd5e1;margin:0;font-size:.95rem}
.scard{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.2rem;text-align:center}
.snum{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:#0f172a}
.slbl{color:#64748b;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.pill{display:inline-block;padding:3px 10px;border-radius:99px;font-size:.75rem;font-weight:600}
.pg{background:#d1fae5;color:#065f46}.pb{background:#dbeafe;color:#1e40af}
.pa{background:#fef3c7;color:#92400e}.pr{background:#fee2e2;color:#991b1b}
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

@st.cache_data
def load_listings():
    if not os.path.exists("nyc_livability.db"):
        return pd.DataFrame()
    c = sqlite3.connect("nyc_livability.db")
    try:
        return pd.read_sql("SELECT * FROM rent_listings", c)
    except:
        return pd.DataFrame()
    finally:
        c.close()

@st.cache_data
def load_redfin():
    if not os.path.exists("nyc_livability.db"):
        return pd.DataFrame()
    c = sqlite3.connect("nyc_livability.db")
    try:
        return pd.read_sql("SELECT * FROM redfin_listings", c)
    except:
        return pd.DataFrame()
    finally:
        c.close()

@st.cache_data
def load_311():
    if not os.path.exists("nyc_livability.db"):
        return pd.DataFrame()
    c = sqlite3.connect("nyc_livability.db")
    try:
        return pd.read_sql("SELECT created_date,complaint_type,nta_name,borough FROM complaints_311 WHERE created_date IS NOT NULL", c)
    except:
        return pd.DataFrame()
    finally:
        c.close()

@st.cache_data
def load_sim():
    if os.path.exists("nta_similarity_matrix.csv"):
        return pd.read_csv("nta_similarity_matrix.csv", index_col=0)
    return None

@st.cache_data
def load_geojson():
    import requests
    for url in ["https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?$limit=500",
                "https://raw.githubusercontent.com/nycehs/NYC_geography/master/NTA.geo.json"]:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except:
            continue
    return None

df = load_master()
listings = load_listings()
redfin = load_redfin()
ts_data = load_311()
sim_mx = load_sim()
geo = load_geojson()

# Detect best rent column
RC = next((c for c in ['redfin_median_rent', 'census_median_rent', 'predicted_rent'] if c in df.columns and df[c].notna().sum() > 20), None)
TC = 'transit_plus_score' if 'transit_plus_score' in df.columns else 'transit_score'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Your Priorities")
    persona = st.selectbox("Preset", ["Custom", "👩‍💻 Professional", "👨‍👩‍👧 Family",
                                       "🎓 Student", "🧘 Retiree", "🎨 Creative"])
    P = {"Custom": (30, 25, 20, 25), "👩‍💻 Professional": (25, 10, 10, 55),
         "👨‍👩‍👧 Family": (40, 30, 25, 5), "🎓 Student": (15, 10, 5, 70),
         "🧘 Retiree": (30, 40, 25, 5), "🎨 Creative": (15, 25, 25, 35)}
    d = P.get(persona, (30, 25, 20, 25))
    sw = st.slider("🛡️ Safety", 0, 100, d[0], 5)
    nw = st.slider("🔇 Quiet", 0, 100, d[1], 5)
    pw = st.slider("🌳 Parks", 0, 100, d[2], 5)
    tw = st.slider("🚇 Transit", 0, 100, d[3], 5)
    st.divider()
    use_b = st.toggle("💰 Budget filter")
    mb = st.slider("Max $/mo", 1500, 10000, 3500, 100) if use_b else None
    st.divider()
    boros = ["All"] + sorted(df['borough'].dropna().unique().tolist())
    sb = st.selectbox("📍 Borough", boros)
    sb = None if sb == "All" else sb
    topn = st.slider("Results to show", 5, 30, 10)

# Compute user score
W = max(sw + nw + pw + tw, 1)
df['user_score'] = (sw / W * df['safety_score'] + nw / W * df['noise_score'] +
                    pw / W * df['parks_score'] + tw / W * df[TC])

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
st.markdown("""<div class="hero"><h1>🏙️ NYC — Where Should I Live?</h1>
<p>197 neighborhoods · 100K+ records · 9 data sources · ML-powered scoring · MCP tool calling</p></div>""",
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
def scard(n, l):
    return f'<div class="scard"><div class="snum">{n}</div><div class="slbl">{l}</div></div>'
with c1:
    st.markdown(scard(len(df), "Neighborhoods"), unsafe_allow_html=True)
with c2:
    st.markdown(scard(len(ft), "Match Filters"), unsafe_allow_html=True)
with c3:
    nm = top.iloc[0]['nta_name'].split('-')[0] if len(top) > 0 else "—"
    st.markdown(scard(f'<span style="font-size:.95rem">{nm}</span>', "#1 Pick"), unsafe_allow_html=True)
with c4:
    rv = f"${top.iloc[0][RC]:,.0f}" if len(top) > 0 and RC and pd.notna(top.iloc[0].get(RC)) else "—"
    st.markdown(scard(rv, "Est. Rent"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🏆 Picks", "🗺️ Map", "🤖 AI Chat", "🚇 Commute", "🔍 Similar",
                "🧬 Clusters", "🏠 Listings", "📈 Trends", "👥 Demographics", "📊 Data"])

# ═══════ TAB 1: PICKS ═════════════════════════════════════════════════════════
with tabs[0]:
    if len(top) == 0:
        st.warning("No results. Adjust filters.")
    else:
        for rk, (_, r) in enumerate(top.iterrows(), 1):
            s = r['user_score']
            css = "pg" if s > .65 else ("pa" if s > .45 else "pr")
            cr, ci, cs, cv = st.columns([.35, 2.5, 2, 1])
            with cr:
                st.markdown(f"### #{rk}")
            with ci:
                st.markdown(f"**{r['nta_name']}** · {r['borough']}")
                if 'cluster_label' in r.index and pd.notna(r.get('cluster_label')):
                    st.markdown(f'<span class="pill pb">{r["cluster_label"]}</span>', unsafe_allow_html=True)
            with cs:
                dims = f"Safety {r['safety_score']:.2f} · Noise {r['noise_score']:.2f} · Parks {r['parks_score']:.2f} · Transit {r[TC]:.2f}"
                if 'rent_score' in r.index:
                    dims += f" · Rent {r['rent_score']:.2f}"
                st.caption(dims)
                st.progress(min(s, 1.0))
            with cv:
                st.markdown(f'<span class="pill {css}">{s:.3f}</span>', unsafe_allow_html=True)
                if RC and pd.notna(r.get(RC)):
                    st.caption(f"~${r[RC]:,.0f}/mo")
            st.divider()

# ═══════ TAB 2: MAP ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Livability Map")
    st.caption("Color = your custom score. Hover for details.")
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
                                       color_continuous_scale='YlGn', range_color=[0, 1],
                                       mapbox_style='carto-positron', zoom=10,
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
with tabs[2]:
    st.markdown("### 🤖 AI Neighborhood Advisor")
    st.markdown("*MCP tool calling — Claude queries the database, finds similar neighborhoods, searches listings, and estimates commutes in real time.*")

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
        ls = listings.copy() if len(listings) > 0 else pd.DataFrame()
        if len(ls) == 0:
            return {"error": "No listings loaded"}
        if borough:
            ls = ls[ls['borough'].str.lower() == borough.lower()]
        if max_price:
            ls = ls[ls['price'] <= max_price]
        if min_bedrooms and 'bedrooms' in ls.columns:
            ls = ls[ls['bedrooms'] >= min_bedrooms]
        cols = [c for c in ['title', 'price', 'borough', 'bedrooms', 'sqft', 'no_fee', 'sentiment_label'] if c in ls.columns]
        return ls[cols].sort_values('price').head(limit).to_dict(orient='records')

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
        return {"neighborhood": r['nta_name'], "distance_km": round(dist, 1), "est_minutes": round(dist / 25 * 60)}

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

        sys_prompt = f"""You are an NYC neighborhood advisor. You have 5 tools to query a database of 197 scored neighborhoods.
ALWAYS use tools to answer — never guess. User priorities: Safety {sw}%, Noise {nw}%, Parks {pw}%, Transit {tw}%.
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
                                tool_log.append(f"🔧 `{blk['name']}({json.dumps(blk['input'], default=str)[:80]})`")
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
                        with st.expander(f"🔧 {len(tool_log)} tool call(s)", expanded=False):
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
with tabs[3]:
    st.markdown("### 🚇 Commute Estimator")
    st.caption("Enter your work address to estimate commute from each neighborhood.")

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
                    st.success(f"📍 {g[0].get('display_name', '')[:80]}")

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
with tabs[4]:
    st.markdown("### 🔍 Find Similar Neighborhoods")
    st.caption('"I like Williamsburg — what else would I like?"')

    ref = st.selectbox("I like...", sorted(df['nta_name'].unique()), key="sim_ref")
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

# ═══════ TAB 6: CLUSTERS ══════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🧬 Neighborhood Clusters")
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
with tabs[6]:
    st.markdown("### 🏠 Rental Listings")

    # Combine Craigslist + Redfin
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
        if mb and 'price' in ls.columns:
            ls = ls[ls['price'] <= mb]
        ls = ls.sort_values('price')
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
with tabs[7]:
    st.markdown("### 📈 311 Complaint Trends")
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
with tabs[8]:
    st.markdown("### 👥 Neighborhood Demographics")
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
        corr = df[num].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto')
        fig.update_layout(height=500, title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run Census cells in the notebook.")

# ═══════ TAB 10: DATA ═════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown("### 📊 Full Dataset")
    defaults = ['nta_name', 'borough', 'user_score', 'safety_score', 'noise_score', 'parks_score', TC]
    if 'rent_score' in df.columns:
        defaults.append('rent_score')
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
        st.download_button("📥 Download CSV", ft[sel].to_csv(index=False), "livability.csv", "text/csv")

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
st.divider()
st.caption("Polis Technologies · Section 1 · "
           f"{len(df)} NTAs · 9 data sources · ML-powered · MCP tool calling · "
           "NYC Open Data · Census ACS · Citi Bike · MTA · Redfin · Craigslist")
