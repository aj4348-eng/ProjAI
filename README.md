# NYC "Where Should I Live?" — Streamlit Dashboard

**Polis Technologies · Section 1**

A 9-tab data product scoring 197 NYC neighborhoods across 6 livability dimensions, with interactive choropleth map, commute calculator, similarity engine, and Census demographics analysis.

---

## Repository Contents

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit dashboard (9 tabs, dual DB support) |
| `requirements.txt` | Python dependencies |
| `migrate.py` | One-time SQLite → Postgres migration script |
| `secrets.toml.example` | Template for Streamlit Cloud secrets |
| `.gitignore` | Prevents secrets and cache files from being pushed |
| `nta_livability_scores.csv` | 197 scored neighborhoods × 35 columns |
| `nta_similarity_matrix.csv` | 197×197 cosine similarity matrix |
| `nta_centroids.csv` | NTA centroids for commute calculator |
| `rent_prediction_model.pkl` | Trained Random Forest regressor |
| `kmeans_model.pkl` | Trained K-Means clustering model |
| `cluster_classifier.pkl` | Trained Random Forest cluster predictor |

---

## Database Modes

The app supports **two database backends** and auto-detects which to use:

| Mode | Trigger | When to use |
|------|---------|-------------|
| **Postgres (Neon)** | `DATABASE_URL` set in env or secrets | Streamlit Cloud, multi-user, persistent |
| **SQLite (local)** | Falls back when `DATABASE_URL` is absent | Local development, demos |

The sidebar shows which DB is active (🟢 Postgres / 🟡 SQLite).

---

## Run Locally (SQLite)

```bash
git clone <your-repo>
cd <repo>
pip install -r requirements.txt
streamlit run app.py
```

For local mode you need `nyc_livability.db` in the repo folder. **Do not commit this file** to GitHub — it's listed in `.gitignore`. Generate it by running the Colab notebook.

---

## Deploy to Streamlit Cloud (Postgres + Neon)

### Step 1 — Migrate SQLite → Postgres (one time)

Locally, with both `nyc_livability.db` present and `DATABASE_URL` set:

```bash
export DATABASE_URL="postgresql://neondb_owner:...@neon.tech/neondb?sslmode=require"
export SQLITE_PATH="nyc_livability.db"
python migrate.py
```

This auto-discovers all SQLite tables and copies them to Neon Postgres.

### Step 2 — Push to GitHub

The repo is already configured. Just initialize and push:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

`.gitignore` ensures `nyc_livability.db` and `secrets.toml` are not pushed.

### Step 3 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. New app → connect your GitHub repo → set `app.py` as main file
3. Click **Advanced settings** → **Secrets** → paste:

```toml
DATABASE_URL = "postgresql://neondb_owner:...@neon.tech/neondb?sslmode=require&channel_binding=require"
```

4. Deploy. You get a permanent public URL.

---

## Tabs

| # | Tab | Description |
|---|-----|-------------|
| 1 | Rankings | Ranked neighborhoods with cluster tags, scores, rent estimates |
| 2 | Map | Interactive choropleth of all 197 NTAs |
| 3 | Commute | Nominatim geocoding + OSRM routing for top 20 NTAs |
| 4 | Similar | Cosine similarity lookup with radar overlay |
| 5 | Clusters | K-Means profiles with radar charts and borough composition |
| 6 | Listings | Combined Craigslist + Redfin with sentiment analysis |
| 7 | Trends | 311 time series with per-neighborhood drill-down |
| 8 | Demographics | Census income/population/age/education with correlation matrix |
| 9 | Data | Full dataset with column selector and CSV download |
