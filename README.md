# NYC "Where Should I Live?" — Streamlit Dashboard

**Polis Technologies · Section 1**

197 neighborhoods · 9 data sources · ML-powered scoring · MCP tool calling

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file → Deploy

## Files

| File | Purpose |
|------|---------|
| `app.py` | 10-tab Streamlit dashboard with MCP AI chat |
| `nta_livability_scores.csv` | 197 scored neighborhoods (35 columns) |
| `nyc_livability.db` | SQLite database (100K+ records, 9 tables) |
| `nta_similarity_matrix.csv` | 197×197 cosine similarity matrix |
| `nta_centroids.csv` | NTA centroids for commute calculator |
| `*.pkl` | Trained ML models (rent prediction, K-Means, classifier) |
