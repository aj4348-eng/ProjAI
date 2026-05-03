# NYC "Where Should I Live?"

**Polis Technologies · Section 01**

A 10-tab data product scoring 197 NYC neighborhoods across 6 livability dimensions, with interactive choropleth map, AI advisor (MCP tool calling), commute calculator, similarity engine, and Census demographics analysis.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, select `app.py` as the main file
4. Click Deploy

No secrets or environment variables needed — the app reads from the SQLite database in this folder.

## Data Sources

NYC Open Data · Census ACS · Citi Bike · MTA · Redfin · Craigslist
