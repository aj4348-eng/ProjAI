import os, sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

SQLITE_PATH = os.environ.get("SQLITE_PATH", "nyc_livability.db")
# TABLES = ["rent_listings", "redfin_listings", "complaints_311"]
# Instead of hardcoding, auto-discover tables in the sqlite file.
IGNORED = {"sqlite_sequence", "sqlite_stat1", "sqlite_stat3", "sqlite_stat4"}

def main():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("ERROR: set DATABASE_URL environment variable to your Postgres (Neon) connection string.")
        sys.exit(1)

    if not os.path.exists(SQLITE_PATH):
        print(f"ERROR: SQLite file not found at {SQLITE_PATH}")
        sys.exit(1)

    try:
        sqlite_engine = create_engine(f"sqlite:///{SQLITE_PATH}")
        pg_engine = create_engine(DATABASE_URL)
    except SQLAlchemyError as e:
        print("ERROR creating engines:", e)
        sys.exit(1)

    # discover tables
    try:
        tbls_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", sqlite_engine)
        tables = [r['name'] for _, r in tbls_df.iterrows() if r['name'] not in IGNORED]
    except Exception as e:
        print("ERROR reading sqlite_master:", e)
        sys.exit(1)

    if not tables:
        print("No user tables found in sqlite.")
        sys.exit(0)

    print("Found tables:", tables)

    for tbl in tables:
        print(f"Reading table '{tbl}' from sqlite...")
        try:
            df = pd.read_sql(f"SELECT * FROM {tbl}", sqlite_engine)
        except Exception as e:
            print(f"Failed to read '{tbl}': {e}")
            continue

        print(f"Copying {len(df)} rows of '{tbl}' to Postgres...")
        try:
            df.to_sql(tbl, pg_engine, if_exists="replace", index=False, method="multi", chunksize=1000)
            print(f"Finished: '{tbl}' -> Postgres")
        except Exception as e:
            print(f"Failed to write table '{tbl}': {e}")

    print("Migration complete.")

if __name__ == "__main__":
    main()