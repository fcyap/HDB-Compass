"""Settings and read-only connections shared by ingestion and market tools."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import duckdb
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESALE_DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
RESALE_SOURCE_URL = f"https://data.gov.sg/datasets/{RESALE_DATASET_ID}/view"
SCHEMA_VERSION = 1


def market_settings(root: Path = PROJECT_ROOT) -> dict:
    """Environment > .env > defaults. Never require or print credentials.

    Read settings without mutating os.environ.
    Relative database paths always resolve against the repository, not the cwd.
    """
    values = {**dotenv_values(root / ".env"), **os.environ}
    path = Path(values.get("HDB_COMPASS_DB_PATH") or "data/local/hdb_compass.duckdb")
    return {
        "db_path": path if path.is_absolute() else root / path,
        "api_key": (values.get("DATA_GOV_SG_API_KEY") or "").strip(),
    }


@contextmanager
def open_market_database(path: Path | None = None):
    path = Path(path or market_settings()["db_path"])
    if not path.is_file():
        raise FileNotFoundError(
            "Market database is missing. Run python -m data.build_market_db first "
            "and check HDB_COMPASS_DB_PATH in .env."
        )
    con = duckdb.connect(str(path), read_only=True)
    try:
        version = con.execute("SELECT schema_version FROM source_snapshot").fetchone()
        if version is None or version[0] != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported market schema. Rebuild with python -m data.build_market_db."
            )
        yield con
    finally:
        con.close()
