"""Build a validated local resale snapshot: python -m data.build_market_db."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from data.gov_download import download_resale_csv, sha256_file
from data.market_database import (
    PROJECT_ROOT,
    RESALE_DATASET_ID,
    RESALE_SOURCE_URL,
    SCHEMA_VERSION,
    market_settings,
)

SOURCE_COLUMNS = (
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
)
FLAT_TYPES = {"1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"}
LEASE_RE = re.compile(r"(?:(\d+)\s+years?)?(?:\s*(\d+)\s+months?)?", re.I)


def canonical(value: str) -> str:
    return " ".join(value.upper().split())


def parse_lease_months(value: str) -> int:
    match = LEASE_RE.fullmatch(value.strip())
    if not match or not any(match.groups()):
        raise ValueError("invalid remaining_lease")
    years, months = (int(part or 0) for part in match.groups())
    total = years * 12 + months
    if months > 11 or not 0 <= total <= 1200:
        raise ValueError("remaining_lease out of range")
    return total


def positive_decimal(value: str, field: str) -> str:
    try:
        number = Decimal(value)
        if not number.is_finite() or number <= 0 or number != number.quantize(Decimal("0.01")):
            raise ValueError(field)
    except (InvalidOperation, ValueError):
        raise ValueError(f"invalid {field}") from None
    return str(number)


def normalise_row(row: dict, row_number: int) -> list:
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in SOURCE_COLUMNS):
        raise ValueError("missing source value")
    if not re.fullmatch(r"\d{4}-\d{2}", row["month"]):
        raise ValueError("invalid month")
    month = date.fromisoformat(row["month"] + "-01")
    town, block, street, flat_type = [
        canonical(row[key]) for key in ("town", "block", "street_name", "flat_type")
    ]
    if not re.fullmatch(r"\d+[A-Z]?", block):
        raise ValueError("invalid block")
    if flat_type not in FLAT_TYPES:
        raise ValueError(f"unrecognised flat_type: {flat_type}")
    storey = re.fullmatch(r"(\d+)\s+TO\s+(\d+)", canonical(row["storey_range"]))
    if not storey or not 0 <= int(storey[1]) <= int(storey[2]) <= 200:
        raise ValueError("invalid storey_range")
    year = int(row["lease_commence_date"])
    # Some records can precede the nominal lease-commencement year.
    if not 1900 <= year <= 2200:
        raise ValueError("invalid lease_commence_date")
    return [
        row_number,
        month.isoformat(),
        town,
        block,
        street,
        f"{block} {street}",
        flat_type,
        row["storey_range"],
        int(storey[1]),
        int(storey[2]),
        positive_decimal(row["floor_area_sqm"], "floor_area_sqm"),
        row["flat_model"].strip(),
        year,
        row["remaining_lease"],
        parse_lease_months(row["remaining_lease"]),
        positive_decimal(row["resale_price"], "resale_price"),
    ]


def normalise_csv(source: Path, target: Path) -> dict:
    errors, error_count, row_count = [], 0, 0
    with (
        source.open(newline="", encoding="utf-8-sig") as stream,
        target.open("w", newline="", encoding="utf-8") as out,
    ):
        reader = csv.DictReader(stream)
        missing = set(SOURCE_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Resale CSV is missing columns: {sorted(missing)}")
        writer = csv.writer(out)
        for row_count, row in enumerate(reader, 1):
            try:
                if None in row:
                    raise ValueError("unexpected extra CSV values")
                writer.writerow(normalise_row(row, row_count))
            except (ValueError, TypeError) as exc:
                error_count += 1
                if len(errors) < 5:
                    errors.append({"source_row_number": row_count, "reason": str(exc)})
        if error_count:
            raise ValueError(
                f"Rejected {error_count} of {row_count} rows; database unchanged. Examples: {errors}"
            )
        if not row_count:
            raise ValueError("Resale CSV contains no records; database unchanged.")
    return {
        "row_count": row_count,
        "source_columns": reader.fieldnames,
        "extra_source_columns": sorted(set(reader.fieldnames) - set(SOURCE_COLUMNS)),
        "rejected_rows": 0,
    }


def load_locations(con, spatial_path: Path | None) -> str | None:
    if spatial_path is None:
        return None
    # Only load the trusted, repository-generated index; never unpickle a downloaded file.
    with spatial_path.open("rb") as stream:
        blocks = pickle.load(stream)["payload"]["blocks"]
    grouped = defaultdict(list)
    for block in blocks:
        key = f"{canonical(str(block['block']))} {canonical(block['street'])}"
        lat, lng = float(block["lat"]), float(block["lng"])
        if not isfinite(lat) or not isfinite(lng):
            continue
        grouped[key].append((block.get("postal"), lat, lng))
    rows = []
    for key, points in grouped.items():
        postals = {p[0] for p in points if p[0]}
        rows.append(
            (
                key,
                next(iter(postals)) if len(postals) == 1 else None,
                sum(p[1] for p in points) / len(points),
                sum(p[2] for p in points) / len(points),
                "repository spatial_index.pkl; mean of matching footprints",
                len(points),
            )
        )
    if rows:
        con.executemany("INSERT INTO block_locations VALUES (?, ?, ?, ?, ?, ?)", rows)
    return sha256_file(spatial_path)


def build_database(
    source: Path,
    database: Path,
    *,
    spatial_path: Path | None = None,
    retrieved_at: datetime | None = None,
) -> dict:
    """Validate a complete replacement in a private directory before publishing it.

    Duplicate-looking transactions are counted and retained, not deduplicated.
    A bad input or failed build leaves the previous database untouched.
    """
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone")
    database = database.resolve()
    if source.resolve() == database:
        raise ValueError("CSV source and database output must be different files")
    database.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="market-build-", dir=database.parent) as temp:
        normalised = Path(temp) / "normalised.csv"
        profile = normalise_csv(source, normalised)
        candidate = Path(temp) / "snapshot.duckdb"
        con = duckdb.connect(str(candidate))
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute(
                (Path(__file__).parent / "sql" / "market_schema.sql").read_text(encoding="utf-8")
            )
            con.execute(
                "INSERT INTO resale_transactions SELECT * FROM read_csv(?, header=false, all_varchar=true, auto_detect=true)",
                [str(normalised)],
            )
            first, latest = con.execute(
                "SELECT min(month), max(month) FROM resale_transactions"
            ).fetchone()
            capture_month = retrieved_at.date().replace(day=1)
            if latest > capture_month:
                raise ValueError("Source contains transactions after its retrieval month.")
            previous_month = date(
                capture_month.year - (capture_month.month == 1),
                (capture_month.month - 2) % 12 + 1,
                1,
            )
            end = min(latest, previous_month)
            if first > end:
                raise ValueError("Snapshot has no completed calendar month to analyse.")
            spatial_hash = load_locations(con, spatial_path)
            raw_hash = sha256_file(source)
            identity = f"{SCHEMA_VERSION}:{raw_hash}:{spatial_hash}:{end}"
            snapshot_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            profile.update(
                {
                    "first_month": first.strftime("%Y-%m"),
                    "latest_month": latest.strftime("%Y-%m"),
                    "analysis_end_month": end.strftime("%Y-%m"),
                    "source_sha256": raw_hash,
                    "source_bytes": source.stat().st_size,
                    "snapshot_id": snapshot_id,
                    "spatial_sha256": spatial_hash,
                }
            )
            columns = "month, town, block, street, flat_type, storey_range_raw, floor_area_sqm, flat_model, lease_commence_year, remaining_lease_raw, resale_price"
            profile["duplicate_looking_rows_retained"] = con.execute(
                f"SELECT coalesce(sum(n - 1), 0) FROM (SELECT count(*) n FROM resale_transactions GROUP BY {columns} HAVING count(*) > 1)"
            ).fetchone()[0]
            profile["distinct_blocks"], profile["blocks_with_coordinates"] = con.execute(
                "SELECT count(DISTINCT t.block_key), count(DISTINCT l.block_key) "
                "FROM resale_transactions t LEFT JOIN block_locations l USING (block_key)"
            ).fetchone()
            profile["towns"] = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT town FROM resale_transactions ORDER BY town"
                ).fetchall()
            ]
            profile["flat_types"] = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT flat_type FROM resale_transactions ORDER BY flat_type"
                ).fetchall()
            ]
            con.execute(
                "INSERT INTO source_snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    snapshot_id,
                    SCHEMA_VERSION,
                    RESALE_DATASET_ID,
                    RESALE_SOURCE_URL,
                    raw_hash,
                    spatial_hash,
                    retrieved_at,
                    datetime.now(timezone.utc),
                    profile["row_count"],
                    first,
                    latest,
                    end,
                    json.dumps(profile),
                ],
            )
            con.execute("COMMIT")
            con.execute("CHECKPOINT")
        finally:
            con.close()
        candidate.replace(database)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Build offline from a local official-format CSV")
    parser.add_argument("--db", type=Path, help="Override HDB_COMPASS_DB_PATH")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download a new source instead of reusing the verified cache",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Skip coordinate enrichment; no OneMap access is used either way",
    )
    args = parser.parse_args()
    settings = market_settings()
    try:
        metadata = {}
        if args.csv:
            source = args.csv
        else:
            print(
                "Fetching public resale CSV (or reusing checksum-verified local cache)...",
                flush=True,
            )
            source, metadata = download_resale_csv(
                PROJECT_ROOT / "data" / "cache" / "market", settings["api_key"], args.refresh
            )
        print("Validating records and building DuckDB snapshot...", flush=True)
        profile = build_database(
            source,
            args.db or settings["db_path"],
            spatial_path=None if args.no_spatial else PROJECT_ROOT / "data" / "spatial_index.pkl",
            retrieved_at=datetime.fromisoformat(metadata["retrieved_at"]) if metadata else None,
        )
        print(json.dumps(profile, indent=2))
        print("Database ready. Full profile is stored in source_snapshot.profile.")
    except (ValueError, RuntimeError, OSError, duckdb.Error) as exc:
        parser.exit(1, f"Market setup failed: {exc}\n")


if __name__ == "__main__":
    main()
