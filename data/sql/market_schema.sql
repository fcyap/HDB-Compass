-- Each build contains one complete source snapshot. Never append repeated downloads.
CREATE TABLE source_snapshot (
    snapshot_id VARCHAR PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    dataset_id VARCHAR NOT NULL,
    source_url VARCHAR NOT NULL,
    source_sha256 VARCHAR NOT NULL,
    spatial_sha256 VARCHAR,
    retrieved_at TIMESTAMPTZ NOT NULL,
    built_at TIMESTAMPTZ NOT NULL,
    row_count BIGINT NOT NULL,
    first_month DATE NOT NULL,
    latest_month DATE NOT NULL,
    analysis_end_month DATE NOT NULL,
    profile JSON NOT NULL
);

CREATE TABLE resale_transactions (
    -- This is a source-file row identity, not a unique flat or a permanent transaction ID.
    source_row_number BIGINT PRIMARY KEY,
    month DATE NOT NULL,
    town VARCHAR NOT NULL,
    block VARCHAR NOT NULL,
    street VARCHAR NOT NULL,
    block_key VARCHAR NOT NULL,
    flat_type VARCHAR NOT NULL,
    storey_range_raw VARCHAR NOT NULL,
    storey_min SMALLINT NOT NULL,
    storey_max SMALLINT NOT NULL CHECK (storey_max >= storey_min),
    floor_area_sqm DECIMAL(10, 2) NOT NULL CHECK (floor_area_sqm > 0),
    flat_model VARCHAR NOT NULL,
    lease_commence_year SMALLINT NOT NULL,
    remaining_lease_raw VARCHAR NOT NULL,
    remaining_lease_months SMALLINT NOT NULL CHECK (remaining_lease_months BETWEEN 0 AND 1200),
    resale_price DECIMAL(14, 2) NOT NULL CHECK (resale_price > 0)
);

CREATE TABLE block_locations (
    block_key VARCHAR PRIMARY KEY,
    postal VARCHAR,
    lat DOUBLE NOT NULL CHECK (lat BETWEEN -90 AND 90),
    lng DOUBLE NOT NULL CHECK (lng BETWEEN -180 AND 180),
    source VARCHAR NOT NULL,
    footprint_count INTEGER NOT NULL
);

CREATE VIEW blocks AS
SELECT block_key, town, block, street, count(*) AS transaction_count,
       min(month) AS first_transaction_month, max(month) AS last_transaction_month
FROM resale_transactions GROUP BY block_key, town, block, street;
