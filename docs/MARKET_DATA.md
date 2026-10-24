# Market data and resale query semantics

The resale-query tool reads a local DuckDB snapshot of official transaction
data. It is enabled in `config/experiment_config.yaml`; the remaining market,
financial and policy tools use mocks.

For installation, environment settings, database builds, tests, and running the
application or MCP server, see [README setup](../README.md#setup).

## Source and schema

Official dataset: [Resale prices from January 2017 onward](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view).
API documentation: [Download dataset](https://guide.data.gov.sg/developer-guide/dataset-apis/download-dataset).

The bulk CSV has 11 columns; the datastore API additionally returns `_id`.
The API ID is not used as a permanent transaction or flat identity.

`data/sql/market_schema.sql` defines schema version 1:

| Relation | Contents |
|---|---|
| `source_snapshot` | Source ID/URL, raw CSV and spatial-index checksums, retrieval/build times, coverage, analysis month, profile |
| `resale_transactions` | One published record per source row; typed transaction month, normalized town/block/street/type, storey range, decimal area/price, lease commencement year, raw and parsed remaining lease |
| `block_locations` | One block/street key with optional postal code, mean coordinates of matching footprints, source and footprint count |
| `blocks` (view) | Distinct block identities and transaction coverage |

The block key is uppercase `"{block} {street}"`, matching the shared contract.
Block numbers and postal codes remain strings. Distinct building footprints
with the same key are combined before joins so they cannot multiply sales.
Unmatched locations remain null; no coordinates or fallback street names are
invented. HDB Property Information is not needed for the resale query.

The original CSV is retained in the ignored cache. Money and floor area are
stored as decimals. Lease strings (including year-only values) are parsed into
integer months, retaining the original text. Source row numbers identify rows
within this snapshot, not unique flats. Identical-looking transactions are
counted in the profile and retained because they may be separate sales.

Row counts, date coverage, duplicate counts and coordinate matches are recorded
in `source_snapshot.profile` for each database build.

## Query semantics

The tool accepts four parameters:

```python
query_resale_blocks_by_criteria(
    towns=["BISHAN"], flat_types=["4 ROOM"],
    max_price=850000, min_lease_years=60,
)
```

- Empty town/type lists select all; blank values, unknown categories and invalid
  numeric limits are errors. Case and whitespace are normalized.
- Use the last 12 calendar months ending at the earlier of the latest source
  month and the month before retrieval. This excludes the in-progress month at
  download time. Missing older months are not fabricated. Reusing a cached CSV
  preserves its retrieval time and analysis window.
- Group by block, street, town and flat type. Compute medians over all matching
  transactions in the window, then apply budget and lease limits to those
  medians. Never discard expensive sales before computing the median.
- Price per square metre is the median of individual transaction ratios.
- Lease is estimated at the shared window-end month: subtract elapsed months
  from each transaction's reported remaining lease before taking the median.
  It is a month-resolution historical estimate, not a current-day or legal
  unit-specific lease determination.
- Results include transaction counts (including single-sale groups), coordinates
  where available, and latest transaction month. These are historical market
  summaries, not asking prices or flats available for purchase.
- Return at most 100 groups, ordered deterministically by median price and
  identity. Metadata reports total matches and whether results were truncated.
- Metadata includes source, snapshot, date window, lease reference and caveats.
  The parameterized SQL template is included; user arguments are never inserted
  as SQL text. Every query opens its own read-only connection, with no network.
- Both direct calls and MCP calls validate arguments and output with the shared
  Pydantic models. Missing databases produce an actionable setup error, with no
  silent fallback to fixtures.

## Validation

Tests use independently calculated synthetic cases. Mock benchmarks explicitly
select mock tools so their fixtures remain reproducible. Category A accuracy on
real data requires a separate labelled evaluation dataset.
