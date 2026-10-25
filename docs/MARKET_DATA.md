# Market data and query semantics

The resale-query, town price-trend and comparable-sale tools read a local DuckDB
snapshot of official transaction data. All three are enabled in `config/experiment_config.yaml`;
BTO comparison and policy tools use mocks. The three financial calculators are
also real; see [financial contracts and assumptions](FINANCIAL_TOOLS.md).

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

## Resale query

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

## Comparable resale transactions

`query_comparable_resale_transactions` accepts verified WGS84 `latitude` and
`longitude`, a `flat_type`, and optional matching criteria. It queries the same
snapshot without changing the database schema, downloading data or calling a
location service. An agent must obtain the location from supplied facts or the
spatial tools; a guessed town centre is not a verified project location.

- Match an inclusive straight-line radius (default 500 m, maximum 10 km), using
  haversine distance with a mean Earth radius of 6,371,008.8 m. Distances refer
  to the indexed block location, not walking routes or exact units. Location
  precision and missing geocodes limit the comparison.
- Require an exact normalized flat type. Optional inclusive min/max floor area,
  remaining lease and storey bounds narrow matching without automatic expansion.
  Unspecified attributes are unrestricted, so nearby sales are not necessarily
  like-for-like sales. The caller should choose and disclose appropriate filters.
- Storey data is published as bands. The entire band must fit supplied bounds;
  for example, a 04–06 band does not satisfy a minimum floor of 5.
- Use `lookback_months` (1–120, default 12) ending at `end_month` (`YYYY-MM`),
  defaulting to the snapshot's last completed analysis month. Reject end months
  beyond that limit instead of silently substituting older data. Historical
  windows support comparisons around a launch date without including later sales.
  Dates are month-resolution; matching by an exact launch day is not supported.
- Adjust each sale's reported lease to the requested end month before filtering
  or calculating the median. This is an estimate, not a legal unit-specific lease.
- Calculate minimum, lower quartile, median, upper quartile and maximum of sale
  prices and individual price-per-square-metre ratios over **all** matching rows.
  Quartiles use continuous linear interpolation. Include block count, median area,
  median lease and observed transaction dates. Preserve duplicate-looking source
  rows because they are not permanent unit identities.
- Return up to `sample_limit` examples (1–100, default 20), ordered by distance,
  then descending month and deterministic identity/price/row tie-breakers. Limiting
  examples never limits the summary. `sample_truncated` discloses this separately.
- `status` is `ok`, `insufficient_sample` (fewer than `min_sample_size`, default 5),
  `no_matches`, or `no_coverage` when the requested end precedes source history.
  Empty populations have null summaries, not zero prices. A low sample retains
  statistics with a warning; the threshold is not a confidence guarantee.
- Metadata includes source/snapshot, UTC retrieval time, first/latest source months,
  requested window, lease reference and history truncation. It also counts sales
  matching the non-spatial criteria that lack coordinates. Their proximity is
  unknown; this count is across the snapshot, not an estimate of nearby omissions.
  Source date coverage does not prove every monthly transaction has been reported.
- The tool returns parameterized summary SQL and validated criteria for auditing.
  Unknown flat types, invalid ranges and malformed arguments fail explicitly.
  Statistics describe historical evidence, not current listings, a modelled
  valuation, an adjusted new-build price or an affordability/eligibility decision.

The new callable is the twelfth inventory entry and is available through the
agent registry, market MCP server and unified MCP server. Structured extraction
can populate `user_constraints.comparable_resale`; the deterministic specialist
then calls it and preserves evidence under `sql_market_results`. The rule-based
text extractor does not geocode names or infer comparison coordinates. Live-model
selection and the location handoff remain integration work.
Comparable-only analytics do not create candidate listings or trigger the
supervisor's zero-result block-search relaxation flow, even if a budget is supplied.

Run `python -m pytest tests/test_comparable_resale.py -q` to verify independent
quartiles/ratios, temporal/spatial/attribute boundaries, missing coordinates,
source-row preservation, validation, deterministic graph execution and MCP calls
using temporary synthetic snapshots. These do not establish Category A accuracy.

## Town price trends

```python
compute_town_price_timeseries(
    towns=["BISHAN", "TAMPINES"], flat_type="4 ROOM", lookback_years=3,
)
```

- Town and flat-type filters are normalized and validated against the database.
  An empty town list selects all towns; `lookback_years` must be an integer from
  1 to 30. Unknown categories and blank values are errors.
- Return up to four calendar quarters per requested year, ending in the quarter
  containing the snapshot's analysis month. Source coverage can shorten the
  window. No data from the in-progress month at retrieval is included.
- Aggregate actual transactions by town, flat type and calendar quarter. Each
  point contains median price, median individual price per square metre and
  transaction count. Observations are sorted by town and quarter. Missing
  observations are omitted rather than filled with zero or carried forward.
- `is_complete` indicates whether snapshot coverage includes the entire calendar
  quarter. It describes date coverage, not a minimum transaction count or a
  guarantee that all transactions have been reported. Partial starting/ending
  quarters remain visible in the series.
- A town's summary uses its latest complete observed quarter within the display
  window. Metadata identifies towns without observations or without complete
  quarters. The summary labels its quarter and includes the sample size.
- Year-on-year change is `(current_median / prior_year_median - 1) * 100`, using
  the same complete quarter one year earlier. One extra year is queried for
  comparison evidence, so a one-year display can still have a valid YoY value.
  Missing calendar quarters are not replaced by neighbouring observations.
- `yoy_appreciation_pct` is null if the prior-year quarter is missing or
  incomplete, with a machine-readable `yoy_unavailable_reason`. The deterministic
  response template renders this as unavailable rather than zero.
- Metadata includes the snapshot, source, requested and available window,
  coverage limitations and interpretation notes. No schema rebuild is needed.
  These medians describe observed sales and are not a mix-adjusted price index,
  matched-flat appreciation measure or forecast.

## Validation

Tests cover independently calculated quarterly medians, per-transaction price
ratios, same-quarter comparisons, incomplete periods, missing data and direct,
agent and MCP execution. Mock benchmarks explicitly
select mock tools so their fixtures remain reproducible. Category A accuracy on
real data requires a separate labelled evaluation dataset.
