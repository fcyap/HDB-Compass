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
- `period_change_pct` compares the latest complete quarter with the earliest
  complete quarter in the display window (`period_start_quarter`), and
  `period_cagr_pct` annualises it. Use these for "over the last N years" questions;
  YoY describes only the most recent year.
- Metadata includes the snapshot, source, requested and available window,
  coverage limitations and interpretation notes. No schema rebuild is needed.
  These medians describe observed sales and are not a mix-adjusted price index,
  matched-flat appreciation measure or forecast.

## Research tools

`mcp_server/market_research_tools.py` answers questions about specific places and
the market as a whole. Every tool that needs a location takes the user's address
text and resolves it with `mcp_server/location_tools.py` (spatial index blocks by
block + street or postal code, MRT/LRT stations, hawker centres, supermarkets and
primary schools; OneMap only as a last resort when credentials are configured).
Ambiguous or unknown addresses return `ambiguous`/`not_found` with candidates, so
the agent asks the user rather than guessing. No tool accepts guessed coordinates;
model tool calls with coordinates that no tool or map pin produced are rejected.

| Tool | Question | Evidence |
|---|---|---|
| `get_block_transaction_history(address \| area \| block_keys, flat_type?, lookback_months=60)` | What has this block (or area) sold for? | Up to 50 latest sales (lease at sale, block and distance for sets) and per-year medians over all sales |
| `get_market_snapshot(towns?, flat_types?)` | Where is the market now? | Latest complete quarter median, price/sqm and volume island-wide and per town, with prior-quarter, 1-, 3- and 5-year same-quarter changes |
| `rank_towns_by_value(flat_type, max_budget?, min_lease_years?, rank_by)` | Which town is best value / cheapest? | Every town's median price, price/sqm, lease and share of sales within budget; low-sample towns listed separately |
| `compare_resale_options(addresses[2-4], flat_type)` | How do these blocks compare? | Per block: sales, median price, price/sqm, area, lease, latest sale and nearest MRT, hawker, supermarket and primary school |
| `assess_resale_price(address, flat_type, asking_price?)` | Is this asking price fair? | Same-block and nearby (radius) price distributions, the asking price's percentile, gap to the median and quartile position |

### Area research: find the blocks, then run set statistics

`mcp_server/area_resale_tools.py` answers "what do flats around X sell for?" by
composition rather than one bespoke query. `find_resale_blocks_near` lists the
blocks in a circle; the set tools then take the same circle or explicit keys.

| Tool | Question | Evidence |
|---|---|---|
| `find_resale_blocks_near(address \| lat/lng, radius_meters=500, flat_type?, lookback_months=12)` | Which blocks are around X, and what does each sell for? | Every resale block within the radius, nearest first: distance, sales, median price, price/sqm, lease, last sale; the named block is flagged |
| `get_area_resale_statistics(area \| block_keys, flat_type?, lookback_months=12)` | What is the price range around X? | Per flat type, pooled over every sale in the set: count, blocks, min/quartiles/max, price/sqm, area, lease; the named block's own figures alongside |
| `get_area_price_trend(area \| block_keys, flat_type, lookback_years=3)` | How have prices around X moved? | Pooled quarterly medians and price/sqm; period change and CAGR between the first and last complete quarters with at least 5 sales |

`area` is `{address | latitude+longitude, radius_meters}`; `block_keys` are
`'BLOCK STREET'` keys, e.g. from `find_resale_blocks_near` or blocks picked on the
map. Distances are great-circle from the resolved place to each block's geocode,
identical to `query_comparable_resale_transactions`. Set statistics are computed
in SQL over the pooled sales, never as a median of block medians, so a model never
aggregates numbers itself. Small areas are noisy: thin quarters are shown but not
used as trend endpoints, and price/sqm is reported beside price because the mix of
flat sizes sold varies by quarter.

Semantics shared with the queries above: windows end at the snapshot's last
complete month; snapshot changes compare quarters only within source coverage
(otherwise null); lease is estimated at the window end except in block history.
Results are historical transactions, not listings, valuations or forecasts.
`query_resale_blocks_by_criteria` also accepts optional `block_keys` so the
supervisor can price blocks found by spatial tools (school, commute) before
applying a budget; blocks without price evidence are never treated as within budget.

## BTO launch data

`data/bto/bto_launches.csv` is transcribed from HDB's official launch annexes
(Table A(1)(a) supply and pricing, and HDB's "resale comparables nearby" tables),
currently the February 2026 and June 2026 exercises. Each row keeps the annex URL.
Prices are indicative ranges for 99-year leases before grants; for projects whose
prices include floor finishes, the annex also quotes lower prices without them,
which are not stored. Towns come from HDB's launch announcements.

`compare_bto_launch_vs_resale(bto_project_name, flat_type="4 ROOM")` returns the
official range, classification, units, waiting time, HDB's quoted nearby resale
comparables and the town-wide resale median for the same flat type over the last
12 months. `price_gap_pct` compares the range midpoint with that median; resale
flats differ in age, lease and size, which the response states. Unknown projects
return `not_found` with the covered list; unoffered flat types return
`flat_type_not_offered`. To add a launch, append rows from its annex; never estimate.

## Validation

Tests cover independently calculated quarterly medians, per-transaction price
ratios, same-quarter comparisons, incomplete periods, missing data and direct,
agent and MCP execution. Mock benchmarks explicitly
select mock tools so their fixtures remain reproducible. Category A accuracy on
real data requires a separate labelled evaluation dataset.
