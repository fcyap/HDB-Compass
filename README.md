# HDB-Compass: Multi-Agent & FastMCP Decision-Support System for Singapore Public Housing

[![Course](https://img.shields.io/badge/SMU%20IS469-Gen%20AI%20with%20LLMs-blue.svg)](https://scis.smu.edu.sg)
[![FastMCP](https://img.shields.io/badge/FastMCP-Model%20Context%20Protocol-orange.svg)](https://github.com/jlowin/fastmcp)
[![LangGraph](https://img.shields.io/badge/LangGraph-Cyclic%20State%20Machine-green.svg)](https://langchain-ai.github.io/langgraph/)
[![MapLibre](https://img.shields.io/badge/MapLibre%20GL-Interactive%20Maps-blueviolet.svg)](https://maplibre.org/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://python.org)

> **HDB-Compass** is an agentic decision-support system designed to help Singaporeans navigate the high-stakes decision of buying public housing by combining multi-agent orchestration, FastMCP tool integration, and interactive geospatial intelligence.

---

## Member 2 Ownership & Deliverables (Categories B & E)

As specified in the **HDB-Compass Project Proposal** (Pages 8, 13, and 14):

- **Workstream:** Member 2 — Geospatial Engine & Interactive Map Sync (Categories B & E).
- **Core Engine:** Singapore SVY21 planar metric coordinate projection (`EPSG:3414`), metric buffer polygon generator, and $O(\log N)$ KD-Tree radius filters.
- **Data Ingestion Layer:** 100% dynamic ingestion from official Singapore public government APIs (`data.gov.sg` and OneMap API), with **zero hardcoded mock data**.
- **Geospatial & Commute Agent:** Cyclic LangGraph state machine streaming spatial reasoning, candidate blocks, and POI markers.
- **FastMCP Tool Server:** Implements the 4 official proposal tool signatures for Category B lookups.
- **Split-Screen Web App:** Responsive React/Vite web application with MapLibre GL live state synchronization.

---

## Repository Structure

The repository follows the project proposal structure (Page 14):

```
HDB-Compass/
├── README.md                        # Project overview and local setup
├── requirements.txt                 # Python dependencies (langgraph, fastmcp, scipy, pyproj, etc.)
├── .env.example                     # Environment configuration template
├── .gitignore                       # Clean gitignore excluding node_modules, .env, and large cache
│
├── config/                          # Experiment configuration (Proposal Page 14)
│   └── experiment_config.yaml       # Models, temp=0.0, seed=42, chunking, top-k
│
├── data/                            # Market ingestion and spatial data
│   ├── build_market_db.py            # Download, validate and build the resale database
│   ├── market_database.py            # Database settings and read-only connections
│   ├── gov_download.py               # Government dataset download and cache
│   ├── sql/market_schema.sql         # DuckDB schema
│   ├── build_spatial_index.py       # 100% dynamic API pipeline from data.gov.sg & OneMap
│   ├── spatial_index.pkl            # Precomputed 2D KD-Tree indices (13,426 blocks, 179 schools, 806 amenities)
│   ├── spatial_index.json           # Compact metadata summary
│   └── geocoded_cache.json          # Cached postal code coordinates
│
├── mcp_server/                      # FastMCP tools
│   ├── server.py                    # FastMCP server entry point
│   ├── market_sql_tools.py          # Resale queries against local DuckDB
│   └── spatial_onemap_tools.py      # Geospatial & commute tools
│
├── agents/                          # Multi-Agent Orchestration Layer
│   ├── state.py                     # Shared HDBCompassState & UserConstraints (Proposal Page 7)
│   ├── arch_3_specialist_5agent.py  # 5-Agent Architecture Integration
│   └── spatial_agent.py             # Geospatial & Commute Agent (LangGraph cyclic graph)
│
├── backend/                         # REST API & WebSocket Server
│   └── main.py                      # FastAPI server with /ws real-time agent stream
│
├── frontend/                        # Split-Screen Web Application (Member 2)
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx                  # Main split-screen view
│       ├── components/
│       │   ├── AgentChat.tsx        # Conversational UI with streaming reasoning steps
│       │   └── InteractiveMap.tsx   # MapLibre GL map with SVY21 buffer circle & interactive popups
│       └── index.css                # Premium glassmorphic theme
│
└── tests/                           # Verification & Regression Test Suites
    ├── test_market_data.py          # Ingestion, resale queries, agent and MCP integration
    ├── test_market_download.py      # Download authentication and cache handling
    ├── test_spatial_engine.py       # SVY21 coordinate roundtrip & KD-Tree queries
    └── test_spatial_tools.py        # All 5 FastMCP tools regression suite
```

---

## Official Government Data Sources

All data ingested by [`data/build_spatial_index.py`](data/build_spatial_index.py) is pulled dynamically from official public government APIs:

| Category | Official Government Dataset | Source & Dataset ID | Indexed Count |
| :--- | :--- | :--- | :--- |
| **HDB Buildings** | HDB Existing Building Footprints | data.gov.sg [`d_16b157c52ed637edd6ba1232e026258d`](https://data.gov.sg) | **13,426 blocks** |
| **HDB Property Info** | HDB Property Information | data.gov.sg [`d_17f5382f26140b1fdae0ba2ef6239d2f`](https://data.gov.sg) | **13,426 records** |
| **Primary Schools** | MOE General Information of Schools | data.gov.sg [`d_688b934f82c1059ed0a6993d2a829089`](https://data.gov.sg) | **179 schools** |
| **Rail Stations** | URA Master Plan Rail Station Layer | data.gov.sg [`d_8d886e3a83934d7447acdf5bc6959999`](https://data.gov.sg) | **249 MRT/LRT** |
| **Hawker Centres** | NEA List of Government Markets & Hawker Centres | data.gov.sg [`d_68a42f09f350881996d83f9cd73ab02f`](https://data.gov.sg) | **104 hawkers** |
| **Supermarkets** | NEA List of Supermarket Licences | data.gov.sg [`d_11edd0117280c5776651d7891114c88c`](https://data.gov.sg) | **453 supermarkets** |
| **Geocoding & Transit** | OneMap Elastic Search & Routing API | OneMap API (`/api/common/elastic/search`) | Island-wide |

---

## Official FastMCP Tool Signatures (Page 8)

Implemented in [`mcp_server/spatial_onemap_tools.py`](mcp_server/spatial_onemap_tools.py):

```python
# 1. MOE 1km / 1-2km Priority Admission Boundary Filter
verify_primary_school_distance(school_name: str, bracket: str = "1km") -> List[str]

# 2. Multi-layer Amenity Proximity Filter
filter_blocks_by_amenity_proximity(towns: List[str], amenity_constraints: Dict[str, int]) -> List[Dict]

# 3. Peak-Hour Dual-Transit Commute Filter
compute_dual_transit_commute(candidate_blocks: List[str], work_dest_a: str, work_dest_b: str, max_minutes: int) -> List[Dict]

# 4. Interactive Map UI State Sync
sync_interactive_map_ui(candidate_blocks: List[str], buffer_radii: Dict[str, int], center_latlon: List[float]) -> Dict
```

---

## Setup

Run commands from the repository root. Use Python 3.11 or 3.12; if your default
`python` is older, use the appropriate interpreter to create the environment.

### 1. Install dependencies

```bash
python -m venv .venv
```

Activate on macOS, Linux or WSL:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies in the activated environment:

```bash
python -m pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env`. If it already exists, add the required settings
instead of replacing it. For local data and tool development:

```dotenv
DATA_GOV_SG_API_KEY=
HDB_COMPASS_DB_PATH=data/local/hdb_compass.duckdb
LLM_PROFILE=none
LANGSMITH_TRACING=false
```

| Setting | Purpose |
|---|---|
| `DATA_GOV_SG_API_KEY` | Optional; leave blank for public resale downloads |
| `HDB_COMPASS_DB_PATH` | Database location, relative to the repository root unless absolute |
| `LLM_PROFILE` | `none` for deterministic agents; `ollama` or `vllm` for model inference |
| `ONEMAP_EMAIL`, `ONEMAP_PASSWORD` | Live geocoding and spatial-data rebuilds |
| `LANGSMITH_API_KEY` | Optional tracing when enabled |

Environment variables override `.env`. Credentials, cached downloads and
generated databases are ignored by Git.

### 3. Build the market database

```bash
python -m data.build_market_db
```

The command downloads and validates the 2017-onward resale dataset, builds
DuckDB, and reuses `data/spatial_index.pkl` for coordinates. It requires neither
OneMap nor a running model. Each developer generates the database locally.

Subsequent builds reuse a checksum-verified cache. To download fresh data:

```bash
python -m data.build_market_db --refresh
```

Stop database readers before refreshing, particularly on Windows. Failed builds
leave the previous database intact. For an offline build, use
`python -m data.build_market_db --csv /path/to/resale.csv`; see `--help` for
alternate database paths and coordinate-free builds.

The [market-data reference](docs/MARKET_DATA.md) describes the DuckDB schema,
provenance and query calculations.

### 4. Verify tools and agents

```bash
python -m pytest tests/ -q
```

Tests use synthetic/mock fixtures and require no downloaded market database,
API credentials or running model. They include agent and in-memory MCP calls.

After building the database, with `LLM_PROFILE=none` in `.env`:

```bash
python -c 'from agents.arch_3_specialist_5agent import run; print(run("Find 4-room flats in Bishan under 850k with at least 60 years lease.")["draft_response"])'
```

The graph uses real resale queries, comparable-sale analytics, town price trends and financial calculators
through the Python tool registry. BTO comparisons and policy retrieval still
use mocks; configuration is in `config/experiment_config.yaml`.

The financial tools need no API key or database. They return missing-input lists
when household, eligibility or funding facts are unknown. See
[financial contracts and assumptions](docs/FINANCIAL_TOOLS.md). To test them:

```bash
python -m pytest tests/test_financial_tools.py -q
python -c 'from mcp_server.financial_calc_tools import calculate_cpf_housing_grants; print(calculate_cpf_housing_grants(monthly_income=8000))'
```

That call intentionally returns `needs_information`. A complete conditional
grant example, using explicitly confirmed sample facts:

```python
from mcp_server.financial_calc_tools import calculate_cpf_housing_grants

result = calculate_cpf_housing_grants(
    citizenship_pair="SC_SC", first_timer_status=True, monthly_income=4400,
    flat_type="4 ROOM", youngest_buyer_age=30, youngest_core_member_age=30,
    remaining_lease_years=70, eligibility_confirmed=True,
    continuous_employment_12_months=True, proximity="none",
)
print(result["total_grants"])  # 150000.0
```

### 5. Run the MCP server

```bash
python -m mcp_server.server
```

An MCP client normally launches this stdio process and communicates over
stdin/stdout. It exposes three market tools, three financial tools and four spatial tools. The
current agent registry executes Python directly and does not need this process.

### 6. Run the web application (optional)

The web backend currently streams the standalone geospatial agent. It does not
yet run the full specialist graph or market query.

Start the backend in an activated Python environment:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

In another terminal, with Node.js and npm installed:

```bash
cd frontend
npm install
npm run dev
```

Open [localhost:3000](http://localhost:3000). Backend health is available at
[localhost:8000/health](http://localhost:8000/health). Live geocoding needs OneMap
credentials; querying the existing spatial index does not.

To rebuild spatial data, configure OneMap credentials and run
`python data/build_spatial_index.py`. Rebuild the market database afterwards to
pick up coordinate changes.

### Troubleshooting

| Problem | Action |
|---|---|
| Missing market database | Run the build command and check `HDB_COMPASS_DB_PATH` |
| Download fails or API returns 429 | Retry later; optionally configure a data.gov.sg key |
| Cached checksum differs | Rebuild with `--refresh` |
| CSV validation fails | Inspect the reported source rows for a source-format change |
| Database replacement fails | Stop database readers before rebuilding |
| Smoke run attempts a model connection | Set `LLM_PROFILE=none`; check for an overriding shell variable |
| Frontend is disconnected | Start the backend on port 8000 |

---

## TODO: Member 1 — Market and financial tools (Categories A & D)

Completed: resale ingestion, the local DuckDB schema, three resale analytics
tools and three financial calculators, with Pydantic validation, agent-registry
integration and FastMCP registration. Six of seven Member 1 tools are real;
`compare_bto_launch_vs_resale` remains mocked. Financial scenario coverage and
exclusions are documented in [Financial tools](docs/FINANCIAL_TOOLS.md).

### Tool checklist

- [x] `query_comparable_resale_transactions`: match sales around verified
  coordinates by flat type and optional area, lease, storey and historical date
  filters. Return price/PSM distributions, source transactions and sample-quality
  information. Statistics use the full matched sample; displayed examples are limited.
- [x] `compute_town_price_timeseries`: calculate quarterly medians, price per
  square metre and year-on-year changes from the existing resale database;
  handle incomplete quarters and missing comparison periods explicitly.
- [x] `calculate_cpf_housing_grants`: implement supported household scenarios
  using verified, versioned eligibility rules and grant tables.
- [x] `calculate_maximum_affordability`: calculate borrowing and purchase limits
  from income, available funds, debts and the applicable loan assumptions.
- [x] `calculate_bsd_and_upfront_outlay`: calculate stamp duty and upfront
  CPF/cash requirements using price, valuation, financing, grants and fees.
- [ ] `compare_bto_launch_vs_resale`: retrieve validated BTO details on demand,
  reuse comparable resale analytics and calculate price differences by flat type
  across the published price range. No complete BTO database is required.

### Try comparable resale analytics

Build the local market database using the setup instructions above, then run:

```python
from mcp_server.comparable_resale_tools import query_comparable_resale_transactions

result = query_comparable_resale_transactions(
    latitude=1.35, longitude=103.85, flat_type="4 ROOM",  # Example coordinates.
    radius_meters=500, lookback_months=12,
    min_floor_area_sqm=85, max_floor_area_sqm=110,
    min_sample_size=5, sample_limit=10,
)
print(result["status"])
print(result["summary"])
print(result["metadata"])
```

Use verified project/property coordinates for a real comparison. No geocoding,
API key or LLM is needed for this query. `end_month="YYYY-MM"` selects a historical
comparison window; otherwise the snapshot's last completed analysis month is used.
Results are historical sales, not current listings or an official valuation.
See [matching and statistical semantics](docs/MARKET_DATA.md#comparable-resale-transactions).

Run the isolated tests without a downloaded database:

```bash
python -m pytest tests/test_comparable_resale.py -q
```

### Next steps for BTO comparison

- [ ] Define a validated BTO project schema: prices before grants by flat type,
  areas, application dates/status, classification, completion estimates, location
  precision, source URLs and retrieval dates. Unpublished facts remain unknown.
- [ ] Connect official-source search/page/PDF reading to that schema for named,
  open, upcoming and closed launches, with a small evidence cache where useful.
- [ ] Coordinate project location resolution with Member 2's spatial interface;
  do not substitute a town centre or silently use approximate coordinates.
- [ ] Coordinate intent and session handling with Member 4: apply now, wait for a
  future launch or compare options; capture the acceptable move-in date and
  willingness to wait for completion separately from willingness to apply.
- [ ] Extend financial calculators for supported BTO grants and payment stages;
  retrieve relevant eligibility/restriction evidence through Member 3's policy
  interface. Existing financial calculations are scoped to resale.
- [ ] Test the complete project-retrieval → location → comparable-sales →
  financial-comparison flow, including missing prices, ambiguous locations,
  closed applications and insufficient resale samples.

### Contracts, integration and verification

- [x] Implement financial input/output schemas, including household details,
  buyer ages, property lease, valuation and funding inputs. Return missing-input
  or unsupported-scenario results explicitly.
- [ ] Review the expanded financial contracts and specialist prompts with Member 4.
- [ ] Review the additional comparable-sale tool, coordinate handoff and updated
  12-tool inventory with Member 4; run live-model tool-selection checks as well
  as deterministic graph/MCP tests.
- [x] Record financial rule sources, known effective dates, assumptions and rounding
  rules. Define how documentation changes are checked and released as tested
  rule versions.
- [x] Implement the financial call sequence: pass confirmed grants and known
  funding inputs between tools, require a positive candidate price for upfront
  costs and include grants once in affordability.
- [x] Validate inputs and outputs, register each completed tool with FastMCP,
  switch its configuration to `real`, and verify direct, agent and MCP calls.
- [x] Add independent financial reference cases, boundary and integration tests.
- [ ] Produce Category A & D accuracy results with Member 5 using fixed data/rule versions.
  Existing unit tests and mock benchmarks do not establish that accuracy.
- [ ] Document the MCP client–server flow and explain how SQL/Python calculations
  supply numeric evidence to the LLM for the role presentation.

### Reliability improvements

After the original deliverables above, prioritize:

- [ ] Expose source coverage, download date and stale-data status consistently
  across market tools, with freshness thresholds based on each source's update
  cadence. Distinguish an old snapshot from delayed source reporting.
- [ ] Validate refreshes before replacing the active snapshot: detect schema
  changes, missing months, rejected rows and unusual changes in coverage or
  values. Preserve the last valid snapshot when validation fails.
- [ ] Coordinate a financial clarification flow with Member 4: collect missing
  household, eligibility and funding facts and retain confirmed answers across
  chat turns without inventing defaults.
- [ ] Assess funding for a selected candidate using its price, actual valuation,
  approved or explicitly modelled loan, taxes and fees. Report cash shortfalls;
  do not treat the purchase-only ceiling or a block's median lease as proof
  that a particular purchase is affordable or eligible.
- [ ] Extend Category A/D evaluation with independently checked answers under
  fixed snapshots and rule versions, including stale data, insufficient samples,
  missing inputs and unsupported scenarios.
- [ ] Add pagination or continuation for truncated market results and consistent
  errors for no matches, insufficient observations, a missing database and
  unavailable sources. Ensure agents disclose incomplete result coverage.
- [ ] Define and test a stale/missing-evidence fallback policy: refresh official
  structured data first; use official documentation or search for policy
  clarification; retain source URLs and dates. Search results must not silently
  replace verified calculator rules or missing numeric evidence.

### Potential analytics expansion

Comparable analytics are implemented; the remaining items are follow-up capabilities:

- [x] Add comparable-sale price distributions and sample-quality information:
  quartiles, price ranges, transaction counts and low-sample flags alongside medians.
- [x] Add comparable-transaction analysis using recent sales matched by location,
  flat type, floor area, remaining lease and storey range. Return matching
  criteria, dates, sample sizes and limitations; comparisons are not an official
  valuation or evidence of currently available listings.
- [ ] Add side-by-side town comparisons for prices, price per square metre,
  lease profiles, transaction activity and historical changes using consistent
  periods and filters.
- [ ] Add deterministic financial scenario comparisons for interest rates,
  tenures, confirmed grants and available funds, showing repayment differences
  and cash shortfalls under explicit assumptions.
- [ ] Add trend composition checks to distinguish changes in the mix of flats
  sold from like-for-like price changes. Avoid describing raw median growth as
  appreciation of an individual flat.

Extend existing tool outputs where practical. Agree any additional callable
tools and contract changes with Member 4, updating schemas, prompts and
evaluation expectations together. The inventory now contains 12 tools: the
original 11 plus comparable resale analytics. Earlier 11-tool benchmark runs
must remain labelled with their original inventory and configuration.

Next: the BTO project schema and on-demand retrieval, contract review and
Category A/D evaluation, followed by
reliability improvements and selected analytics expansions. Broader financial
scenarios remain future work. Complete financial estimates require the household,
eligibility and funding facts described in the financial contracts.

---

## Member 4 — Multi-Agent Orchestration & Architecture Study (Category E)

Owns the LangGraph state machine, Agent 1 (Supervisor & Constraint Negotiator, incl. zero-result
relaxation) and the three benchmark architectures of Experiment 1. Contract for every other
workstream: [`docs/INTERFACES.md`](docs/INTERFACES.md). Graph drawings: [`docs/architecture_graphs.md`](docs/architecture_graphs.md).

| File | What it is |
|---|---|
| `agents/state.py` | `HDBCompassState` (proposal §4.2) + `tool_call_log`, `relaxation`, `variant`, `errors` |
| `agents/schemas.py` | Pydantic I/O models for all 12 tools; shared tool inventory |
| `agents/tool_registry.py` | mock/real switch per tool (`config → tools.use_mock`), strict argument validation, tool-call logging |
| `agents/llm.py` | one `get_llm()` for every node; profiles `ollama` / `vllm` / `none` |
| `agents/supervisor.py` | extraction → routing → parallel `Send` fan-out → specialist ReAct loops → SQL ∩ spatial intersection → relaxation (E2) → synthesis + map sync (E1) → auditor hook → finalize |
| `agents/arch_1_monolith.py` | Variant A: one ReAct agent, 12 tools, no auditor |
| `agents/arch_2_coarse.py` | Variant B: supervisor + combined data agent (11 tools) + policy agent (1 tool), no auditor |
| `agents/arch_3_specialist_5agent.py` | Variant C: proposed 5-agent graph + auditor self-correction loop |
| `mcp_server/mock_tools.py` | deterministic fixtures for Members 1 & 3's tools until the real ones land |
| `evaluation/category_e_queries.json` | 15 Category E golden queries with expected routing/tool calls (`build_category_e.py`) |
| `evaluation/architecture_metrics.py` | Tool Selection F1, Parameter Exact-Match, CSR, routing accuracy, aggregation |
| `cluster/` | vLLM Slurm job, SSH tunnel and smoke test for the SCIS GPU cluster |

```bash
# deterministic tests: mock fixtures are isolated from the real market database
LLM_PROFILE=none .venv/bin/pytest tests/ -q

# run one variant end to end (ollama profile by default; see config/experiment_config.yaml)
.venv/bin/python agents/arch_3_specialist_5agent.py
```

Local Ollama note: the default 4096-token context is too small for Variant A (12 tool schemas). Quit the Ollama
menu-bar app and start the server with a larger context before benchmarking locally:

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```
