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
├── README.md                        # Project documentation & setup guide
├── requirements.txt                 # Python dependencies (langgraph, fastmcp, scipy, pyproj, etc.)
├── .env.example                     # OneMap API credentials template
├── .gitignore                       # Clean gitignore excluding node_modules, .env, and large cache
│
├── config/                          # Experiment configuration (Proposal Page 14)
│   └── experiment_config.yaml       # Models, temp=0.0, seed=42, chunking, top-k
│
├── data/                            # Data Ingestion & Spatial Precomputations (Member 2)
│   ├── build_spatial_index.py       # 100% dynamic API pipeline from data.gov.sg & OneMap
│   ├── spatial_index.pkl            # Precomputed 2D KD-Tree indices (13,426 blocks, 179 schools, 806 amenities)
│   ├── spatial_index.json           # Compact metadata summary
│   └── geocoded_cache.json          # Cached postal code coordinates
│
├── mcp_server/                      # FastMCP Tool Server (Member 2)
│   ├── server.py                    # FastMCP server entry point
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

## Quickstart Guide

### 1. Environment Setup

Create and activate a virtual environment, then install Python dependencies:

```bash
# Clone the repository
git clone https://github.com/fcyap/HDB-Compass.git
cd HDB-Compass

# Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and provide your free [OneMap API](https://www.onemap.gov.sg/apidocs/) credentials:

```ini
ONEMAP_EMAIL=your_email@example.com
ONEMAP_PASSWORD=your_password
```

### 2. Verify Spatial Engine & FastMCP Tools

Run the automated regression suites:

```bash
# Verify spatial coordinate math, metric buffer generation & KD-Tree queries
python tests/test_spatial_engine.py

# Verify all FastMCP tools
python tests/test_spatial_tools.py
```

### 3. (Optional) Rebuild Dynamic Geospatial Index

To pull the latest official datasets from `data.gov.sg` and rebuild the spatial index:

```bash
python data/build_spatial_index.py
```

### 4. Run the Full Application (Backend + Frontend)

**Terminal 1 — Backend API & WebSocket Server:**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Frontend Split-Screen UI:**
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000/](http://localhost:3000/) in your browser to interact with the split-screen chat and map!
