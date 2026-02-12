# Multi-Agent AI Finance Assistant (Streamlit Frontend)

This directory contains the Streamlit frontend for the Multi-Agent AI Finance Assistant.
It connects to a FastAPI backend orchestrator and presents portfolio analysis, risk diagnostics, news context, charts, and forecast outputs.

## Prerequisites

- Python 3.10+
- Streamlit installed in your environment
- FastAPI backend/orchestrator running and reachable

## Current Features

- Global symbol selection (category/search/custom entry)
- User-entered `Portfolio Value (USD)` input
- Query templates and free-text analysis prompt
- Multi-step pipeline progress tracker
- Structured output tabs:
  - `📊 Summary`
  - `⚠️ Risk`
  - `📰 News`
  - `📈 Charts`
  - `🔮 Forecast`
- Agent diagnostics table with status and timing
- Execution timing metrics (total, market, news, AI analysis)
- Deterministic portfolio position table in Summary
- Data quality badges (`Live`, `Fallback`, `Cached`)

## Recent Updates

### 1) Portfolio value is user-driven

- Added sidebar input:
  - `💰 Portfolio Value (USD)`
- The entered value is:
  - included in the query context sent to backend
  - displayed in report context
  - used for deterministic position calculations

### 2) Improved output formatting and consistency

- Query is enriched with formatting instructions:
  - Portfolio Overview
  - Position Details
  - Market Context
  - Risk Assessment
  - Recommendations
- Summary tab includes a deterministic `Portfolio Position Details` table computed from:
  - selected symbols
  - equal-weight assumption (when custom weights are unavailable)
  - user-entered portfolio value

### 3) Agent verification and diagnostics

- Added `🧠 Agents Used` table with:
  - agent name
  - state (`Success`, `Cached`, `Fallback Mode`, `Failed`)
  - timing (`N/A`, `0 ms (not reported)`, or real ms)
- Added warnings for:
  - missing expected agents
  - failed agents

### 4) Timing display improvements

- Timing cards now show `N/A` when a timing key is absent (instead of misleading `0.0s`)
- AI Analysis timing is computed from available components:
  - risk analysis
  - language
  - earnings

### 5) News quality checks

- Coverage details now include:
  - total context items
  - unique items
  - duplicate items
- Added warning when context appears mostly generic/repetitive (fallback-like)

### 6) Forecast clarity improvements

- Forecast tab explicitly warns when advanced prediction modules are disabled.
- In that case, forecast should be interpreted as baseline trend-based output only.

### 7) Risk messaging for single-symbol portfolios

- If only one symbol is selected, 100% concentration is explicitly explained as expected behavior.

### 8) UI tweak

- Increased tab label font size for result tabs for better readability.

## Interpretation Notes

- `Risk` metrics are diagnostic, not trade instructions.
- `Forecast` signals are trend-based and probabilistic.
- If AI narrative values conflict with deterministic table values, use deterministic table values as source of truth.
- Informational output only; not investment advice.

## Local Run

### 1) Start backend (example)

```bash
uvicorn orchestrator.orchestrator:app --host 0.0.0.0 --port 8001
```

### 2) Configure frontend backend URL (optional)

```bash
export API_URL=http://localhost:8001
```

You can also configure `API_URL` via Streamlit secrets.

### 3) Run frontend

From this directory:

```bash
streamlit run app.py
```

Ensure the backend orchestrator is reachable via `API_URL` (default: `http://localhost:8001`).

## Output Quality Checks

- News quality degrades if context is repetitive/generic (`Market Update...` repeated).
- `0 ms (not reported)` for an agent means timing metadata was not provided by backend.
- Forecast is baseline trend-only when advanced prediction modules are disabled.
- Single-symbol runs will show `100%` concentration by design.

## Troubleshooting

- Backend unreachable: start FastAPI orchestrator and verify port.
- Empty/weak news context: backend may be returning fallback or generic context.
- `0 ms` agent timings: backend did not report detailed timings for those agents.
- Forecast marked baseline-only: advanced prediction modules are disabled in this build.
