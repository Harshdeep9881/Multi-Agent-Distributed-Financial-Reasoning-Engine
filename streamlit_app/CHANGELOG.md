# Changelog

All notable frontend changes for this `streamlit_app` module are documented here.

## 2026-02-12

### Added

- Sidebar `Portfolio Value (USD)` user input.
- Deterministic portfolio position table in Summary tab.
- Agent diagnostics table (`Agent`, `State`, `Time`).
- Missing-agent and failed-agent warnings.
- News coverage quality stats (`total`, `unique`, `duplicate`).
- Generic/repetitive context warning in News tab.
- Single-symbol concentration explanatory note in Risk tab.

### Changed

- Backend query context now includes:
  - user-entered portfolio value
  - structured formatting guidance for detailed report output
- Timing cards now show `N/A` for absent values instead of defaulting to `0.0s`.
- Forecast tab warning now clearly states baseline trend-only behavior when advanced modules are disabled.
- Result tab label font size increased for readability.

### Notes

- Concentration can be `100%` by design when one symbol is selected.
- Deterministic table values should be treated as authoritative if narrative text is inconsistent.

