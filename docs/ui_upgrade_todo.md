# UI Upgrade TODO (Step-by-Step)

## Phase 1 - Guided UX

- [x] Add visible 3-step flow for user journey
- [x] Add analysis mode selector (quick, deep risk, compare, voice)
- [x] Add query starter templates
- [x] Add "what this run will execute" explainer

## Phase 2 - Result Clarity

- [x] Add results overview with quality signal
- [x] Add tabbed results: market brief, risk exposure, news context, earnings, voice
- [x] Add fallback/dummy detection warnings
- [x] Add raw pipeline payload expander for debugging

## Phase 3 - Capability Safety

- [x] Detect unavailable optional UI modules (prediction/graphing)
- [x] Show backend `/get_earnings` availability status
- [ ] Add dynamic feature flags from backend health endpoint

## Phase 4 - Voice UX

- [x] Move voice flow into explicit optional step
- [ ] Add in-app recording (not just upload)
- [ ] Show transcript preview before submit
- [ ] Add "retry with same symbols" button for voice errors

## Phase 5 - Polish

- [ ] Improve market brief extraction into structured key findings
- [ ] Add per-context source links and timestamps
- [ ] Add consistent currency formatting across all tabs
- [ ] Add responsive layout checks for mobile
