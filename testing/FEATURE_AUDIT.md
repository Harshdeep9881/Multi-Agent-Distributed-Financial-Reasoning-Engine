# Feature Audit Guide

This audit verifies:

1. Core API feature paths work end-to-end.
2. Fallback behavior is triggered when dependencies fail.
3. Dummy/default values appear as designed when upstream steps fail.

## Scope Covered

- `GET /` health check
- `GET /retrieve/retrieve` success path
- `GET /retrieve/retrieve` market-data failure path
- `GET /retrieve/retrieve` fallback context path (news + retrieval fail)
- `POST /analyze/analyze` fallback dummy exposure + earnings values
- `POST /analyze/analyze` fallback text summary when language generation fails
- `POST /process_query` audio pipeline output path

## What Is Explicitly Audited as Fallback/Dummy

- Fallback exposure price: `100.0`
- Fallback earnings years: `2023`, `2024`
- Fallback analysis summary template path when LLM fails
- Fallback context construction when scraping/retrieval fails

## Run

```bash
venv/bin/python -m unittest -v testing/test_feature_audit.py
```

## Notes

- Tests monkeypatch network/LLM/voice integrations to keep audit deterministic.
- The suite validates the orchestrator backend behavior (`orchestrator/orchestrator.py`).
