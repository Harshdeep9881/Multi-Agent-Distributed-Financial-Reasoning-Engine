import importlib
import io
import sys
import asyncio
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.responses import StreamingResponse

from agents.retriever_agent import RetrieverAgent
from data_ingestion.api_agent import APIAgent


def _sample_market_df():
    return pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02"],
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 101.0],
            "Close": [103.0, 104.0],
            "Volume": [1_000_000, 1_200_000],
        }
    )


class FeatureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(APIAgent, "get_market_data", lambda self, *args, **kwargs: {}), patch.object(
            RetrieverAgent, "index_documents", lambda self, docs: None
        ):
            module_name = "orchestrator.orchestrator"
            if module_name in sys.modules:
                del sys.modules[module_name]
            cls.orchestrator = importlib.import_module(module_name)

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    def test_health_endpoint(self):
        payload = self._run(self.orchestrator.health_check())
        self.assertEqual(payload["status"], "healthy")

    def test_retrieve_success_path(self):
        market_data = {"AAPL": _sample_market_df()}
        with patch.object(self.orchestrator.api_agent, "get_market_data", lambda symbols: market_data), patch.object(
            self.orchestrator.api_agent,
            "serialize_market_data",
            lambda data: {"AAPL": data["AAPL"].to_dict(orient="records")},
        ), patch.object(
            self.orchestrator.scraping_agent,
            "scrape_news",
            lambda urls, timeout=15: [{"title": "AAPL Update", "text": "Apple gains", "url": urls[0]}],
        ), patch.object(
            self.orchestrator.retriever_agent,
            "retrieve",
            lambda query, k=3: [{"page_content": "Apple gains"}],
        ):
            payload = self._run(self.orchestrator.retrieve(query="Analyze AAPL", symbols="AAPL"))
        self.assertEqual(payload["symbols"], ["AAPL"])
        self.assertIn("AAPL", payload["market_data"])
        self.assertTrue(payload["context"])

    def test_retrieve_market_data_failure_path(self):
        with patch.object(self.orchestrator.api_agent, "get_market_data", lambda symbols: {}):
            payload = self._run(self.orchestrator.retrieve(query="Analyze BAD", symbols="BAD"))
        self.assertIn("error", payload)
        self.assertEqual(payload["attempted_symbols"], ["BAD"])

    def test_retrieve_fallback_context_path(self):
        market_data = {"AAPL": _sample_market_df()}
        with patch.object(self.orchestrator.api_agent, "get_market_data", lambda symbols: market_data), patch.object(
            self.orchestrator.api_agent,
            "serialize_market_data",
            lambda data: {"AAPL": data["AAPL"].to_dict(orient="records")},
        ), patch.object(
            self.orchestrator.scraping_agent, "scrape_news", lambda urls, timeout=15: []
        ), patch.object(self.orchestrator.retriever_agent, "retrieve", lambda query, k=3: []):
            payload = self._run(self.orchestrator.retrieve(query="Analyze AAPL", symbols="AAPL"))
        self.assertTrue(payload["context"])
        self.assertIn("Current price", payload["context"][0])
        self.assertIn("Average price over period", payload["context"][0])

    def test_analyze_dummy_fallback_values(self):
        captured = {}

        def capture_language(context, exposure, earnings):
            captured["exposure"] = exposure
            captured["earnings"] = earnings
            return "audit-summary"

        with patch.object(self.orchestrator.analysis_agent, "analyze_risk_exposure", lambda market_data: {}), patch.object(
            self.orchestrator.api_agent, "get_earnings", lambda symbol: None
        ), patch.object(self.orchestrator.language_agent, "generate_brief", capture_language):
            payload = self._run(
                self.orchestrator.analyze(
                    {
                        "data": {
                            "symbols": ["AAPL"],
                            "query": "Analyze AAPL",
                            "context": ["sample context"],
                            "market_data": {"AAPL": _sample_market_df().to_dict(orient="records")},
                        }
                    }
                )
            )

        self.assertEqual(payload["summary"], "audit-summary")
        self.assertIn("'price': 100.0", captured["exposure"])
        self.assertIn("2023", captured["earnings"])
        self.assertIn("2024", captured["earnings"])

    def test_analyze_language_fallback_template(self):
        with patch.object(
            self.orchestrator.analysis_agent,
            "analyze_risk_exposure",
            lambda market_data: {"AAPL": {"weight": 0.1, "value": 100000.0, "price": 150.0, "currency": "USD"}},
        ), patch.object(
            self.orchestrator.api_agent,
            "get_earnings",
            lambda symbol: pd.DataFrame({"Year": [2022, 2023], "Earnings": [9.2e9, 10.1e9]}),
        ), patch.object(
            self.orchestrator.language_agent,
            "generate_brief",
            lambda *args: (_ for _ in ()).throw(RuntimeError("LLM failed")),
        ):
            payload = self._run(
                self.orchestrator.analyze(
                    {
                        "data": {
                            "symbols": ["AAPL"],
                            "query": "Analyze AAPL",
                            "context": ["sample context"],
                            "market_data": {"AAPL": _sample_market_df().to_dict(orient="records")},
                        }
                    }
                )
            )

        summary = payload["summary"]
        self.assertIn("Market Brief: Analyze AAPL", summary)
        self.assertIn("Portfolio Analysis for Apple Inc. (AAPL)", summary)

    def test_process_query_audio_pipeline(self):
        class DummyUploadFile:
            def __init__(self):
                self.file = io.BytesIO(b"dummy-audio")

        with patch.object(self.orchestrator.voice_agent, "speech_to_text", lambda audio_file: ""), patch.object(
            self.orchestrator.api_agent, "get_market_data", lambda symbols: {"AAPL": _sample_market_df()}
        ), patch.object(
            self.orchestrator.scraping_agent, "scrape_news", lambda urls: []
        ), patch.object(
            self.orchestrator.retriever_agent, "retrieve", lambda query, k=3: [{"page_content": "AAPL context"}]
        ), patch.object(
            self.orchestrator.analysis_agent,
            "analyze_risk_exposure",
            lambda market_data: {"AAPL": {"weight": 0.1, "value": 100000.0, "price": 150.0, "currency": "USD"}},
        ), patch.object(
            self.orchestrator.api_agent,
            "get_earnings",
            lambda symbol: pd.DataFrame({"Year": [2022, 2023], "Earnings": [9.2e9, 10.1e9]}),
        ), patch.object(
            self.orchestrator.language_agent, "generate_brief", lambda *args: "brief text"
        ), patch.object(
            self.orchestrator.voice_agent, "text_to_speech", lambda text: io.BytesIO(b"mp3-bytes")
        ):
            response = self._run(self.orchestrator.process_query(audio=DummyUploadFile(), symbols="AAPL"))

        self.assertIsInstance(response, StreamingResponse)
        self.assertEqual(response.media_type, "audio/mp3")


if __name__ == "__main__":
    unittest.main()
