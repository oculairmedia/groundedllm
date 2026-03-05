import json
from unittest.mock import patch, MagicMock

from haystack import Document

from pipelines.search_documents.pipeline_wrapper import (
    DEFAULT_RESPONSE_MODE,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
    PipelineWrapper,
    _normalize_response_mode,
    _normalize_top_k,
    _pick_alpha,
    _format_raw_result,
)


class TestNormalizeTopK:
    def test_omitted_uses_default(self):
        value, error = _normalize_top_k(None)
        assert value == DEFAULT_TOP_K
        assert error is None

    def test_coerces_string(self):
        invalid_top_k = json.loads('"3"')
        value, error = _normalize_top_k(invalid_top_k)
        assert value == 3
        assert error is None

    def test_rejects_bool(self):
        value, error = _normalize_top_k(True)
        assert value == DEFAULT_TOP_K
        assert error == "Invalid top_k type: expected integer, got bool"

    def test_rejects_out_of_bounds(self):
        value, error = _normalize_top_k(MAX_TOP_K + 1)
        assert value == DEFAULT_TOP_K
        assert error == f"Invalid top_k value: must be between {MIN_TOP_K} and {MAX_TOP_K}"


class TestNormalizeResponseMode:
    def test_defaults_to_synthesis(self):
        value, error = _normalize_response_mode(None, None)
        assert value == DEFAULT_RESPONSE_MODE
        assert error is None

    def test_legacy_smart_maps_to_synthesis(self):
        value, error = _normalize_response_mode(None, "smart")
        assert value == "synthesis"
        assert error is None

    def test_invalid_mode_rejected(self):
        value, error = _normalize_response_mode("invalid", None)
        assert value is None
        assert error == "Invalid response_mode: expected one of synthesis, raw, both"


class TestRunApiTopKValidation:
    @staticmethod
    def _wrapper() -> PipelineWrapper:
        return PipelineWrapper.__new__(PipelineWrapper)

    def test_run_api_applies_default_when_top_k_omitted(self):
        wrapper = self._wrapper()
        captured: dict[str, int] = {}

        def fake_retrieve(query: str, top_k: int, filename_filter: str | None = None):
            del query, filename_filter
            captured["top_k"] = top_k
            return []

        wrapper._retrieve = fake_retrieve

        response = json.loads(wrapper.run_api(query="deductible details", mode="raw"))
        assert captured["top_k"] == DEFAULT_TOP_K
        assert response["status"] == "ok"
        assert response["total_results"] == 0

    def test_run_api_accepts_string_top_k(self):
        wrapper = self._wrapper()
        captured: dict[str, int] = {}

        def fake_retrieve(query: str, top_k: int, filename_filter: str | None = None):
            del query, filename_filter
            captured["top_k"] = top_k
            return []

        wrapper._retrieve = fake_retrieve

        invalid_top_k = json.loads('"3"')
        response = json.loads(wrapper.run_api(query="deductible details", top_k=invalid_top_k))
        assert response["status"] == "ok"
        assert captured["top_k"] == 3


class TestRunApiResponseMode:
    @staticmethod
    def _wrapper() -> PipelineWrapper:
        return PipelineWrapper.__new__(PipelineWrapper)

    @staticmethod
    def _document() -> Document:
        return Document(
            id="chunk-1",
            content="Policy excludes cosmetic procedures.",
            score=0.91,
            meta={
                "source_filename": "CPAP.pdf",
                "chunk_index": 2,
                "total_chunks": 12,
                "page_number": 8,
                "section_title": "Exclusions",
            },
        )

    def test_raw_mode_returns_verbatim_evidence(self):
        wrapper = self._wrapper()
        wrapper._retrieve = lambda query, top_k, filename_filter=None: [self._document()]

        response = json.loads(wrapper.run_api(query="What exclusions exist?", response_mode="raw"))

        assert response["status"] == "ok"
        assert response["response_mode"] == "raw"
        assert response["results"][0]["chunk_id"] == "chunk-1"
        assert response["results"][0]["content"] == "Policy excludes cosmetic procedures."
        assert response["results"][0]["score"] == 0.91
        assert response["results"][0]["page"] == 8
        assert response["results"][0]["section"] == "Exclusions"

    def test_both_mode_includes_synthesis_and_evidence(self):
        wrapper = self._wrapper()
        wrapper._retrieve = lambda query, top_k, filename_filter=None: [self._document()]
        wrapper._smart_answer = lambda query, documents: "Coverage excludes cosmetic procedures."

        response = json.loads(wrapper.run_api(query="Summarize exclusions", response_mode="both"))

        assert response["status"] == "ok"
        assert response["response_mode"] == "both"
        assert response["answer"] == "Coverage excludes cosmetic procedures."
        assert len(response["evidence"]) == 1
        assert response["evidence"][0]["content"] == "Policy excludes cosmetic procedures."

    def test_default_behavior_is_synthesis(self):
        wrapper = self._wrapper()
        wrapper._retrieve = lambda query, top_k, filename_filter=None: [self._document()]
        wrapper._smart_answer = lambda query, documents: "Synthesis answer"

        response = json.loads(wrapper.run_api(query="Summarize policy"))

        assert response["status"] == "ok"
        assert response["response_mode"] == "synthesis"
        assert response["mode"] == "smart"
        assert response["answer"] == "Synthesis answer"

    def test_run_api_rejects_out_of_bounds_top_k(self):
        wrapper = self._wrapper()
        called = {"retrieve": False}

        def fake_retrieve(query: str, top_k: int, filename_filter: str | None = None):
            del query, top_k, filename_filter
            called["retrieve"] = True
            return []

        wrapper._retrieve = fake_retrieve

        response = json.loads(wrapper.run_api(query="deductible details", top_k=0))
        assert response == {
            "status": "error",
            "detail": f"Invalid top_k value: must be between {MIN_TOP_K} and {MAX_TOP_K}",
        }
        assert called["retrieve"] is False

    def test_score_threshold_filters_low_score_chunks(self):
        wrapper = self._wrapper()
        high_score_doc = Document(
            id="high", content="High score content", score=0.9,
            meta={"source_filename": "test.pdf", "chunk_index": 0, "total_chunks": 2},
        )
        low_score_doc = Document(
            id="low", content="Low score content", score=0.3,
            meta={"source_filename": "test.pdf", "chunk_index": 1, "total_chunks": 2},
        )
        wrapper._retrieve = lambda query, top_k, filename_filter=None: [high_score_doc, low_score_doc]

        response = json.loads(wrapper.run_api(query="test query", response_mode="raw", score_threshold=0.5))

        assert response["status"] == "ok"
        assert response["total_results"] == 1
        assert response["results"][0]["chunk_id"] == "high"

    @patch('pipelines.search_documents.pipeline_wrapper.get_document_store')
    def test_list_documents_returns_inventory(self, mock_get_store):
        wrapper = PipelineWrapper.__new__(PipelineWrapper)
        docs = [
            Document(id="a", content="chunk a", score=0.9,
                meta={"source_filename": "policy.pdf", "total_chunks": 5, "ingested_at": "2025-01-01T00:00:00Z"}),
            Document(id="b", content="chunk b", score=0.8,
                meta={"source_filename": "invoice.pdf", "total_chunks": 3, "ingested_at": "2025-01-02T00:00:00Z"}),
        ]
        mock_store = MagicMock()
        mock_store.filter_documents.return_value = docs
        mock_get_store.return_value = mock_store

        response = json.loads(wrapper.run_api(query="__list_documents__"))

        assert response["status"] == "ok"
        assert response["mode"] == "list_documents"
        assert response["total_documents"] == 2
        filenames = [d["filename"] for d in response["documents"]]
        assert "invoice.pdf" in filenames
        assert "policy.pdf" in filenames

    def test_format_source_includes_page_and_section(self):
        from pipelines.search_documents.pipeline_wrapper import _format_source
        doc = self._document()
        source = _format_source(doc)
        assert source["page"] == 8
        assert source["section"] == "Exclusions"


class TestPickAlpha:
    def test_filename_query_returns_zero(self):
        assert _pick_alpha("CPAP.pdf") == 0.0
        assert _pick_alpha("report.docx") == 0.0

    def test_quoted_query_returns_zero(self):
        assert _pick_alpha('"exact phrase match"') == 0.0

    def test_short_query_returns_low_alpha(self):
        assert _pick_alpha("deductible copay") == 0.3

    def test_long_query_returns_high_alpha(self):
        assert _pick_alpha("what is the deductible and copay for CPAP equipment") == 0.7


class TestFormatRawResult:
    def test_no_text_key_in_result(self):
        doc = Document(id="x", content="some text", score=0.8,
            meta={"source_filename": "test.pdf", "chunk_index": 0, "total_chunks": 1})
        result = _format_raw_result(doc)
        assert "text" not in result
        assert result["content"] == "some text"
