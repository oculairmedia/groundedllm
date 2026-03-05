"""
Search Documents Pipeline

Queries the shared Weaviate document store for relevant chunks.
This pipeline is exposed as an MCP tool so any Letta agent can search
documents that were ingested via the ingest_document pipeline.

Modes:
  synthesis (default) — retrieves chunks, feeds them to an LLM with a citation
                         prompt, and returns a synthesized answer with source pointers.
  raw                 — returns the raw chunks as-is.
  both                — returns both the synthesis and raw evidence.

Flow (synthesis):
  query → WeaviateHybridRetriever (BM25 + vector) → PromptBuilder → OpenAIGenerator → answer + sources
Flow (raw):
  query → WeaviateHybridRetriever (BM25 + vector) → formatted JSON
"""

import functools
import json
import os
import re
from typing import List, Optional, Union

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.weaviate import WeaviateHybridRetriever
from loguru import logger as log

from resources.docstore import get_document_store
from resources.retry import run_with_weaviate_retry
from resources.utils import read_resource_file

DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 25
DEFAULT_RESPONSE_MODE = "synthesis"




# ---------------------------------------------------------------------------
# Retrieval pipeline (shared by both modes)
# ---------------------------------------------------------------------------

def create_retrieval_pipeline() -> Pipeline:
    """Build a pipeline that retrieves matching chunks using hybrid (BM25 + vector) search.

    The embedder generates the query vector for the similarity component.
    The hybrid retriever combines BM25 (keyword) + vector (semantic) results.
    """
    embedding_model = os.getenv("HAYHOOKS_EMBEDDING_MODEL", "gemini/text-embedding-004")
    api_base = os.getenv("HAYHOOKS_EMBEDDING_API_BASE", "http://100.81.139.20:11450/v1")

    embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=api_base,
        model=embedding_model,
    )

    retriever = WeaviateHybridRetriever(
        document_store=get_document_store(),
        top_k=DEFAULT_TOP_K,
    )

    pipe = Pipeline()
    pipe.add_component("embedder", embedder)
    pipe.add_component("retriever", retriever)
    pipe.connect("embedder.embedding", "retriever.query_embedding")

    return pipe


# ---------------------------------------------------------------------------
# Smart-mode LLM pipeline (prompt + generator)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_prompt_template() -> str:
    """Read and cache the prompt template at module level."""
    return read_resource_file("document_search_prompt.md")


def create_smart_pipeline() -> Pipeline:
    """Build a pipeline that takes retrieved docs + query and produces an LLM answer."""
    prompt_builder = PromptBuilder(
        template=_get_prompt_template(),
        required_variables=["query", "documents"],
    )

    search_model = os.getenv("HAYHOOKS_SEARCH_MODEL")
    if not search_model:
        raise ValueError("HAYHOOKS_SEARCH_MODEL environment variable is not set!")

    api_base_url = os.getenv("OPENAI_API_BASE")
    if not api_base_url:
        raise ValueError("OPENAI_API_BASE environment variable is not set!")

    log.info(f"Smart search using model: {search_model}")
    llm = OpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=api_base_url,
        model=search_model,
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder", "llm")

    return pipe


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

def _format_source(doc: Document) -> dict:
    """Build a source pointer dict from a retrieved document."""
    source = {
        "filename": doc.meta.get("source_filename", "unknown"),
        "chunk": doc.meta.get("chunk_index"),
        "of": doc.meta.get("total_chunks"),
        "score": round(doc.score, 3) if doc.score is not None else None,
    }
    # Include page and section metadata when available
    page = doc.meta.get("page_number") or doc.meta.get("page")
    section = doc.meta.get("section_title") or doc.meta.get("section")
    if page is not None:
        source["page"] = page
    if section:
        source["section"] = section
    return source


def _format_raw_result(doc: Document) -> dict:
    chunk_id = doc.meta.get("chunk_id")
    if chunk_id is None:
        chunk_id = doc.id or f"{doc.meta.get('source_filename', 'unknown')}:{doc.meta.get('chunk_index')}"

    return {
        "chunk_id": chunk_id,
        "content": doc.content,
        "score": doc.score,
        "filename": doc.meta.get("source_filename", "unknown"),
        "chunk_index": doc.meta.get("chunk_index"),
        "total_chunks": doc.meta.get("total_chunks"),
        "page": doc.meta.get("page_number") or doc.meta.get("page"),
        "section": doc.meta.get("section_title") or doc.meta.get("section"),
        "room_id": doc.meta.get("source_room_id", ""),
        "sender": doc.meta.get("source_sender", ""),
        "ingested_at": doc.meta.get("ingested_at", ""),
    }


def _normalize_top_k(top_k: Optional[Union[int, str]]) -> tuple[int, Optional[str]]:
    """Normalize and validate top_k at the tool boundary.

    Returns
    -------
    tuple[int, Optional[str]]
        (normalized_value, validation_error)
    """
    if top_k is None:
        return DEFAULT_TOP_K, None

    # Bool check must come before str/int checks because bool is a subclass of int
    if isinstance(top_k, bool):
        return DEFAULT_TOP_K, f"Invalid top_k type: expected integer, got {type(top_k).__name__}"

    # Coerce string to int if possible (Phase 1 fix)
    if isinstance(top_k, str):
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            return DEFAULT_TOP_K, f"Invalid top_k: expected integer, got '{top_k}'"

    if not isinstance(top_k, int):
        return DEFAULT_TOP_K, f"Invalid top_k type: expected integer, got {type(top_k).__name__}"

    if top_k < MIN_TOP_K or top_k > MAX_TOP_K:
        return DEFAULT_TOP_K, f"Invalid top_k value: must be between {MIN_TOP_K} and {MAX_TOP_K}"

    return top_k, None


def _normalize_response_mode(response_mode: Optional[str], legacy_mode: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    requested_mode = response_mode if response_mode is not None else legacy_mode
    if requested_mode is None:
        return DEFAULT_RESPONSE_MODE, None

    mode_normalized = requested_mode.strip().lower()
    alias_map = {
        "smart": "synthesis",
        "synthesis": "synthesis",
        "raw": "raw",
        "both": "both",
    }
    resolved = alias_map.get(mode_normalized)
    if resolved is None:
        return None, "Invalid response_mode: expected one of synthesis, raw, both"
    return resolved, None


def _pick_alpha(query: str) -> float:
    """Dynamically choose BM25/vector balance based on query intent.

    Returns
    -------
    float
        0.0 = pure BM25 (keyword), 1.0 = pure vector (semantic).
    """
    # Exact filename lookups — pure keyword
    if re.match(r'^[\w.\-]+\.(pdf|docx|txt|xlsx|csv)$', query, re.IGNORECASE):
        return 0.0
    # Quoted strings — user wants exact match
    if query.startswith('"') and query.endswith('"'):
        return 0.0
    # Very short queries (1-3 words) — favor keywords
    if len(query.split()) <= 3:
        return 0.3
    # Default: favor vector similarity for natural language questions
    return 0.7


# ---------------------------------------------------------------------------
# Pipeline wrapper (Hayhooks entry point)
# ---------------------------------------------------------------------------

class PipelineWrapper(BasePipelineWrapper):
    """Search the shared document store for relevant content.

    Supports two modes:
      - **smart** (default): retrieves chunks and uses an LLM to synthesize
        a concise answer with source citations.
      - **raw**: returns the raw chunks with metadata (original behaviour).
    """

    def setup(self) -> None:
        self.retrieval_pipeline = create_retrieval_pipeline()
        self.smart_pipeline = create_smart_pipeline()
        # Pre-warm: force connection + JIT compilation on startup
        try:
            self._retrieve("warmup initialization test for document search pipeline", top_k=1)
            log.info("Search pipeline pre-warmed successfully")
        except Exception:
            log.warning("Search pipeline pre-warm failed (will retry on first real query)")

    def _retrieve(
        self,
        query: str,
        top_k: int,
        filename_filter: Optional[str] = None,
    ) -> List[Document]:
        """Run the hybrid retrieval pipeline, with stale-connection retry."""
        # Build native Weaviate filters if filename_filter is provided
        filters = None
        if filename_filter:
            filters = {
                "field": "source_filename",
                "operator": "==",
                "value": filename_filter,
            }

        # Dynamic alpha: skip embedding for keyword-type queries
        alpha = _pick_alpha(query)

        def _do_retrieve():
            run_params = {
                "retriever": {
                    "query": query,
                    "top_k": top_k,
                    "filters": filters,
                    "alpha": alpha,
                },
            }
            # Only embed if alpha > 0 (vector component is needed)
            if alpha > 0:
                run_params["embedder"] = {"text": query}
            result = self.retrieval_pipeline.run(run_params)
            return result.get("retriever", {}).get("documents", [])

        def _rebuild():
            self.retrieval_pipeline = create_retrieval_pipeline()

        return run_with_weaviate_retry(_do_retrieve, _rebuild, context="search")

    def _smart_answer(self, query: str, documents: List[Document]) -> str:
        """Feed retrieved documents + query to the LLM and return the answer."""
        MAX_CHUNK_CHARS = 500
        truncated = [
            Document(content=d.content[:MAX_CHUNK_CHARS], meta=d.meta, score=d.score)
            for d in documents
        ]
        result = self.smart_pipeline.run({
            "prompt_builder": {
                "query": query,
                "documents": truncated,
            },
        })
        replies = result.get("llm", {}).get("replies", [])
        if replies:
            return replies[0]
        raise RuntimeError("LLM returned no reply")

    def run_api(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: Optional[Union[int, str]] = None,
        filename_filter: Optional[str] = None,
        response_mode: Optional[str] = None,
        score_threshold: float = 0.3,
    ) -> str:
        """Search previously uploaded documents for relevant content.

        Use this tool to find information from documents (PDFs, Word files, etc.)
        that have been shared in Matrix rooms.

        Parameters
        ----------
        query : str
            The search query describing what information you need. Use "__list_documents__"
            as the query to see a list of all currently indexed documents.
        mode : str, optional
            DEPRECATED: Use response_mode instead.
        top_k : int or str, optional
            Maximum number of chunks to retrieve (default: 5).
        filename_filter : str, optional
            Filter results to a specific filename (exact match).
        response_mode : str, optional
            "synthesis" (default) — LLM-synthesized answer with source citations.
            "raw" — return raw document chunks with full metadata.
            "both" — return both answer and raw evidence chunks.
        score_threshold : float, optional
            Min confidence score (0.0 to 1.0) to include a chunk (default: 0.3).

        Returns
        -------
        str
            JSON with search results. In smart mode: answer + sources.
            In raw mode: array of chunks with content and metadata.
        """
        # Early-exit for empty queries (before any normalization or retrieval)
        if not query or not query.strip():
            return json.dumps({"status": "error", "detail": "Empty query provided"}, ensure_ascii=False)

        # Handle list_documents mode — direct store query, no embedding needed
        if query == "__list_documents__":
            try:
                store = get_document_store()
                all_docs = store.filter_documents()
                inventory = {}
                for d in all_docs:
                    fn = d.meta.get("source_filename", "unknown")
                    if fn not in inventory:
                        inventory[fn] = {
                            "filename": fn,
                            "chunk_count": d.meta.get("total_chunks"),
                            "ingested_at": d.meta.get("ingested_at", "unknown"),
                        }
                return json.dumps({
                    "status": "ok",
                    "mode": "list_documents",
                    "documents": sorted(list(inventory.values()), key=lambda x: x["filename"]),
                    "total_documents": len(inventory),
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"status": "error", "detail": f"Failed to list documents: {e}"}, ensure_ascii=False)

        normalized_response_mode, response_mode_error = _normalize_response_mode(response_mode, mode)
        if response_mode_error:
            return json.dumps({"status": "error", "detail": response_mode_error}, ensure_ascii=False)

        normalized_top_k, top_k_error = _normalize_top_k(top_k)
        if top_k_error:
            log.warning(
                "search_documents top_k validation failed: "
                f"value={top_k!r} type={type(top_k).__name__} error='{top_k_error}'"
            )
            return json.dumps({"status": "error", "detail": top_k_error}, ensure_ascii=False)

        top_k = normalized_top_k
        response_mode = normalized_response_mode

        log.info(f"Searching: query='{query[:100]}...', top_k={top_k}, threshold={score_threshold}")

        try:
            documents = self._retrieve(query, top_k, filename_filter)

            # Apply score threshold filtering
            original_count = len(documents)
            documents = [d for d in documents if d.score is not None and d.score >= score_threshold]
            if len(documents) < original_count:
                log.info(f"Filtered {original_count - len(documents)} chunks below threshold {score_threshold}")

            if not documents:
                return json.dumps({
                    "status": "ok",
                    "mode": "smart" if response_mode == "synthesis" else response_mode,
                    "response_mode": response_mode,
                    "query": query,
                    "answer": "No documents found matching your query." if response_mode in {"synthesis", "both"} else None,
                    "results": [],
                    "evidence": [],
                    "total_results": 0,
                }, ensure_ascii=False)

            if response_mode == "raw":
                results = [_format_raw_result(doc) for doc in documents]
                log.info(f"Raw search returned {len(results)} results for: {query[:100]}...")
                return json.dumps({
                    "status": "ok",
                    "mode": "raw",
                    "response_mode": "raw",
                    "query": query,
                    "results": results,
                    "evidence": results,
                    "total_results": len(results),
                }, ensure_ascii=False)

            try:
                answer = self._smart_answer(query, documents)
            except Exception as llm_err:
                log.warning(f"Smart search LLM failed, falling back to raw: {llm_err}")
                results = [_format_raw_result(doc) for doc in documents]
                return json.dumps({
                    "status": "ok",
                    "mode": "raw",
                    "response_mode": "raw",
                    "query": query,
                    "detail": f"LLM unavailable, returning raw chunks: {llm_err}",
                    "results": results,
                    "evidence": results,
                    "total_results": len(results),
                }, ensure_ascii=False)

            sources = [_format_source(doc) for doc in documents]
            log.info(f"Smart search answered query with {len(sources)} sources: {query[:100]}...")

            payload = {
                "status": "ok",
                "mode": "smart",
                "response_mode": "synthesis",
                "query": query,
                "answer": answer,
                "sources": sources,
                "total_chunks_searched": len(documents),
            }
            # Only build evidence in "both" mode (Fix #4)
            if response_mode == "both":
                payload["response_mode"] = "both"
                payload["evidence"] = [_format_raw_result(doc) for doc in documents]
                payload["total_results"] = len(documents)

            return json.dumps(payload, ensure_ascii=False)

        except Exception as e:
            log.exception(f"Error searching documents: {e}")
            return json.dumps({
                "status": "error",
                "detail": f"Search failed: {str(e)}",
            }, ensure_ascii=False)
