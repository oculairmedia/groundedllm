"""
Search Documents Pipeline

Queries the shared Weaviate document store for relevant chunks.
This pipeline is exposed as an MCP tool so any Letta agent can search
documents that were ingested via the ingest_document pipeline.

Modes:
  smart (default) — retrieves chunks, feeds them to an LLM with a citation
                     prompt, and returns a synthesized answer with source pointers.
  raw             — returns the raw chunks as-is (original behaviour).

Flow (smart):
  query → OpenAITextEmbedder → WeaviateEmbeddingRetriever → PromptBuilder → OpenAIGenerator → answer + sources
Flow (raw):
  query → OpenAITextEmbedder → WeaviateEmbeddingRetriever → formatted JSON
"""

import json
import os
from typing import List, Optional

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from loguru import logger as log

from resources.utils import read_resource_file
from resources.docstore import get_document_store




# ---------------------------------------------------------------------------
# Retrieval pipeline (shared by both modes)
# ---------------------------------------------------------------------------

def create_retrieval_pipeline() -> Pipeline:
    """Build a pipeline that embeds a query and retrieves matching chunks."""
    embedding_model = os.getenv("HAYHOOKS_EMBEDDING_MODEL", "gemini/text-embedding-004")
    api_base = os.getenv("HAYHOOKS_EMBEDDING_API_BASE", "http://100.81.139.20:11450/v1")

    embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=api_base,
        model=embedding_model,
    )

    retriever = WeaviateEmbeddingRetriever(
        document_store=get_document_store(),
        top_k=10,
    )

    pipe = Pipeline()
    pipe.add_component("embedder", embedder)
    pipe.add_component("retriever", retriever)
    pipe.connect("embedder.embedding", "retriever.query_embedding")

    return pipe


# ---------------------------------------------------------------------------
# Smart-mode LLM pipeline (prompt + generator)
# ---------------------------------------------------------------------------

def create_smart_pipeline() -> Pipeline:
    """Build a pipeline that takes retrieved docs + query and produces an LLM answer."""
    template = read_resource_file("document_search_prompt.md")
    prompt_builder = PromptBuilder(
        template=template,
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
    return {
        "filename": doc.meta.get("source_filename", "unknown"),
        "chunk": doc.meta.get("chunk_index"),
        "of": doc.meta.get("total_chunks"),
        "score": round(doc.score, 3) if doc.score is not None else None,
    }


def _format_raw_result(doc: Document) -> dict:
    """Build a full result dict for raw mode."""
    return {
        "content": doc.content,
        "score": doc.score,
        "filename": doc.meta.get("source_filename", "unknown"),
        "chunk_index": doc.meta.get("chunk_index"),
        "total_chunks": doc.meta.get("total_chunks"),
        "room_id": doc.meta.get("source_room_id", ""),
        "sender": doc.meta.get("source_sender", ""),
        "ingested_at": doc.meta.get("ingested_at", ""),
    }


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

    def _retrieve(
        self,
        query: str,
        top_k: int,
        filename_filter: Optional[str] = None,
    ) -> List[Document]:
        """Run the retrieval pipeline, with stale-connection retry."""
        for attempt in range(2):
            try:
                result = self.retrieval_pipeline.run({
                    "embedder": {"text": query},
                    "retriever": {"top_k": top_k},
                })
                documents = result.get("retriever", {}).get("documents", [])

                if filename_filter:
                    documents = [
                        d for d in documents
                        if d.meta.get("source_filename", "") == filename_filter
                    ]
                return documents

            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and ("closed" in err_str or "connect" in err_str or "schema" in err_str or "graphql" in err_str):
                    log.warning(f"Weaviate connection stale, rebuilding retrieval pipeline: {e}")
                    self.retrieval_pipeline = create_retrieval_pipeline()
                    continue
                raise

        return []  # unreachable but satisfies return type

    def _smart_answer(self, query: str, documents: List[Document]) -> str:
        """Feed retrieved documents + query to the LLM and return the answer."""
        result = self.smart_pipeline.run({
            "prompt_builder": {
                "query": query,
                "documents": documents,
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
        top_k: Optional[int] = None,
        filename_filter: Optional[str] = None,
    ) -> str:
        """Search previously uploaded documents for relevant content.

        Use this tool to find information from documents (PDFs, Word files, etc.)
        that have been shared in Matrix rooms.

        Parameters
        ----------
        query : str
            The search query describing what information you need.
        mode : str, optional
            "smart" (default) — LLM-synthesized answer with source citations.
            "raw" — return raw document chunks with full metadata.
        top_k : int, optional
            Maximum number of chunks to retrieve (default: 5).
        filename_filter : str, optional
            Filter results to a specific filename (exact match).

        Returns
        -------
        str
            JSON with search results. In smart mode: answer + sources.
            In raw mode: array of chunks with content and metadata.
        """
        if mode is None:
            mode = "smart"
        if top_k is None:
            top_k = 5

        log.info(f"Searching documents: query='{query[:80]}...', mode={mode}, top_k={top_k}")

        if not query or not query.strip():
            return json.dumps({"status": "error", "detail": "Empty query provided"})

        try:
            documents = self._retrieve(query, top_k, filename_filter)

            if not documents:
                return json.dumps({
                    "status": "ok",
                    "mode": mode,
                    "query": query,
                    "answer": "No documents found matching your query." if mode == "smart" else None,
                    "results": [],
                    "total_results": 0,
                })

            if mode == "raw":
                results = [_format_raw_result(doc) for doc in documents]
                log.info(f"Raw search returned {len(results)} results for: {query[:50]}...")
                return json.dumps({
                    "status": "ok",
                    "mode": "raw",
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                })

            # --- Smart mode: LLM synthesis ---
            try:
                answer = self._smart_answer(query, documents)
            except Exception as llm_err:
                log.warning(f"Smart search LLM failed, falling back to raw: {llm_err}")
                results = [_format_raw_result(doc) for doc in documents]
                return json.dumps({
                    "status": "ok",
                    "mode": "raw",
                    "query": query,
                    "detail": f"LLM unavailable, returning raw chunks: {llm_err}",
                    "results": results,
                    "total_results": len(results),
                })

            sources = [_format_source(doc) for doc in documents]
            log.info(f"Smart search answered query with {len(sources)} sources: {query[:50]}...")

            return json.dumps({
                "status": "ok",
                "mode": "smart",
                "query": query,
                "answer": answer,
                "sources": sources,
                "total_chunks_searched": len(documents),
            })

        except Exception as e:
            log.exception(f"Error searching documents: {e}")
            return json.dumps({
                "status": "error",
                "detail": f"Search failed: {str(e)}",
            })
