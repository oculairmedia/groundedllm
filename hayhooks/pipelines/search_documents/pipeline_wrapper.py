"""
Search Documents Pipeline

Queries the shared Weaviate document store for relevant chunks.
This pipeline is exposed as an MCP tool so any Letta agent can search
documents that were ingested via the ingest_document pipeline.

Flow:
  query → OpenAITextEmbedder → WeaviateEmbeddingRetriever → formatted results
"""

import json
import os
from typing import Optional

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from loguru import logger as log


def get_document_store() -> WeaviateDocumentStore:
    """Create a WeaviateDocumentStore connected to the shared docstore."""
    weaviate_url = os.getenv("WEAVIATE_URL", "http://docstore-weaviate:8080")
    collection = os.getenv("WEAVIATE_COLLECTION", "Documents")
    return WeaviateDocumentStore(
        url=weaviate_url,
        collection_settings={
            "class": collection,
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "source_filename", "dataType": ["text"]},
                {"name": "source_room_id", "dataType": ["text"]},
                {"name": "source_sender", "dataType": ["text"]},
                {"name": "chunk_index", "dataType": ["int"]},
                {"name": "total_chunks", "dataType": ["int"]},
                {"name": "ingested_at", "dataType": ["text"]},
            ],
        },
    )


def create_pipeline() -> Pipeline:
    """Build a fresh search pipeline with a new Weaviate connection."""
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


class PipelineWrapper(BasePipelineWrapper):
    """Search the shared document store for relevant content.

    Embeds the query, retrieves the most relevant document chunks
    from Weaviate, and returns them with metadata.
    """

    def setup(self) -> None:
        # Create an initial pipeline; run_api will rebuild if connection is stale
        self.pipeline = create_pipeline()

    def run_api(
        self,
        query: str,
        top_k: Optional[int] = None,
        filename_filter: Optional[str] = None,
    ) -> str:
        """Search previously uploaded documents for relevant content.

        Use this tool to find information from documents (PDFs, Word files, etc.)
        that have been shared in Matrix rooms. Returns the most relevant text
        chunks with source information.

        Parameters
        ----------
        query : str
            The search query describing what information you need.
        top_k : int, optional
            Maximum number of results to return (default: 5).
        filename_filter : str, optional
            Filter results to a specific filename (exact match).

        Returns
        -------
        str
            JSON array of matching document chunks with content and metadata.
        """
        if top_k is None:
            top_k = 5

        log.info(f"Searching documents: query='{query[:80]}...', top_k={top_k}")

        if not query or not query.strip():
            return json.dumps({
                "status": "error",
                "detail": "Empty query provided",
            })

        # Try with existing pipeline first; rebuild on connection errors
        for attempt in range(2):
            try:
                result = self.pipeline.run({
                    "embedder": {"text": query},
                    "retriever": {"top_k": top_k},
                })

                documents = result.get("retriever", {}).get("documents", [])

                # Apply filename filter if provided
                if filename_filter:
                    documents = [
                        d for d in documents
                        if d.meta.get("source_filename", "") == filename_filter
                    ]

                # Format results
                results = []
                for doc in documents:
                    results.append({
                        "content": doc.content,
                        "score": doc.score,
                        "filename": doc.meta.get("source_filename", "unknown"),
                        "room_id": doc.meta.get("source_room_id", ""),
                        "sender": doc.meta.get("source_sender", ""),
                        "ingested_at": doc.meta.get("ingested_at", ""),
                    })

                log.info(f"Search returned {len(results)} results for query: {query[:50]}...")

                return json.dumps({
                    "status": "ok",
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                })

            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and ("closed" in err_str or "connect" in err_str):
                    log.warning(f"Weaviate connection stale, rebuilding pipeline: {e}")
                    self.pipeline = create_pipeline()
                    continue
                log.exception(f"Error searching documents: {e}")
                return json.dumps({
                    "status": "error",
                    "detail": f"Search failed: {str(e)}",
                })

        # Should not reach here
        return json.dumps({
            "status": "error",
            "detail": "Search failed after retries",
        })
