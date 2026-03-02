"""
Ingest Document Pipeline

Receives extracted document text, chunks it, embeds it, and stores it
in the shared Weaviate document store. Any Letta agent can then search
this store via the search_documents pipeline/tool.

Flow:
  text + metadata → DocumentSplitter → AddChunkMetadata → OpenAIDocumentEmbedder → WeaviateDocumentWriter
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from haystack import Document, Pipeline, component
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils import Secret
from resources.docstore import get_document_store
from haystack.components.writers import DocumentWriter
from loguru import logger as log


@component
class AddChunkMetadata:
    """Add stable chunk position metadata to split documents."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        total_chunks = len(documents)
        for idx, doc in enumerate(documents):
            doc.meta["chunk_index"] = idx
            doc.meta["total_chunks"] = total_chunks
        return {"documents": documents}




def create_pipeline() -> Pipeline:
    """Build a fresh ingest pipeline with a new Weaviate connection."""
    embedding_model = os.getenv("HAYHOOKS_EMBEDDING_MODEL", "gemini/text-embedding-004")
    api_base = os.getenv("HAYHOOKS_EMBEDDING_API_BASE", "http://100.81.139.20:11450/v1")

    splitter = DocumentSplitter(
        split_by="word",
        split_length=512,
        split_overlap=64,
    )
    chunk_metadata = AddChunkMetadata()

    embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=api_base,
        model=embedding_model,
        batch_size=128,  # vLLM handles large batches; reduce round-trips
    )

    writer = DocumentWriter(document_store=get_document_store())

    pipe = Pipeline()
    pipe.add_component("splitter", splitter)
    pipe.add_component("chunk_metadata", chunk_metadata)
    pipe.add_component("embedder", embedder)
    pipe.add_component("writer", writer)

    pipe.connect("splitter.documents", "chunk_metadata.documents")
    pipe.connect("chunk_metadata.documents", "embedder.documents")
    pipe.connect("embedder.documents", "writer.documents")

    return pipe


class PipelineWrapper(BasePipelineWrapper):
    """Ingest extracted document text into the shared Weaviate document store.

    Chunks text, generates embeddings via LiteLLM (OpenAI-compatible),
    and writes to Weaviate for cross-agent retrieval.
    """

    def setup(self) -> None:
        # Create an initial pipeline; run_api will rebuild if connection is stale
        self.pipeline = create_pipeline()

    def run_api(
        self,
        text: str,
        filename: str,
        room_id: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> str:
        """Ingest a document into the shared document store.

        Call this after extracting text from a document (PDF, DOCX, etc.)
        to chunk, embed, and store it for cross-agent search.

        Parameters
        ----------
        text : str
            The full extracted text content of the document.
        filename : str
            Original filename of the document.
        room_id : str, optional
            Matrix room ID where the document was uploaded.
        sender : str, optional
            Matrix user ID of the person who uploaded the document.

        Returns
        -------
        str
            JSON with ingestion results (document_id, chunk_count, status).
        """
        log.info(f"Ingesting document: {filename} ({len(text)} chars)")

        if not text or not text.strip():
            return json.dumps({
                "status": "error",
                "detail": "Empty document text provided",
            })

        now = datetime.now(timezone.utc).isoformat()

        # Create a single Document for the splitter
        doc = Document(
            content=text,
            meta={
                "source_filename": filename,
                "source_room_id": room_id or "",
                "source_sender": sender or "",
                "ingested_at": now,
            },
        )

        # Try with existing pipeline first; rebuild on connection errors
        for attempt in range(2):
            try:
                result = self.pipeline.run({
                    "splitter": {"documents": [doc]},
                })

                written = result.get("writer", {}).get("documents_written", 0)

                log.info(f"Document '{filename}' ingested: {written} chunks stored")

                return json.dumps({
                    "status": "ok",
                    "filename": filename,
                    "chunks_stored": written,
                    "ingested_at": now,
                })

            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and ("closed" in err_str or "connect" in err_str or "schema" in err_str or "graphql" in err_str):
                    log.warning(f"Weaviate connection stale, rebuilding pipeline: {e}")
                    self.pipeline = create_pipeline()
                    continue
                log.exception(f"Error ingesting document '{filename}': {e}")
                return json.dumps({
                    "status": "error",
                    "detail": f"Ingestion failed: {str(e)}",
                })

        # Should not reach here, but just in case
        return json.dumps({
            "status": "error",
            "detail": "Ingestion failed after retries",
        })
