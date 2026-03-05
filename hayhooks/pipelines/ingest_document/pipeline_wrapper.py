"""
Ingest Document Pipeline

Receives extracted document text, chunks it, embeds it, and stores it
in the shared Weaviate document store. Any Letta agent can then search
this store via the search_documents pipeline/tool.

Flow:
  text + metadata → DocumentSplitter → AddChunkMetadata → ParallelDocumentEmbedder → WeaviateDocumentWriter
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from haystack import Document, Pipeline, component
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from resources.docstore import get_document_store
from resources.retry import run_with_weaviate_retry
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


@component
class ParallelDocumentEmbedder:
    """Parallelize document embedding using ThreadPoolExecutor.
    
    Wraps OpenAIDocumentEmbedder to process multiple batches concurrently,
    dramatically reducing total HTTP roundtrip wait time for large document sets.
    """
    
    def __init__(self, embedder: OpenAIDocumentEmbedder, max_workers: int = 8, batch_size: int = 128):
        self.embedder = embedder
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        if not documents:
            return {"documents": []}
        
        # Split documents into batches matching embedder batch_size
        batches = [
            documents[i:i + self.batch_size] 
            for i in range(0, len(documents), self.batch_size)
        ]
        n_batches = len(batches)
        effective_workers = min(self.max_workers, n_batches)
        
        log.info(
            f"ParallelDocumentEmbedder: Processing {len(documents)} docs in {n_batches} "
            f"batches with {effective_workers} workers"
        )

        if n_batches == 1:
            return self.embedder.run(documents=batches[0])
        
        # Process batches in parallel using ThreadPoolExecutor
        results = [None] * len(batches)  # Pre-allocate to preserve order
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all batch embedding tasks
            future_to_idx = {
                executor.submit(self.embedder.run, documents=batch): idx
                for idx, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result.get("documents", [])
                except Exception as e:
                    log.error(f"Error embedding batch {idx}: {e}")
                    raise
        
        # Flatten results preserving batch order
        all_documents = []
        for batch_docs in results:
            if batch_docs:
                all_documents.extend(batch_docs)
        
        log.info(f"ParallelDocumentEmbedder: Completed embedding {len(all_documents)} documents")
        return {"documents": all_documents}



def create_pipeline() -> Pipeline:
    """Build a fresh ingest pipeline with a new Weaviate connection."""
    embedding_model = os.getenv("HAYHOOKS_EMBEDDING_MODEL", "gemini/text-embedding-004")
    api_base = os.getenv("HAYHOOKS_EMBEDDING_API_BASE", "http://100.81.139.20:11450/v1")

    splitter = DocumentSplitter(
        split_by="word",
        split_length=400,  # ~20 sentences, faster than sentence splitting
        split_overlap=40,
    )
    chunk_metadata = AddChunkMetadata()

    embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=api_base,
        model=embedding_model,
        batch_size=128,  # vLLM handles large batches; reduce round-trips
        progress_bar=False,  # Disable I/O overhead in headless API environment
    )
    
    # Wrap embedder with parallel processing
    parallel_embedder = ParallelDocumentEmbedder(embedder, max_workers=8, batch_size=128)

    # DuplicatePolicy.OVERWRITE enables Weaviate's batch API for faster writes
    writer = DocumentWriter(
        document_store=get_document_store(),
        policy=DuplicatePolicy.OVERWRITE,
    )

    pipe = Pipeline()
    pipe.add_component("splitter", splitter)
    pipe.add_component("chunk_metadata", chunk_metadata)
    pipe.add_component("parallel_embedder", parallel_embedder)
    pipe.add_component("writer", writer)

    pipe.connect("splitter.documents", "chunk_metadata.documents")
    pipe.connect("chunk_metadata.documents", "parallel_embedder.documents")
    pipe.connect("parallel_embedder.documents", "writer.documents")

    return pipe


class PipelineWrapper(BasePipelineWrapper):
    """Ingest extracted document text into the shared Weaviate document store.

    Chunks text, generates embeddings via LiteLLM (OpenAI-compatible),
    and writes to Weaviate for cross-agent retrieval.
    """

    def setup(self) -> None:
        # Create an initial pipeline; run_api will rebuild if connection is stale
        self.pipeline = create_pipeline()
        try:
            warmup_doc = Document(content="warmup", meta={})
            parallel_embedder = self.pipeline.get_component("parallel_embedder")
            embedder = getattr(parallel_embedder, "embedder", None)
            run_fn = getattr(embedder, "run", None)
            if callable(run_fn):
                run_fn(documents=[warmup_doc])
                log.info("Ingest embedder pre-warmed")
            else:
                log.warning("Ingest embedder warm-up skipped: component missing run()")
        except Exception:
            log.warning("Ingest embedder warm-up failed")

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
            }, ensure_ascii=False)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        now = datetime.now(timezone.utc).isoformat()

        base_meta = {
            "source_filename": filename,
            "source_room_id": room_id or "",
            "source_sender": sender or "",
            "ingested_at": now,
        }

        pre_split_threshold_chars = 500_000
        max_doc_chars = 100_000
        if len(text) > pre_split_threshold_chars:
            sections = [text[i:i + max_doc_chars] for i in range(0, len(text), max_doc_chars)]
            docs = [
                Document(
                    content=section,
                    meta={
                        **base_meta,
                        "source_section_index": idx,
                        "source_section_total": len(sections),
                    },
                )
                for idx, section in enumerate(sections)
            ]
            log.info(
                f"Pre-split large document '{filename}' into {len(docs)} sections "
                f"(len={len(text)} chars)"
            )
        else:
            docs = [Document(content=text, meta=base_meta)]

        # Try with existing pipeline first; rebuild on connection errors
        def _do_ingest():
            store = get_document_store()
            stale_docs = store.filter_documents(filters={
                "field": "source_filename",
                "operator": "==",
                "value": filename,
            })
            stale_ids = [d.id for d in stale_docs if d.id]
            if stale_ids:
                store.delete_documents(document_ids=stale_ids)
                log.info(f"Deleted {len(stale_ids)} stale chunks for '{filename}' before re-ingest")

            result = self.pipeline.run({
                "splitter": {"documents": docs},
            })
            written = result.get("writer", {}).get("documents_written", 0)
            log.info(f"Document '{filename}' ingested: {written} chunks stored")
            return json.dumps({
                "status": "ok",
                "filename": filename,
                "chunks_stored": written,
                "ingested_at": now,
            }, ensure_ascii=False)

        def _rebuild():
            self.pipeline = create_pipeline()

        try:
            return run_with_weaviate_retry(_do_ingest, _rebuild, context="ingest")
        except Exception as e:
            log.exception(f"Error ingesting document '{filename}': {e}")
            return json.dumps({
                "status": "error",
                "detail": f"Ingestion failed: {str(e)}",
            }, ensure_ascii=False)
