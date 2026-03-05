"""Shared Weaviate document store factory with singleton caching."""

import json
import logging
import os
import threading
import urllib.request
import urllib.error

from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore

logger = logging.getLogger(__name__)

# Schema definition — single source of truth for both ingest and search pipelines.
COLLECTION_PROPERTIES = [
    {"name": "content", "dataType": ["text"]},
    {"name": "source_filename", "dataType": ["text"]},
    {"name": "source_room_id", "dataType": ["text"]},
    {"name": "source_sender", "dataType": ["text"]},
    {"name": "chunk_index", "dataType": ["int"]},
    {"name": "total_chunks", "dataType": ["int"]},
    {"name": "ingested_at", "dataType": ["text"]},
    {"name": "page_number", "dataType": ["int"]},
    {"name": "section_title", "dataType": ["text"]},
]

# HNSW index tuning — optimized for speed at typical document volumes (<100k chunks)
VECTOR_INDEX_CONFIG = {
    "efConstruction": 64,   # default 128 — halves index build time
    "ef": 64,               # default -1 (auto) — cap search effort
    "maxConnections": 16,   # default 32 — less RAM per vector
}

# Stopword removal — smaller BM25 inverted index, faster keyword matching
INVERTED_INDEX_CONFIG = {
    "stopwords": {"preset": "en"},
}


def _ensure_collection_exists(weaviate_url: str, collection_name: str) -> None:
    """Create the Weaviate collection if it doesn't exist yet."""
    try:
        req = urllib.request.Request(f"{weaviate_url}/v1/schema/{collection_name}", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return  # already exists
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        pass  # can't reach or 404 — try to create

    logger.info(f"Collection '{collection_name}' not found, creating it...")
    try:
        payload = json.dumps({
            "class": collection_name,
            "properties": COLLECTION_PROPERTIES,
            "vectorizer": "none",
            "vectorIndexConfig": VECTOR_INDEX_CONFIG,
            "invertedIndexConfig": INVERTED_INDEX_CONFIG,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{weaviate_url}/v1/schema",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status < 300:
                logger.info(f"Collection '{collection_name}' created successfully")
            else:
                logger.error(f"Failed to create collection: HTTP {resp.status}")
                raise RuntimeError(f"Weaviate schema creation returned {resp.status}")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        logger.error(f"Failed to auto-create collection '{collection_name}': {e}")
        raise


# ---------------------------------------------------------------------------
# Singleton document store — avoids redundant schema checks per call
# ---------------------------------------------------------------------------

_store_lock = threading.Lock()
_store_instance: WeaviateDocumentStore | None = None


def get_document_store() -> WeaviateDocumentStore:
    """Get a WeaviateDocumentStore singleton, auto-creating the collection on first call."""
    global _store_instance

    if _store_instance is not None:
        return _store_instance

    with _store_lock:
        # Double-check after acquiring lock
        if _store_instance is not None:
            return _store_instance

        weaviate_url = os.getenv("WEAVIATE_URL", "http://docstore-weaviate:8080")
        grpc_url = os.getenv("WEAVIATE_GRPC_URL")  # e.g. "docstore-weaviate:50051"
        collection = os.getenv("WEAVIATE_COLLECTION", "Documents")

        _ensure_collection_exists(weaviate_url, collection)

        # Build kwargs — use native grpc_port parameter if gRPC is configured
        store_kwargs = {
            "url": weaviate_url,
            "collection_settings": {
                "class": collection,
                "properties": COLLECTION_PROPERTIES,
            },
        }
        if grpc_url:
            # WeaviateDocumentStore has built-in grpc_port/grpc_secure params
            try:
                grpc_port = int(grpc_url.split(":")[1])
                store_kwargs["grpc_port"] = grpc_port
                logger.info(f"gRPC enabled on port {grpc_port}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse gRPC URL ({grpc_url}), falling back to REST: {e}")
        _store_instance = WeaviateDocumentStore(**store_kwargs)
        logger.info("WeaviateDocumentStore singleton created")
        return _store_instance


def reset_document_store() -> None:
    """Reset the cached document store (useful after connection errors)."""
    global _store_instance
    with _store_lock:
        _store_instance = None
        logger.info("WeaviateDocumentStore singleton reset")
