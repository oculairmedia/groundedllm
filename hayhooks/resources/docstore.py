"""Shared Weaviate document store factory with auto-create."""

import json
import logging
import os
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
]


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


def get_document_store() -> WeaviateDocumentStore:
    """Get a WeaviateDocumentStore, auto-creating the collection if needed."""
    weaviate_url = os.getenv("WEAVIATE_URL", "http://docstore-weaviate:8080")
    collection = os.getenv("WEAVIATE_COLLECTION", "Documents")

    _ensure_collection_exists(weaviate_url, collection)

    return WeaviateDocumentStore(
        url=weaviate_url,
        collection_settings={
            "class": collection,
            "properties": COLLECTION_PROPERTIES,
        },
    )
