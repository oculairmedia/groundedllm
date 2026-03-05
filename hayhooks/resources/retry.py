"""Shared retry helper for Weaviate pipelines.

Extracts the duplicated stale-connection retry logic used by both
search_documents and ingest_document pipeline wrappers.
"""

from typing import Callable, TypeVar

from loguru import logger as log

from resources.docstore import reset_document_store

T = TypeVar("T")

# Error substrings that indicate a stale or broken Weaviate connection
_STALE_CONNECTION_MARKERS = ("closed", "connect", "schema", "graphql")


def run_with_weaviate_retry(
    operation: Callable[[], T],
    rebuild_pipeline: Callable[[], None],
    context: str = "pipeline",
) -> T:
    """Execute an operation with one automatic retry on stale Weaviate connections.

    Parameters
    ----------
    operation : callable
        The function to execute (e.g. pipeline.run(...)).
    rebuild_pipeline : callable
        Function that rebuilds the pipeline with a fresh Weaviate connection.
        Called on the first failure before retrying.
    context : str
        Label for log messages (e.g. "search" or "ingest").

    Returns
    -------
    T
        The result of the operation.

    Raises
    ------
    Exception
        Re-raises the original exception if retry also fails or if the error
        is not a stale connection error.
    """
    for attempt in range(2):
        try:
            return operation()
        except Exception as e:
            err_str = str(e).lower()
            is_stale = any(marker in err_str for marker in _STALE_CONNECTION_MARKERS)
            if attempt == 0 and is_stale:
                log.warning(f"Weaviate connection stale ({context}), rebuilding: {e}")
                reset_document_store()
                rebuild_pipeline()
                continue
            raise

    # Unreachable but satisfies type checker
    raise RuntimeError(f"Retry exhausted for {context}")
