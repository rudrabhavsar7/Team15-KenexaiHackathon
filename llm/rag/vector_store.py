"""
llm/rag/vector_store.py

RAG Pipeline — Stage 3: Vector Store (Qdrant)
----------------------------------------------
Stores document embeddings in a Qdrant collection and exposes similarity
search so the retriever can find the most relevant incident documents for
any given query embedding.

Pipeline position
-----------------
embedding_model  ──►  vector_store  ──►  retriever

Qdrant modes supported
-----------------------
• In-memory  (default, zero config — great for hackathon demos)
• On-disk     (pass ``storage_path`` to :func:`init_qdrant_client`)
• Remote      (pass ``host`` + ``port`` to :func:`init_qdrant_client`)

Usage
-----
    from llm.rag.vector_store import (
        init_qdrant_client,
        create_collection,
        upsert_documents,
        search_similar_documents,
    )

    client     = init_qdrant_client()
    create_collection(client, "incidents", vector_size=384)
    upsert_documents(client, "incidents", embeddings, documents)
    results    = search_similar_documents(client, "incidents", query_vec)

Author: Team 15 — KenexAI Hackathon
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION_NAME: str = "incidents"
DEFAULT_VECTOR_SIZE: int = 384          # BAAI/bge-small-en-v1.5 output dim
DEFAULT_DISTANCE: Distance = Distance.COSINE
DEFAULT_TOP_K: int = 5
DEFAULT_BATCH_SIZE: int = 256           # points per upsert batch

# Payload key under which the raw document text is stored.
DOCUMENT_PAYLOAD_KEY: str = "document"


# ---------------------------------------------------------------------------
# 1. init_qdrant_client
# ---------------------------------------------------------------------------

def init_qdrant_client(
    storage_path: Optional[str] = None,
    host: Optional[str] = None,
    port: int = 6333,
    *,
    in_memory: bool = True,
) -> QdrantClient:
    """Initialise and return a Qdrant client.

    The client mode is resolved in the following priority order:

    1. **Remote** — if *host* is provided, connects to a running Qdrant server.
    2. **On-disk** — if *storage_path* is provided, persists data to that path.
    3. **In-memory** — default; data is lost when the process exits.

    Parameters
    ----------
    storage_path : str, optional
        Directory path for on-disk persistence (e.g. ``"./qdrant_storage"``).
    host : str, optional
        Hostname of a remote Qdrant instance (e.g. ``"localhost"``).
    port : int, optional
        TCP port of the remote Qdrant instance.  Defaults to ``6333``.
    in_memory : bool, keyword-only
        When *storage_path* and *host* are both ``None``, ``True`` selects
        ``:memory:`` mode (default).  Setting ``False`` with no other args
        raises ``ValueError``.

    Returns
    -------
    QdrantClient
        Ready-to-use client instance.

    Raises
    ------
    ValueError
        If no valid backend can be inferred from the provided arguments.
    RuntimeError
        If the Qdrant client cannot connect / initialise.

    Examples
    --------
    >>> client = init_qdrant_client()                          # in-memory
    >>> client = init_qdrant_client(storage_path="./data/qdrant")  # on-disk
    >>> client = init_qdrant_client(host="localhost", port=6333)   # remote
    """
    try:
        if host:
            logger.info("Connecting to remote Qdrant at %s:%d …", host, port)
            client = QdrantClient(host=host, port=port)
            mode = f"remote ({host}:{port})"

        elif storage_path:
            logger.info("Initialising on-disk Qdrant at '%s' …", storage_path)
            client = QdrantClient(path=storage_path)
            mode = f"on-disk ('{storage_path}')"

        elif in_memory:
            logger.info("Initialising in-memory Qdrant client …")
            client = QdrantClient(":memory:")
            mode = "in-memory"

        else:
            raise ValueError(
                "Cannot initialise Qdrant: provide 'host', 'storage_path', "
                "or set in_memory=True."
            )

        logger.info("Qdrant client ready — mode: %s.", mode)
        return client

    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Qdrant client: {exc}") from exc


# ---------------------------------------------------------------------------
# 2. create_collection
# ---------------------------------------------------------------------------

def create_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    distance: Distance = DEFAULT_DISTANCE,
    *,
    recreate_if_exists: bool = False,
) -> None:
    """Create a Qdrant vector collection for incident embeddings.

    Parameters
    ----------
    client : QdrantClient
        An initialised Qdrant client (from :func:`init_qdrant_client`).
    collection_name : str, optional
        Name of the Qdrant collection.  Defaults to ``"incidents"``.
    vector_size : int, optional
        Dimensionality of the vectors to store.  Must match the embedding
        model output (384 for ``BAAI/bge-small-en-v1.5``).
    distance : Distance, optional
        Similarity metric.  Defaults to ``Distance.COSINE``.
    recreate_if_exists : bool, keyword-only
        If ``True`` and the collection already exists, it is **deleted and
        re-created** (full reset).  Defaults to ``False`` (skip creation if
        collection already exists).

    Raises
    ------
    RuntimeError
        If the collection cannot be created.
    """
    existing = _get_existing_collection_names(client)

    if collection_name in existing:
        if recreate_if_exists:
            logger.warning(
                "Collection '%s' already exists — recreating (data will be lost).",
                collection_name,
            )
            client.delete_collection(collection_name)
        else:
            logger.info(
                "Collection '%s' already exists — skipping creation.",
                collection_name,
            )
            return

    logger.info(
        "Creating collection '%s' — vector_size=%d, distance=%s.",
        collection_name, vector_size, distance,
    )
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        logger.info("Collection '%s' created successfully.", collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create collection '{collection_name}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 3. upsert_documents
# ---------------------------------------------------------------------------

def upsert_documents(
    client: QdrantClient,
    collection_name: str,
    embeddings: np.ndarray,
    documents: List[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    start_id: int = 0,
) -> int:
    """Insert or update (upsert) document embeddings in a Qdrant collection.

    Each point in Qdrant stores:

    * **id** — a unique integer (auto-assigned, starting from *start_id*).
    * **vector** — a Python list of floats derived from *embeddings*.
    * **payload** — a dict containing the raw document text under the key
      ``"document"``.

    Points are written in batches of *batch_size* to avoid large payloads.

    Parameters
    ----------
    client : QdrantClient
        An initialised Qdrant client.
    collection_name : str
        Target Qdrant collection (must already exist).
    embeddings : np.ndarray
        2-D array of shape ``(N, D)`` produced by ``embedding_model.embed_documents``.
    documents : List[str]
        List of ``N`` source document strings (aligned with *embeddings* by index).
    batch_size : int, keyword-only
        Number of points per upsert request.  Defaults to ``256``.
    start_id : int, keyword-only
        Integer offset for point IDs.  Useful when adding documents to an
        existing collection incrementally.  Defaults to ``0``.

    Returns
    -------
    int
        Total number of points successfully upserted.

    Raises
    ------
    ValueError
        If *embeddings* and *documents* have different lengths, or if either
        is empty.
    RuntimeError
        If any Qdrant upsert call fails.
    """
    _validate_upsert_inputs(embeddings, documents)

    # Ensure embeddings are a plain NumPy array.
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)

    n_points = len(documents)
    logger.info(
        "Upserting %d document(s) into collection '%s' — batch_size=%d.",
        n_points, collection_name, batch_size,
    )

    total_upserted = 0
    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        batch_docs = documents[batch_start:batch_end]
        batch_vecs = embeddings[batch_start:batch_end]

        points = [
            PointStruct(
                id=start_id + batch_start + i,
                vector=batch_vecs[i].tolist(),          # Qdrant requires list[float]
                payload={DOCUMENT_PAYLOAD_KEY: batch_docs[i]},
            )
            for i in range(len(batch_docs))
        ]

        try:
            client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)
            logger.debug(
                "Upserted batch [%d:%d] — %d points.",
                batch_start, batch_end, len(points),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant upsert failed on batch [{batch_start}:{batch_end}]: {exc}"
            ) from exc

    logger.info(
        "Upsert complete — %d point(s) stored in '%s'.",
        total_upserted, collection_name,
    )
    return total_upserted


# ---------------------------------------------------------------------------
# 4. search_similar_documents
# ---------------------------------------------------------------------------

def search_similar_documents(
    client: QdrantClient,
    collection_name: str,
    query_embedding: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Perform semantic similarity search against the Qdrant collection.

    Parameters
    ----------
    client : QdrantClient
        An initialised Qdrant client.
    collection_name : str
        The collection to search (must already contain points).
    query_embedding : np.ndarray
        1-D query vector of shape ``(embedding_dimension,)`` produced by
        ``embedding_model.embed_query``.
    top_k : int, optional
        Maximum number of results to return.  Defaults to ``5``.

    Returns
    -------
    List[Dict[str, Any]]
        Ranked list of matching documents.  Each item contains:

        ``id``
            Qdrant point ID (int).
        ``score``
            Cosine similarity score in ``[-1, 1]`` (higher is more similar).
        ``document``
            Raw document text stored in the payload.

    Raises
    ------
    ValueError
        If *query_embedding* is ``None`` or not a 1-D array.
    RuntimeError
        If the Qdrant search call fails.

    Example
    -------
    >>> results = search_similar_documents(client, "incidents", query_vec, top_k=3)
    >>> for r in results:
    ...     print(r["score"], r["document"][:60])
    """
    if query_embedding is None:
        raise ValueError("'query_embedding' must not be None.")

    query_vec = np.squeeze(np.array(query_embedding, dtype=np.float32))
    if query_vec.ndim != 1:
        raise ValueError(
            f"'query_embedding' must be a 1-D vector, got shape {query_vec.shape}."
        )

    logger.info(
        "Searching '%s' — top_k=%d, query_dim=%d.",
        collection_name, top_k, query_vec.shape[0],
    )

    try:
        # qdrant-client >= 1.10  uses query_points() — .search() was removed.
        response = client.query_points(
            collection_name=collection_name,
            query=query_vec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        hits = response.points          # List[ScoredPoint]
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant search failed on collection '{collection_name}': {exc}"
        ) from exc

    results: List[Dict[str, Any]] = [
        {
            "id": hit.id,
            "score": round(float(hit.score), 6),
            "document": hit.payload.get(DOCUMENT_PAYLOAD_KEY, ""),
        }
        for hit in hits
    ]

    logger.info(
        "Search returned %d result(s) — top score: %s.",
        len(results),
        results[0]["score"] if results else "N/A",
    )
    return results


# ---------------------------------------------------------------------------
# 5. delete_collection
# ---------------------------------------------------------------------------

def delete_collection(
    client: QdrantClient,
    collection_name: str,
) -> bool:
    """Delete a Qdrant collection and all its data.

    This is a destructive, irreversible operation.  Use it to reset the
    vector store during development or testing.

    Parameters
    ----------
    client : QdrantClient
        An initialised Qdrant client.
    collection_name : str
        Name of the collection to delete.

    Returns
    -------
    bool
        ``True`` if the collection was deleted; ``False`` if it did not exist.

    Raises
    ------
    RuntimeError
        If the deletion call fails for reasons other than the collection not
        being found.
    """
    existing = _get_existing_collection_names(client)
    if collection_name not in existing:
        logger.warning(
            "Collection '%s' does not exist — nothing to delete.", collection_name
        )
        return False

    logger.warning("Deleting collection '%s' — all data will be lost.", collection_name)
    try:
        client.delete_collection(collection_name)
        logger.info("Collection '%s' deleted.", collection_name)
        return True
    except Exception as exc:
        raise RuntimeError(
            f"Failed to delete collection '{collection_name}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 6. get_collection_info  (optional helper)
# ---------------------------------------------------------------------------

def get_collection_info(
    client: QdrantClient,
    collection_name: str,
) -> Dict[str, Any]:
    """Return metadata about a Qdrant collection.

    Parameters
    ----------
    client : QdrantClient
        An initialised Qdrant client.
    collection_name : str
        Target collection name.

    Returns
    -------
    dict
        A summary dict with keys ``name``, ``status``, ``vectors_count``,
        ``points_count``, and ``vector_size``.

    Raises
    ------
    RuntimeError
        If the collection does not exist or the call fails.
    """
    try:
        info = client.get_collection(collection_name)
        summary = {
            "name": collection_name,
            "status": str(info.status),
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
        }
        logger.info("Collection info for '%s': %s", collection_name, summary)
        return summary
    except Exception as exc:
        raise RuntimeError(
            f"Could not retrieve info for collection '{collection_name}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_existing_collection_names(client: QdrantClient) -> List[str]:
    """Return a list of existing collection names in the Qdrant instance."""
    try:
        return [c.name for c in client.get_collections().collections]
    except Exception:
        return []


def _validate_upsert_inputs(embeddings: Any, documents: List[str]) -> None:
    """Raise ``ValueError`` for mis-matched or empty upsert inputs."""
    if embeddings is None or documents is None:
        raise ValueError("'embeddings' and 'documents' must not be None.")

    n_emb = len(embeddings)
    n_doc = len(documents)

    if n_emb == 0 or n_doc == 0:
        raise ValueError(
            "'embeddings' and 'documents' must each contain at least one element."
        )

    if n_emb != n_doc:
        raise ValueError(
            f"Length mismatch: {n_emb} embedding(s) vs {n_doc} document(s). "
            "They must be the same length."
        )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as _np

    _COLLECTION = "smoke_test_incidents"
    _DIM = 384

    _sample_docs = [
        "Incident ID: 1023\nDevice: router-1\nAlert Type: CPU High\nSeverity: Critical",
        "Incident ID: 1024\nDevice: switch-3\nAlert Type: Packet Loss\nSeverity: Warning",
        "Incident ID: 1025\nDevice: ap-7\nAlert Type: Access Point Offline\nSeverity: Emergency",
    ]

    # Simulate random embeddings (replace with real embed_documents() output).
    _rng = _np.random.default_rng(42)
    _embeddings = _rng.random((len(_sample_docs), _DIM)).astype(_np.float32)
    # L2-normalise so cosine sim is meaningful.
    _embeddings /= _np.linalg.norm(_embeddings, axis=1, keepdims=True)

    _query_vec = _rng.random(_DIM).astype(_np.float32)
    _query_vec /= _np.linalg.norm(_query_vec)

    print("\n" + "=" * 60)
    print("  Vector Store — Smoke Test (Qdrant in-memory)")
    print("=" * 60 + "\n")

    _client = init_qdrant_client()
    create_collection(_client, _COLLECTION, vector_size=_DIM)
    upsert_documents(_client, _COLLECTION, _embeddings, _sample_docs)

    _info = get_collection_info(_client, _COLLECTION)
    print("Collection info:", _info, "\n")

    _results = search_similar_documents(_client, _COLLECTION, _query_vec, top_k=3)
    print("Search results:")
    for r in _results:
        print(f"  ID={r['id']}  score={r['score']:.4f}  doc={r['document'][:55]} …")

    delete_collection(_client, _COLLECTION)
