"""
llm/rag/retriever.py

RAG Pipeline — Stage 4: Retriever
-----------------------------------
Given a natural-language user query, this module embeds the query, searches
the Qdrant vector store, and returns the most semantically relevant incident
documents as structured results or a ready-to-use LLM context string.

Pipeline position
-----------------
vector_store  ──►  retriever  ──►  LLM / AI Copilot

Dependencies
------------
• llm.rag.embedding_model  — embed_query(), load_embedding_model()
• llm.rag.vector_store     — init_qdrant_client(), search_similar_documents()

Usage
-----
    from llm.rag.retriever import Retriever

    # Preferred: use the stateful Retriever class
    r = Retriever()
    docs    = r.retrieve_similar_documents("Why did the payment API fail?")
    context = r.build_context("Why did the payment API fail?")

    # Or use module-level convenience functions (use a shared default Retriever)
    from llm.rag.retriever import retrieve_similar_documents, build_context
    docs    = retrieve_similar_documents("Why did the payment API fail?")
    context = build_context("Why did the payment API fail?")

Author: Team 15 — KenexAI Hackathon
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from llm.rag.embedding_model import embed_query, load_embedding_model
from llm.rag.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DOCUMENT_PAYLOAD_KEY,
    init_qdrant_client,
    search_similar_documents,
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

DEFAULT_TOP_K: int = 5
CONTEXT_SEPARATOR: str = "\n\n---\n\n"   # Visual separator between documents in context


# ---------------------------------------------------------------------------
# Retriever class (stateful, preferred interface)
# ---------------------------------------------------------------------------

class Retriever:
    """Stateful retriever that owns a Qdrant client and an embedding model.

    Creating a single ``Retriever`` instance and reusing it across calls is
    the most efficient pattern because both the Qdrant connection and the
    embedding model are expensive to initialise.

    Parameters
    ----------
    client : QdrantClient, optional
        Pre-initialised Qdrant client.  If omitted, an in-memory client is
        created automatically.
    model : SentenceTransformer, optional
        Pre-loaded embedding model.  If omitted, ``BAAI/bge-small-en-v1.5``
        is loaded automatically.
    collection_name : str, optional
        The Qdrant collection to search.  Defaults to ``"incidents"``.
    default_top_k : int, optional
        Default number of results returned when *top_k* is not specified in
        individual calls.  Defaults to ``5``.

    Examples
    --------
    >>> r = Retriever()
    >>> docs = r.retrieve_similar_documents("Why did router-1 go offline?")
    >>> context = r.build_context("Why did router-1 go offline?", top_k=3)
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        model: Optional[SentenceTransformer] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.collection_name = collection_name
        self.default_top_k = default_top_k

        logger.info("Initialising Retriever — collection: '%s'.", collection_name)

        self._client: QdrantClient = client or init_qdrant_client()
        self._model: SentenceTransformer = model or load_embedding_model()

        logger.info("Retriever ready.")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def retrieve_similar_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Retrieve the most relevant incident document strings for *query*.

        Steps
        -----
        1. Validate and clean the query string.
        2. Embed the query with ``embed_query()`` (BGE prefix applied internally).
        3. Search the Qdrant collection for the top-*k* nearest vectors.
        4. Extract and return the document text from each result payload.

        Parameters
        ----------
        query : str
            Natural-language question from the engineer or AI copilot.
        top_k : int, optional
            Number of documents to retrieve.  Falls back to ``self.default_top_k``.

        Returns
        -------
        List[str]
            Ranked list of incident document strings (most similar first).
            Returns an empty list if the collection is empty or no match is found.

        Raises
        ------
        ValueError
            If *query* is ``None`` or an empty string.

        Example
        -------
        >>> docs = retriever.retrieve_similar_documents("Why did the API fail?")
        >>> for d in docs:
        ...     print(d[:80])
        """
        _validate_query(query)
        k = top_k if top_k is not None else self.default_top_k

        logger.info("Embedding query for retrieval — top_k=%d.", k)
        query_embedding = embed_query(query, self._model)

        logger.info(
            "Searching Qdrant collection '%s' — top_k=%d.", self.collection_name, k
        )
        raw_results = search_similar_documents(
            client=self._client,
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            top_k=k,
        )

        documents = _extract_documents(raw_results)

        if not documents:
            logger.warning(
                "No documents retrieved from '%s' for query: '%s …'.",
                self.collection_name,
                query[:60],
            )
        else:
            logger.info(
                "Retrieved %d document(s) — top score: %.4f.",
                len(documents),
                raw_results[0]["score"] if raw_results else 0.0,
            )

        return documents

    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = CONTEXT_SEPARATOR,
    ) -> str:
        """Build a single context string ready to be injected into an LLM prompt.

        Calls :meth:`retrieve_similar_documents` and joins the results with
        *separator* so the LLM receives a clear, readable block of evidence.

        Parameters
        ----------
        query : str
            Natural-language question.
        top_k : int, optional
            Number of documents to retrieve.  Defaults to ``self.default_top_k``.
        separator : str, optional
            String inserted between consecutive documents.
            Defaults to ``"\\n\\n---\\n\\n"``.

        Returns
        -------
        str
            Combined context text.  Returns an empty string when no relevant
            documents are found.

        Example
        -------
        >>> ctx = retriever.build_context("What caused the VPN outage?")
        >>> print(ctx)
        Incident ID: 1041 ...
        ---
        Incident ID: 1082 ...
        """
        documents = self.retrieve_similar_documents(query, top_k=top_k)
        context = separator.join(documents)

        logger.info(
            "Context built — %d document(s), %d characters.",
            len(documents), len(context),
        )
        return context

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents together with their similarity scores.

        Useful when the caller needs to apply a score threshold or display
        confidence values in the UI.

        Parameters
        ----------
        query : str
            Natural-language question.
        top_k : int, optional
            Number of results.  Defaults to ``self.default_top_k``.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict has keys ``"id"``, ``"score"``, ``"document"``.
            Results are sorted by *score* descending (Qdrant guarantees this).
        """
        _validate_query(query)
        k = top_k if top_k is not None else self.default_top_k

        logger.info("Embedding query (scored retrieval) — top_k=%d.", k)
        query_embedding = embed_query(query, self._model)

        logger.info("Searching Qdrant collection '%s'.", self.collection_name)
        raw_results = search_similar_documents(
            client=self._client,
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            top_k=k,
        )

        structured = format_retrieved_results(raw_results)
        logger.info("Scored retrieval returned %d result(s).", len(structured))
        return structured


# ---------------------------------------------------------------------------
# 1. retrieve_similar_documents (module-level convenience)
# ---------------------------------------------------------------------------

def retrieve_similar_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    client: Optional[QdrantClient] = None,
    model: Optional[SentenceTransformer] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> List[str]:
    """Module-level convenience wrapper around :meth:`Retriever.retrieve_similar_documents`.

    Creates a one-shot ``Retriever`` using the provided (or default) client and
    model.  For repeated calls, prefer constructing a ``Retriever`` instance
    directly to avoid re-initialising the model on every call.

    Parameters
    ----------
    query : str
        Natural-language question.
    top_k : int, optional
        Number of documents to retrieve.  Defaults to ``5``.
    client : QdrantClient, optional
        Qdrant client.  An in-memory client is created if omitted.
    model : SentenceTransformer, optional
        Embedding model.  ``BAAI/bge-small-en-v1.5`` is loaded if omitted.
    collection_name : str, optional
        Qdrant collection name.  Defaults to ``"incidents"``.

    Returns
    -------
    List[str]
        Ranked list of incident document strings.

    Example
    -------
    >>> docs = retrieve_similar_documents("Why did router-1 go offline?", top_k=3)
    """
    r = Retriever(
        client=client,
        model=model,
        collection_name=collection_name,
        default_top_k=top_k,
    )
    return r.retrieve_similar_documents(query, top_k=top_k)


# ---------------------------------------------------------------------------
# 2. build_context (module-level convenience)
# ---------------------------------------------------------------------------

def build_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    client: Optional[QdrantClient] = None,
    model: Optional[SentenceTransformer] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    separator: str = CONTEXT_SEPARATOR,
) -> str:
    """Module-level convenience wrapper around :meth:`Retriever.build_context`.

    Parameters
    ----------
    query : str
        Natural-language question.
    top_k : int, optional
        Number of documents to include in the context.  Defaults to ``5``.
    client : QdrantClient, optional
        Qdrant client.  An in-memory client is created if omitted.
    model : SentenceTransformer, optional
        Embedding model.  ``BAAI/bge-small-en-v1.5`` is loaded if omitted.
    collection_name : str, optional
        Qdrant collection name.  Defaults to ``"incidents"``.
    separator : str, optional
        Separator inserted between documents.

    Returns
    -------
    str
        LLM-ready context string.

    Example
    -------
    >>> ctx = build_context("What caused the VPN outage?")
    """
    r = Retriever(
        client=client,
        model=model,
        collection_name=collection_name,
        default_top_k=top_k,
    )
    return r.build_context(query, top_k=top_k, separator=separator)


# ---------------------------------------------------------------------------
# 3. format_retrieved_results
# ---------------------------------------------------------------------------

def format_retrieved_results(
    raw_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert raw Qdrant search results into clean structured dictionaries.

    ``search_similar_documents`` from ``vector_store.py`` already returns
    dicts, but this function guarantees a stable, well-typed schema and fills
    in default values for any missing keys.

    Parameters
    ----------
    raw_results : List[Dict[str, Any]]
        Raw output from :func:`~llm.rag.vector_store.search_similar_documents`.
        Each item is expected to contain ``"id"``, ``"score"``, and
        ``"document"`` keys.

    Returns
    -------
    List[Dict[str, Any]]
        List of dicts guaranteed to have the following keys:

        ``id``
            Point ID in Qdrant (``int``).
        ``score``
            Cosine similarity score rounded to 6 decimal places (``float``).
        ``document``
            Raw document text (``str``).

    Example
    -------
    >>> results = format_retrieved_results(raw_qdrant_hits)
    >>> print(results[0])
    {"id": 1, "score": 0.9213, "document": "Incident ID: 1023 ..."}
    """
    if not raw_results:
        logger.warning("format_retrieved_results received an empty result list.")
        return []

    formatted: List[Dict[str, Any]] = []
    for item in raw_results:
        formatted.append(
            {
                "id": item.get("id", -1),
                "score": round(float(item.get("score", 0.0)), 6),
                "document": item.get("document", item.get(DOCUMENT_PAYLOAD_KEY, "")),
            }
        )

    logger.debug("Formatted %d retrieval result(s).", len(formatted))
    return formatted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_query(query: Any) -> None:
    """Raise ``ValueError`` when *query* is None or blank."""
    if not query or not str(query).strip():
        raise ValueError(
            "'query' must be a non-empty string. "
            f"Received: {repr(query)}"
        )


def _extract_documents(raw_results: List[Dict[str, Any]]) -> List[str]:
    """Pull the document text string out of each Qdrant result dict."""
    docs: List[str] = []
    for item in raw_results:
        text = item.get("document", item.get(DOCUMENT_PAYLOAD_KEY, "")).strip()
        if text:
            docs.append(text)
        else:
            logger.debug("Result id=%s had no document text — skipping.", item.get("id"))
    return docs


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as _np
    from llm.rag.vector_store import create_collection, upsert_documents

    _COLLECTION = "smoke_test_retriever"
    _DIM = 384

    _sample_docs = [
        "Incident ID: 1023\nDevice: router-1\nAlert Type: CPU High\nSeverity: Critical\nDescription: CPU usage exceeded 95% threshold",
        "Incident ID: 1024\nDevice: switch-3\nAlert Type: Packet Loss\nSeverity: Warning\nDescription: 12% packet loss on uplink interface",
        "Incident ID: 1025\nDevice: ap-7\nAlert Type: Access Point Offline\nSeverity: Emergency\nDescription: Ubiquiti U6+ has gone offline",
        "Incident ID: 1026\nDevice: asa5515x-01\nAlert Type: VPN Gateway Lost\nSeverity: Critical\nDescription: VPN remote gateway 12.207.114.41 unreachable",
        "Incident ID: 1027\nDevice: MX68-FW\nAlert Type: VPN Connectivity Changed\nSeverity: Warning\nDescription: VPN tunnel status changed on JB-CLACKAMAS-FW",
    ]

    print("\n" + "=" * 60)
    print("  Retriever — Smoke Test")
    print("=" * 60 + "\n")

    # Build a client + model manually so the Retriever reuses them.
    _client   = init_qdrant_client()
    _model    = load_embedding_model()

    # Seed the vector store with sample embeddings.
    from llm.rag.embedding_model import embed_documents as _embed_docs
    _embeddings = _embed_docs(_sample_docs, _model)
    create_collection(_client, _COLLECTION, vector_size=_DIM, recreate_if_exists=True)
    upsert_documents(_client, _COLLECTION, _embeddings, _sample_docs)

    # Instantiate retriever with the pre-built client + model.
    _retriever = Retriever(
        client=_client,
        model=_model,
        collection_name=_COLLECTION,
        default_top_k=3,
    )

    _query = "VPN gateway is unreachable"
    print(f"Query: '{_query}'\n")

    print("── retrieve_similar_documents ──")
    _docs = _retriever.retrieve_similar_documents(_query)
    for i, d in enumerate(_docs, 1):
        print(f"  [{i}] {d[:80].replace(chr(10), ' ')} …")

    print("\n── retrieve_with_scores ──")
    _scored = _retriever.retrieve_with_scores(_query)
    for r in _scored:
        print(f"  ID={r['id']}  score={r['score']:.4f}  {r['document'][:60].replace(chr(10),' ')} …")

    print("\n── build_context (joined string) ──")
    _ctx = _retriever.build_context(_query, top_k=2, separator="\n---\n")
    print(_ctx)
