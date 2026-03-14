"""
llm/rag/embedding_model.py

RAG Pipeline — Stage 2: Embedding Model
-----------------------------------------
Converts plain-text incident documents and user queries into dense vector
embeddings using the BAAI/bge-small-en-v1.5 sentence-transformer model.

Pipeline position
-----------------
document_loader  ──►  embedding_model  ──►  vector_store  ──►  retriever

Usage
-----
    from llm.rag.embedding_model import load_embedding_model, embed_documents, embed_query

    model = load_embedding_model()
    doc_embeddings = embed_documents(documents, model)   # (N, 384) ndarray
    query_embedding = embed_query("Why did the API fail?", model)  # (384,) ndarray

Author: Team 15 — KenexAI Hackathon
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

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

DEFAULT_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

# BGE models require this instruction prefix on queries (not on documents)
# to align the query embedding space with the passage embedding space.
BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

# Default batch size for encoding large document collections.
DEFAULT_BATCH_SIZE: int = 64

# Module-level cache — avoids re-loading the model inside the same process.
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


# ---------------------------------------------------------------------------
# 1. load_embedding_model
# ---------------------------------------------------------------------------

def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformer:
    """Load (or return a cached) SentenceTransformer embedding model.

    The model is cached at the module level so repeated calls within the same
    Python process return the same instance without re-downloading weights.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to ``"BAAI/bge-small-en-v1.5"``.

    Returns
    -------
    SentenceTransformer
        A ready-to-use sentence-transformer model.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded (e.g. network error, invalid name).

    Example
    -------
    >>> model = load_embedding_model()
    >>> model = load_embedding_model("BAAI/bge-small-en-v1.5")
    """
    if model_name in _MODEL_CACHE:
        logger.info("Returning cached embedding model: '%s'.", model_name)
        return _MODEL_CACHE[model_name]

    logger.info("Loading embedding model: '%s' …", model_name)
    try:
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
        dim = get_embedding_dimension(model)
        logger.info(
            "Model '%s' loaded successfully — embedding dimension: %d.",
            model_name, dim,
        )
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 2. embed_documents
# ---------------------------------------------------------------------------

def embed_documents(
    documents: List[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """Convert a list of text documents into a 2-D embedding matrix.

    Documents are encoded **without** the BGE query prefix — they represent
    passages/content rather than search queries.

    Parameters
    ----------
    documents : List[str]
        Incident documents produced by ``document_loader.build_incident_documents``.
    model : SentenceTransformer
        A loaded embedding model (from :func:`load_embedding_model`).
    batch_size : int, optional
        Number of documents encoded per forward pass.  Larger values are faster
        on GPU; smaller values are safer on memory-constrained machines.
        Defaults to ``64``.
    normalize : bool, optional
        Whether to L2-normalise the output embeddings.  Normalisation is
        recommended for cosine-similarity search.  Defaults to ``True``.
    show_progress : bool, optional
        Show a tqdm progress bar during encoding.  Useful for large corpora.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(documents), embedding_dimension)``.

    Raises
    ------
    ValueError
        If *documents* is ``None`` or empty.

    Example
    -------
    >>> model = load_embedding_model()
    >>> docs = ["Incident ID: 1\nDevice: router-1\nSeverity: Critical"]
    >>> embeddings = embed_documents(docs, model)
    >>> embeddings.shape
    (1, 384)
    """
    _validate_documents(documents)

    # Filter out blank strings and warn about them.
    clean_docs, skipped = _filter_empty(documents)
    if skipped:
        logger.warning("%d blank document(s) were skipped before embedding.", skipped)

    logger.info(
        "Embedding %d document(s) — batch_size=%d, normalize=%s.",
        len(clean_docs), batch_size, normalize,
    )

    embeddings: np.ndarray = model.encode(
        clean_docs,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    logger.info(
        "Documents embedded — output shape: %s.", embeddings.shape
    )
    return embeddings


# ---------------------------------------------------------------------------
# 3. embed_query
# ---------------------------------------------------------------------------

def embed_query(
    query: str,
    model: SentenceTransformer,
    normalize: bool = True,
) -> np.ndarray:
    """Convert a user query string into a 1-D embedding vector.

    BGE models require a specific instruction prefix on queries to align the
    query embedding space with the passage embedding space.  This function
    prepends that prefix automatically.

    Parameters
    ----------
    query : str
        Raw natural-language question from the engineer / AI copilot.
    model : SentenceTransformer
        A loaded embedding model (from :func:`load_embedding_model`).
    normalize : bool, optional
        L2-normalise the output vector.  Defaults to ``True``.

    Returns
    -------
    np.ndarray
        1-D embedding vector of shape ``(embedding_dimension,)``.

    Raises
    ------
    ValueError
        If *query* is ``None`` or an empty string.

    Example
    -------
    >>> model = load_embedding_model()
    >>> vec = embed_query("Why did the payment API fail today?", model)
    >>> vec.shape
    (384,)
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Prepend BGE's retrieval instruction to the query.
    prefixed_query = f"{BGE_QUERY_PREFIX}{query.strip()}"
    logger.debug("Embedding query (with BGE prefix): '%s'", prefixed_query[:120])

    embedding: np.ndarray = model.encode(
        prefixed_query,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    # model.encode returns (dim,) for a single string — ensure 1-D.
    embedding = np.squeeze(embedding)
    logger.info("Query embedded — vector shape: %s.", embedding.shape)
    return embedding


# ---------------------------------------------------------------------------
# 4. get_embedding_dimension
# ---------------------------------------------------------------------------

def get_embedding_dimension(model: SentenceTransformer) -> int:
    """Return the output embedding dimension of the loaded model.

    Parameters
    ----------
    model : SentenceTransformer
        A loaded embedding model.

    Returns
    -------
    int
        Number of dimensions in each embedding vector.
        For ``BAAI/bge-small-en-v1.5`` this is ``384``.

    Example
    -------
    >>> model = load_embedding_model()
    >>> get_embedding_dimension(model)
    384
    """
    dim: int = model.get_sentence_embedding_dimension()
    logger.debug("Embedding dimension: %d.", dim)
    return dim


# ---------------------------------------------------------------------------
# 5. normalize_embeddings  (optional helper)
# ---------------------------------------------------------------------------

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise a batch of embedding vectors in-place safe manner.

    Normalised embeddings allow cosine similarity to be computed as a simple
    dot product, which is faster in most vector stores.

    Parameters
    ----------
    embeddings : np.ndarray
        2-D array of shape ``(N, D)`` or 1-D array of shape ``(D,)``.

    Returns
    -------
    np.ndarray
        Unit-normalised embeddings with the same shape as input.

    Raises
    ------
    ValueError
        If *embeddings* is not a NumPy array or has unsupported dimensions.

    Example
    -------
    >>> normed = normalize_embeddings(raw_embeddings)
    >>> np.allclose(np.linalg.norm(normed, axis=-1), 1.0)
    True
    """
    if not isinstance(embeddings, np.ndarray):
        raise ValueError(
            f"Expected a NumPy ndarray, got {type(embeddings).__name__}."
        )
    if embeddings.ndim not in (1, 2):
        raise ValueError(
            f"Expected 1-D or 2-D array, got shape {embeddings.shape}."
        )

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    # Avoid division by zero for zero vectors.
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_documents(documents: Optional[List[str]]) -> None:
    """Raise ``ValueError`` if the document list is None or empty."""
    if documents is None:
        raise ValueError("'documents' must not be None.")
    if len(documents) == 0:
        raise ValueError("'documents' must contain at least one string.")


def _filter_empty(documents: List[str]) -> tuple[List[str], int]:
    """Remove blank strings from a document list.

    Returns
    -------
    tuple
        ``(clean_documents, number_of_skipped)``
    """
    clean = [d for d in documents if d and d.strip()]
    skipped = len(documents) - len(clean)
    return clean, skipped


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sample_docs = [
        "Incident ID: 1023\nDevice: router-1\nAlert Type: CPU High\nSeverity: Critical",
        "Incident ID: 1024\nDevice: switch-3\nAlert Type: Packet Loss\nSeverity: Warning",
        "Incident ID: 1025\nDevice: ap-7\nAlert Type: Access Point Offline\nSeverity: Emergency",
    ]
    _query = "Why did the payment API fail today?"

    print("\n" + "=" * 60)
    print("  Embedding Model — Smoke Test")
    print("=" * 60 + "\n")

    _model = load_embedding_model()

    print(f"Embedding dimension : {get_embedding_dimension(_model)}\n")

    _doc_embeddings = embed_documents(_sample_docs, _model, show_progress=True)
    print(f"Document embeddings : {_doc_embeddings.shape}\n")

    _query_embedding = embed_query(_query, _model)
    print(f"Query embedding     : {_query_embedding.shape}\n")

    # Quick cosine similarity between the query and each document.
    similarities = _doc_embeddings @ _query_embedding
    print("Cosine similarities (query ↔ each document):")
    for i, sim in enumerate(similarities):
        print(f"  Doc {i + 1}: {sim:.4f}")
