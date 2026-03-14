"""Agent tools for incident search and explanation.

This module provides small, composable tools used by the agent layer.
It intentionally reuses the existing RAG components and does not
reimplement embeddings or vector search.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Tuple

from llm.summarizer import generate_answer


logger = logging.getLogger(__name__)


def search_incidents(question: str, retriever, top_k: int = 3) -> str:
	"""Search for incidents relevant to a question using a Retriever.

	Parameters
	----------
	question : str
		User question to search for.
	retriever : Retriever
		Initialised retriever instance that exposes ``build_context``.
	top_k : int, optional
		Number of incident documents to retrieve. Defaults to ``3``.

	Returns
	-------
	str
		Retrieved context string containing relevant incident records.

	Raises
	------
	ValueError
		If question is empty or top_k is invalid.
	TypeError
		If retriever does not expose ``build_context``.
	RuntimeError
		If retrieval fails.
	"""
	if not question or not question.strip():
		raise ValueError("Question must be a non-empty string.")
	if top_k <= 0:
		raise ValueError("top_k must be a positive integer.")
	if not hasattr(retriever, "build_context"):
		raise TypeError("retriever must provide a build_context(question, top_k) method.")

	try:
		logger.info("Searching incidents for question: '%s' (top_k=%d)", question[:120], top_k)
		context = retriever.build_context(question, top_k=top_k)
		if not context or not context.strip():
			logger.warning("No incident context returned for question: '%s'", question[:120])
			return ""

		logger.info("Incident search complete. Context size=%d chars", len(context))
		return context
	except Exception as exc:
		logger.exception("Incident search failed.")
		raise RuntimeError(f"Failed to search incidents: {exc}") from exc


def explain_incident(question: str, retriever=None, top_k: int = 3) -> str:
	"""Generate a full incident explanation using the RAG summarizer.

	Parameters
	----------
	question : str
		User question about an incident.
	retriever : Retriever, optional
		Retriever instance to reuse indexed context from the active vector store.
	top_k : int, optional
		Number of documents to retrieve for generation. Defaults to ``3``.

	Returns
	-------
	str
		Structured LLM-generated answer from the existing summarizer.

	Raises
	------
	ValueError
		If question is empty.
	RuntimeError
		If explanation generation fails.
	"""
	if not question or not question.strip():
		raise ValueError("Question must be a non-empty string.")
	if top_k <= 0:
		raise ValueError("top_k must be a positive integer.")

	try:
		logger.info("Generating incident explanation for question: '%s'", question[:120])
		answer = generate_answer(question, top_k=top_k, retriever=retriever)
		logger.info("Incident explanation generated. Answer size=%d chars", len(answer))
		return answer
	except Exception as exc:
		logger.exception("Incident explanation generation failed.")
		raise RuntimeError(f"Failed to explain incident: {exc}") from exc


def get_recent_incidents(context: str, max_items: int = 3) -> str:
	"""Return a short summary of the most recent incidents from context text.

	The function parses incident blocks from retriever context output and sorts
	them by ``Timestamp`` (newest first) when available.

	Parameters
	----------
	context : str
		Retriever context text containing one or more incident documents.
	max_items : int, optional
		Number of incidents to include in the summary. Defaults to ``3``.

	Returns
	-------
	str
		Human-readable summary of recent incidents.

	Raises
	------
	ValueError
		If ``max_items`` is not positive.
	"""
	if max_items <= 0:
		raise ValueError("max_items must be a positive integer.")

	if not context or not context.strip():
		return "No incidents found in context."

	try:
		incidents = _parse_incident_blocks(context)
		if not incidents:
			return "No parseable incidents found in context."

		incidents_sorted = sorted(
			incidents,
			key=lambda item: (item[0] is not None, item[0] or datetime.min),
			reverse=True,
		)

		selected = incidents_sorted[:max_items]
		lines: List[str] = [f"Recent Incidents (top {len(selected)}):"]
		for idx, (_, fields) in enumerate(selected, start=1):
			lines.append(
				f"{idx}. Incident ID: {fields.get('Incident ID', 'N/A')} | "
				f"Device: {fields.get('Device', 'N/A')} | "
				f"Alert: {fields.get('Alert Type', 'N/A')} | "
				f"Severity: {fields.get('Severity', 'N/A')} | "
				f"Timestamp: {fields.get('Timestamp', 'N/A')}"
			)

		summary = "\n".join(lines)
		logger.info("Built recent incident summary with %d item(s)", len(selected))
		return summary
	except Exception:
		logger.exception("Failed to summarize recent incidents.")
		return "Failed to summarize recent incidents from context."


def _parse_incident_blocks(context: str) -> List[Tuple[datetime | None, Dict[str, str]]]:
	"""Parse incident documents from context and extract key fields."""
	blocks = [b.strip() for b in context.split("\n\n---\n\n") if b.strip()]
	parsed: List[Tuple[datetime | None, Dict[str, str]]] = []

	for block in blocks:
		fields: Dict[str, str] = {}
		for raw_line in block.splitlines():
			line = raw_line.strip()
			if not line or ":" not in line:
				continue
			key, value = line.split(":", 1)
			fields[key.strip()] = value.strip()

		timestamp = _parse_iso_timestamp(fields.get("Timestamp"))
		parsed.append((timestamp, fields))

	return parsed


def _parse_iso_timestamp(value: str | None) -> datetime | None:
	"""Parse ISO timestamp text safely; return None when unavailable/invalid."""
	if not value:
		return None

	text = value.strip()
	if not text:
		return None

	# Handle UTC suffix styles accepted in monitoring exports.
	if text.endswith("Z"):
		text = text[:-1] + "+00:00"

	try:
		return datetime.fromisoformat(text)
	except ValueError:
		return None


__all__ = [
	"search_incidents",
	"explain_incident",
	"get_recent_incidents",
]

