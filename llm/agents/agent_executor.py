"""Execution layer for the AI Incident Intelligence agent.

This module coordinates retriever initialization and agent execution so other
parts of the application can run incident investigations through one entrypoint.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from llm.agents.incident_agent import IncidentAgent
from llm.rag.document_loader import load_incident_documents
from llm.rag.embedding_model import embed_documents
from llm.rag.retriever import Retriever
from llm.rag.vector_store import create_collection, upsert_documents


logger = logging.getLogger(__name__)

DEFAULT_GOLD_JSON_PATH = os.path.normpath(
	os.path.join(
		os.path.dirname(os.path.abspath(__file__)),
		"..",
		"..",
		"analytics",
		"outputs",
		"incident_summary.json",
	)
)


class AgentExecutor:
	"""Coordinator that executes IncidentAgent against a user question.

	Parameters
	----------
	retriever : Retriever, optional
		Pre-built retriever instance. If omitted, a default Retriever is
		created.
	agent : IncidentAgent, optional
		Pre-built agent instance. If omitted, a default IncidentAgent is
		created using the retriever.
	top_k : int, optional
		Default top-k retrieval value when constructing internal components.
		Defaults to ``3``.
	"""

	def __init__(
		self,
		retriever: Optional[Retriever] = None,
		agent: Optional[IncidentAgent] = None,
		top_k: int = 3,
		gold_json_path: Optional[str] = None,
	) -> None:
		if top_k <= 0:
			raise ValueError("top_k must be a positive integer")

		self.top_k = top_k
		self.gold_json_path = gold_json_path or os.getenv("GOLD_LAYER_JSON_PATH", DEFAULT_GOLD_JSON_PATH)

		self.retriever: Retriever = retriever or Retriever(default_top_k=top_k)
		self._bootstrap_gold_layer(self.gold_json_path)
		self.agent: IncidentAgent = agent or IncidentAgent(
			retriever=self.retriever,
			top_k=top_k,
		)

		logger.info(
			"AgentExecutor initialized | top_k=%d | model=%s | tools=%s",
			self.top_k,
			self.agent.model,
			", ".join(sorted(self.agent.tools.keys())),
		)

	def _bootstrap_gold_layer(self, json_path: str) -> None:
		"""Load and index gold-layer analytics JSON into the active retriever store."""
		if not json_path or not os.path.isfile(json_path):
			logger.warning("Gold-layer JSON not found, skipping bootstrap: %s", json_path)
			return

		try:
			logger.info("Bootstrapping retriever from gold-layer JSON: %s", json_path)
			documents = load_incident_documents(json_path)
			if not documents:
				logger.warning("No documents produced from gold-layer JSON: %s", json_path)
				return

			embeddings = embed_documents(documents, self.retriever._model, show_progress=False)
			vector_size = int(embeddings.shape[1])

			create_collection(
				client=self.retriever._client,
				collection_name=self.retriever.collection_name,
				vector_size=vector_size,
				recreate_if_exists=True,
			)
			upsert_documents(
				client=self.retriever._client,
				collection_name=self.retriever.collection_name,
				embeddings=embeddings,
				documents=documents,
			)
			logger.info("Gold-layer bootstrap complete: indexed %d document(s)", len(documents))
		except Exception:
			logger.exception("Failed to bootstrap gold-layer JSON into vector store")

	def run(self, question: str) -> str:
		"""Run the incident reasoning workflow and return final answer.

		Workflow
		--------
		1. Validate user question.
		2. Use initialized ``Retriever`` and ``IncidentAgent``.
		3. Execute agent reasoning via ``IncidentAgent.run(question)``.
		4. Return the final AI answer.

		Parameters
		----------
		question : str
			User question about incident behavior, root cause, or remediation.

		Returns
		-------
		str
			Final LLM-generated answer.

		Raises
		------
		ValueError
			If question is empty.
		RuntimeError
			If execution fails.
		"""
		if not question or not question.strip():
			raise ValueError("question must be a non-empty string")

		try:
			logger.info("AgentExecutor.run started for question: '%s'", question[:120])
			answer = self.agent.run(question)
			logger.info("AgentExecutor.run completed successfully")
			return answer
		except Exception as exc:
			logger.exception("Agent execution failed")
			raise RuntimeError(f"Failed to execute incident agent: {exc}") from exc


__all__ = ["AgentExecutor"]

