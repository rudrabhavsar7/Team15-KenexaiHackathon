"""Incident Intelligence agent for RAG-based incident investigation.

This module defines a reusable agent that orchestrates retrieval tools and a
Groq LLM to help engineers investigate infrastructure incidents.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

from llm.agents.tools import explain_incident, get_recent_incidents, search_incidents
from llm.rag.retriever import Retriever


logger = logging.getLogger(__name__)


# Load llm/.env automatically so GROQ_API_KEY and GROQ_MODEL can be resolved.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.normpath(os.path.join(_HERE, "..", ".env"))
if os.path.isfile(_ENV_FILE):
	load_dotenv(dotenv_path=_ENV_FILE, override=False)


SYSTEM_PROMPT = """You are an AI Incident Intelligence assistant for Site Reliability Engineers.

Your responsibilities:
- Analyze incident alerts and correlated events from infrastructure monitoring systems.
- Identify the most likely root cause using the provided incident evidence.
- Recommend practical next actions for triage, diagnosis, escalation, and prevention.
- Rely on incident data and retrieved context; do not invent unavailable facts.

Output style:
- Be concise, structured, and operationally useful.
- Include uncertainty when evidence is insufficient.
- Prioritize engineer actions by urgency.
"""


REASONING_INSTRUCTIONS = """Reasoning workflow:
1) Retrieve relevant incidents for the user's question.
2) Summarize the newest/high-signal incidents.
3) Generate an explanation grounded in context.
4) Return actionable recommendations and confidence notes.
"""


class IncidentAgent:
	"""Reusable incident investigation agent.

	The agent wires together retrieval tools and Groq generation with a stable
	prompt contract so it can be used from APIs, CLIs, or orchestration layers.

	Parameters
	----------
	retriever : Retriever, optional
		Pre-initialized retriever instance. If omitted, a default Retriever is
		created.
	model : str, optional
		Groq model name. Defaults to ``llama-3.1-8b-instant`` or ``GROQ_MODEL``.
	top_k : int, optional
		Number of documents retrieved during search. Defaults to ``3``.
	temperature : float, optional
		LLM sampling temperature. Defaults to ``0.2``.
	groq_client : Groq, optional
		Pre-initialized Groq client. If omitted, a new client is created.
	"""

	def __init__(
		self,
		retriever: Optional[Retriever] = None,
		model: Optional[str] = None,
		top_k: int = 3,
		temperature: float = 0.2,
		groq_client: Optional[Groq] = None,
	) -> None:
		self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
		self.top_k = top_k
		self.temperature = temperature

		self.retriever = retriever or Retriever(default_top_k=top_k)
		self.groq_client = groq_client or Groq()

		self.tools: Dict[str, Callable[..., Any]] = {
			"search_incidents": search_incidents,
			"explain_incident": explain_incident,
			"get_recent_incidents": get_recent_incidents,
		}

		logger.info(
			"IncidentAgent initialized | model=%s | top_k=%d | temperature=%.2f",
			self.model,
			self.top_k,
			self.temperature,
		)

	def run(self, question: str) -> str:
		"""Investigate an incident question and return an intent-adaptive response.

		The method orchestrates the registered tools and returns the summarizer
		output directly so intent-specific prompt formatting is preserved.

		Parameters
		----------
		question : str
			User question about incident behavior, root cause, or remediation.

		Returns
		-------
		str
			Agent response containing grounded analysis and recommendations.

		Raises
		------
		ValueError
			If question is empty.
		RuntimeError
			If tool execution or LLM generation fails.
		"""
		if not question or not question.strip():
			raise ValueError("question must be a non-empty string")

		try:
			logger.info("IncidentAgent.run started for question: '%s'", question[:120])

			context = self.tools["search_incidents"](question, self.retriever, top_k=self.top_k)
			_ = self.tools["get_recent_incidents"](context)

			# Reuse the project summarizer tool for consistent RAG behavior.
			rag_explanation = self.tools["explain_incident"](
				question,
				retriever=self.retriever,
				top_k=self.top_k,
			)

			answer = rag_explanation or ""
			if not answer.strip():
				raise RuntimeError("Groq returned an empty response")

			logger.info("IncidentAgent.run completed successfully")
			return answer

		except Exception as exc:
			logger.exception("IncidentAgent.run failed")
			raise RuntimeError(f"Incident agent failed: {exc}") from exc

	def run_with_metadata(self, question: str) -> Dict[str, str]:
		"""Run the incident flow and return response plus intermediate artifacts."""
		if not question or not question.strip():
			raise ValueError("question must be a non-empty string")

		context = self.tools["search_incidents"](question, self.retriever, top_k=self.top_k)
		recent_summary = self.tools["get_recent_incidents"](context)
		rag_explanation = self.tools["explain_incident"](
			question,
			retriever=self.retriever,
			top_k=self.top_k,
		)
		final_answer = self.run(question)

		return {
			"question": question,
			"context": context,
			"recent_incidents": recent_summary,
			"rag_explanation": rag_explanation,
			"answer": final_answer,
		}


__all__ = [
	"SYSTEM_PROMPT",
	"REASONING_INSTRUCTIONS",
	"IncidentAgent",
]

