"""Main API interface for the AI Incident Intelligence Copilot.

This module provides a stable entrypoint for external callers such as a
Streamlit UI or CLI. It delegates incident reasoning to the agent layer.
"""

from __future__ import annotations

import logging
from typing import Optional

from llm.agents.agent_executor import AgentExecutor


logger = logging.getLogger(__name__)


class CopilotAPI:
	"""Public API wrapper for asking incident-related questions.

	Parameters
	----------
	executor : AgentExecutor, optional
		Pre-initialized executor instance. If omitted, a default executor is
		created.
	"""

	def __init__(self, executor: Optional[AgentExecutor] = None) -> None:
		self.executor = executor or AgentExecutor()
		logger.info("CopilotAPI initialized successfully")

	def ask(self, question: str) -> str:
		"""Process a user question and return an AI-generated answer.

		Steps
		-----
		1. Validate the question.
		2. Execute the agent workflow via :class:`AgentExecutor`.
		3. Return the final response string.

		Parameters
		----------
		question : str
			User question about incidents, root causes, or remediation.

		Returns
		-------
		str
			AI-generated response, or a meaningful error message when execution
			fails.
		"""
		if not isinstance(question, str) or not question.strip():
			logger.warning("Received empty or invalid question")
			return "Invalid question: please provide a non-empty incident question."

		cleaned_question = question.strip()

		try:
			logger.info("CopilotAPI.ask received question: '%s'", cleaned_question[:120])
			answer = self.executor.run(cleaned_question)

			if not answer or not answer.strip():
				logger.error("Agent returned an empty response")
				return (
					"The copilot could not generate an answer at this time. "
					"Please retry your question."
				)

			logger.info("CopilotAPI.ask completed successfully")
			return answer

		except ValueError as exc:
			logger.warning("Question validation failed: %s", exc)
			return f"Invalid question: {exc}"
		except RuntimeError as exc:
			logger.exception("Agent runtime error")
			return f"Copilot execution error: {exc}"
		except Exception as exc:
			logger.exception("Unexpected copilot error")
			return f"Unexpected copilot error: {exc}"


__all__ = ["CopilotAPI"]

