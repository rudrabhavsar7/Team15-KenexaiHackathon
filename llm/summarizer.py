"""
llm/summarizer.py

RAG Pipeline — Final Stage: LLM Summarizer (Groq)
---------------------------------------------------
Orchestrates the complete RAG answer-generation loop:

  1. Receive a user question
  2. Retrieve relevant incident context via build_context()
  3. Load the prompt template from disk
  4. Fill the template with context and question
  5. Send the assembled prompt to the Groq LLM
  6. Return the generated incident analysis

Pipeline position
-----------------
retriever (build_context)  ──►  summarizer  ──►  AI Copilot response

LLM provider : Groq  (https://groq.com)
Model        : llama-3.1-8b-instant
SDK          : groq         (pip install groq)

Environment variables  (.env file or shell export)
---------------------------------------------------
GROQ_API_KEY      — required  — Groq API authentication key
GROQ_MODEL        — optional  — override the default model name
GROQ_TOP_K        — optional  — number of RAG documents to retrieve (default 5)
GROQ_TEMPERATURE  — optional  — LLM sampling temperature              (default 0.2)

.env file
---------
Create  llm/.env  (same folder as this file) with:

    GROQ_API_KEY="gsk_..."

The module loads it automatically at import time via python-dotenv.

Usage
-----
    from llm.summarizer import generate_answer

    answer = generate_answer("Why did the VPN gateway go offline?")
    print(answer)

Author: Team 15 — KenexAI Hackathon
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from llm.rag.retriever import Retriever, build_context

# ---------------------------------------------------------------------------
# Load .env automatically
# ---------------------------------------------------------------------------
# Looks for a .env file in the same directory as this file (llm/.env).
# Falls back gracefully if the file does not exist or dotenv is unavailable.
_ENV_FILE: Path = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.is_file():
    load_dotenv(dotenv_path=_ENV_FILE, override=False)  # override=False: shell exports win
    logging.getLogger(__name__).debug("Loaded env vars from '%s'.", _ENV_FILE)

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
# Configuration
# ---------------------------------------------------------------------------

# Resolve the prompt template path relative to this file so the module works
# regardless of the working directory the caller uses.
_HERE: Path = Path(__file__).resolve().parent

PROMPT_TEMPLATE_PATH: Path = _HERE / "prompt_templates" / "incident_prompt.txt"

DEFAULT_MODEL: str       = os.getenv("GROQ_MODEL",       "llama-3.1-8b-instant")
DEFAULT_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
DEFAULT_TOP_K: int       = int(os.getenv("GROQ_TOP_K",   "5"))

# Placeholder tokens expected inside the prompt template file.
_CONTEXT_TOKEN:  str = "{context}"
_QUESTION_TOKEN: str = "{question}"

# Fallback inserted into {context} when the retriever returns nothing.
_EMPTY_CONTEXT_FALLBACK: str = (
    "No relevant incident records were retrieved from the knowledge base "
    "for this query. Please base your response on general infrastructure "
    "best practices and state clearly that no specific incident data was found."
)

_MISSING_INFO_BULLET: str = "- The available incident context does not contain this information."

_SECTION_HEADER_BULLETS = {
    "Devices",
    "Related Alert Types",
    "Top Issues",
    "Most Affected Entities",
    "Recommended Actions",
}

_NON_GROUNDED_ALLOWED_KEYS = {
    "Intent",
    "Question Scope",
}

_VALID_SCHEMA_KEYS = {
    "Intent",
    "Question Scope",
    "Total Alerts",
    "Organization",
    "Critical Alerts",
    "Devices",
    "Likely Cause",
    "Related Alert Types",
    "Incident Count",
    "Top Issues",
    "Most Affected Entities",
    "Recommended Actions",
}


# ---------------------------------------------------------------------------
# 1. load_prompt_template
# ---------------------------------------------------------------------------

def load_prompt_template(
    template_path: Path = PROMPT_TEMPLATE_PATH,
) -> str:
    """Load the incident analysis prompt template from disk.

    Reads the file at *template_path* and validates that it contains the
    mandatory ``{context}`` and ``{question}`` placeholders.

    Parameters
    ----------
    template_path : Path, optional
        Path to the ``.txt`` prompt template.
        Defaults to ``llm/prompt_templates/incident_prompt.txt``.

    Returns
    -------
    str
        Raw template string with ``{context}`` and ``{question}`` placeholders
        ready to be filled by :func:`build_prompt`.

    Raises
    ------
    FileNotFoundError
        If the template file does not exist at *template_path*.
    ValueError
        If the template is missing one or both required placeholders.

    Example
    -------
    >>> template = load_prompt_template()
    >>> "{context}" in template
    True
    """
    logger.info("Loading prompt template: %s", template_path)

    if not template_path.is_file():
        raise FileNotFoundError(
            f"Prompt template not found: '{template_path}'.\n"
            "Ensure 'llm/prompt_templates/incident_prompt.txt' exists."
        )

    template: str = template_path.read_text(encoding="utf-8")

    # Guard: both placeholders must be present.
    missing = [t for t in (_CONTEXT_TOKEN, _QUESTION_TOKEN) if t not in template]
    if missing:
        raise ValueError(
            f"Prompt template is missing required placeholder(s): {missing}.\n"
            f"Template path: '{template_path}'"
        )

    logger.info("Template loaded — %d characters.", len(template))
    return template


# ---------------------------------------------------------------------------
# 2. build_prompt
# ---------------------------------------------------------------------------

def build_prompt(
    context: str,
    question: str,
    template: Optional[str] = None,
) -> str:
    """Assemble the final LLM prompt by substituting template placeholders.

    Replaces ``{context}`` with the retrieved incident context block and
    ``{question}`` with the engineer's natural-language question.

    If *context* is empty or blank a safe fallback message is used so the
    LLM still receives a coherent, non-broken prompt.

    Parameters
    ----------
    context : str
        Incident documents joined into a single text block
        (typically the output of ``retriever.build_context()``).
    question : str
        The engineer's natural-language question.
    template : str, optional
        Pre-loaded prompt template.  When ``None``, the template is read
        from disk automatically via :func:`load_prompt_template`.

    Returns
    -------
    str
        Fully assembled prompt string with no remaining placeholders.

    Raises
    ------
    ValueError
        If *question* is ``None`` or empty.

    Example
    -------
    >>> prompt = build_prompt("Incident ID: 1023 ...", "Why did router-1 fail?")
    >>> "{context}" not in prompt and "{question}" not in prompt
    True
    """
    _validate_question(question)

    if not context or not context.strip():
        logger.warning(
            "Empty context — using fallback message in prompt."
        )
        context = _EMPTY_CONTEXT_FALLBACK

    tmpl = template if template is not None else load_prompt_template()

    prompt = tmpl.replace(_CONTEXT_TOKEN, context.strip())
    prompt = prompt.replace(_QUESTION_TOKEN, question.strip())

    logger.debug(
        "Prompt assembled — total %d chars "
        "(context %d chars, question %d chars).",
        len(prompt), len(context), len(question),
    )
    return prompt


# ---------------------------------------------------------------------------
# 3. generate_answer
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    groq_client: Optional[Groq] = None,
    collection_name: str = "incidents",
    retriever: Optional[Retriever] = None,
) -> str:
    """Run the full RAG pipeline and return an LLM-generated incident analysis.

    This is the **main entry point** for the summarizer module.

    Steps
    -----
    1. Validate *question* — raise ``ValueError`` if blank.
    2. Call ``build_context(question, top_k)`` from ``retriever.py`` to
       retrieve the most semantically relevant incident documents.
    3. Load the prompt template from disk.
    4. Substitute ``{context}`` and ``{question}`` to build the final prompt.
    5. Send the prompt to the Groq LLM via the Chat Completions API.
    6. Return the model's response text.

    Parameters
    ----------
    question : str
        Natural-language question from the engineer or the AI Copilot UI.
        Example: ``"Why did the VPN gateway go offline?"``
    top_k : int, optional
        Number of incident documents to retrieve from the vector store.
        Defaults to ``5`` (or the ``GROQ_TOP_K`` environment variable).
    model : str, optional
        Groq model identifier.
        Defaults to ``"llama-3.1-8b-instant"`` (or ``GROQ_MODEL`` env var).
    temperature : float, optional
        LLM sampling temperature ``[0, 2]``.  Lower values produce more
        deterministic, factual output.
        Defaults to ``0.2`` (or ``GROQ_TEMPERATURE`` env var).
    groq_client : Groq, optional
        Pre-built :class:`groq.Groq` client instance.  When ``None``, a new
        client is created from the ``GROQ_API_KEY`` environment variable.
        Pass a shared instance to avoid creating a new HTTP session on each
        call in high-throughput scenarios.
    collection_name : str, optional
        Qdrant collection name passed through to the retriever.
        Defaults to ``"incidents"``.
    retriever : Retriever, optional
        Pre-loaded retriever to use for context lookup. When provided, this
        avoids creating a new empty in-memory store and ensures context is
        retrieved from the same indexed knowledge base as the agent.

    Returns
    -------
    str
        The LLM-generated incident analysis, structured as:

        | Root Cause:
        | Affected Systems:
        | Analysis:
        | Recommended Action:

    Raises
    ------
    ValueError
        If *question* is ``None`` or an empty string.
    EnvironmentError
        If ``GROQ_API_KEY`` is not set and no *groq_client* is provided.
    RuntimeError
        If the Groq API call fails (network error, rate limit, etc.).

    Example
    -------
    >>> answer = generate_answer("Why did the VPN gateway go offline?")
    >>> print(answer)
    ROOT CAUSE
    The VPN remote gateway 12.207.114.41 was lost ...
    """
    # ── Validate ────────────────────────────────────────────────────────── #
    _validate_question(question)
    _check_groq_api_key(groq_client)

    logger.info("=" * 50)
    logger.info("generate_answer — question: '%s …'", question[:80])
    logger.info("model=%s | top_k=%d | temperature=%.2f", model, top_k, temperature)

    # ── Step 1: Retrieve incident context ───────────────────────────────── #
    logger.info("Retrieving incident context from vector store …")
    if retriever is not None:
        context = retriever.build_context(query=question, top_k=top_k)
    else:
        context = build_context(
            query=question,
            top_k=top_k,
            collection_name=collection_name,
        )

    if context.strip():
        doc_count = context.count("\n\n---\n\n") + 1
        logger.info(
            "Context retrieved — ~%d document(s), %d characters.",
            doc_count, len(context),
        )
    else:
        logger.warning("Retriever returned empty context — fallback will be used.")

    # ── Step 2: Build prompt ─────────────────────────────────────────────── #
    logger.info("Loading prompt template and building prompt …")
    template: str = load_prompt_template()
    prompt: str   = build_prompt(context=context, question=question, template=template)

    # ── Step 3: Call Groq LLM ────────────────────────────────────────────── #
    logger.info("Sending prompt to Groq ('%s') …", model)
    client: Groq = groq_client or Groq()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI Incident Intelligence assistant helping "
                        "Site Reliability Engineers diagnose infrastructure failures. "
                        "Analyze the provided incident context carefully and respond "
                        "using exactly the structured format requested. "
                        "Base your answers only on the information provided."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Groq API call failed (model='{model}'): {exc}"
        ) from exc

    # ── Step 4: Extract and return answer ────────────────────────────────── #
    answer: str = response.choices[0].message.content
    answer = _sanitize_grounded_answer(answer=answer, context=context)

    logger.info(
        "Answer generated — %d characters | finish_reason: %s | tokens: %d prompt / %d completion.",
        len(answer),
        response.choices[0].finish_reason,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    return answer


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_question(question: object) -> None:
    """Raise ``ValueError`` if *question* is ``None`` or blank."""
    if not question or not str(question).strip():
        raise ValueError(
            f"'question' must be a non-empty string. Got: {repr(question)}"
        )


def _check_groq_api_key(groq_client: Optional[Groq]) -> None:
    """Raise ``EnvironmentError`` if no API key is available and no client provided."""
    if groq_client is None and not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Add it to  llm/.env :\n\n"
            "    GROQ_API_KEY=\"gsk_...\"\n\n"
            "Or export it in your shell:\n\n"
            "    export GROQ_API_KEY=your_key_here\n\n"
            "Or pass a pre-built Groq client via the groq_client= argument."
        )


def _sanitize_grounded_answer(answer: str, context: str) -> str:
    """Keep only schema bullets and replace unsupported values with missing-info."""
    if not answer or not answer.strip():
        return _MISSING_INFO_BULLET

    lines = [line.strip() for line in answer.splitlines()]

    # Drop any preamble/meta text and keep only bullet-form output.
    bullet_lines = [line for line in lines if line.startswith("- ")]
    if not bullet_lines:
        return _MISSING_INFO_BULLET

    sanitized_lines = []
    current_list_section: Optional[str] = None
    for line in bullet_lines:
        if line == _MISSING_INFO_BULLET:
            sanitized_lines.append(line)
            continue

        if ":" in line:
            key, value = line[2:].split(":", 1)
            key = key.strip()
            value = value.strip()

            if key not in _VALID_SCHEMA_KEYS:
                # Drop unknown/meta keys to keep strict schema shape.
                continue

            if key in _NON_GROUNDED_ALLOWED_KEYS:
                sanitized_lines.append(f"- {key}: {value}" if value else f"- {key}:")
                current_list_section = None
                continue

            if key in _SECTION_HEADER_BULLETS and not value:
                sanitized_lines.append(f"- {key}:")
                current_list_section = key
                continue

            if not value:
                sanitized_lines.append(f"- {key}:")
                current_list_section = None
                continue

            if _is_value_grounded_in_context(value, context):
                sanitized_lines.append(f"- {key}: {value}")
            else:
                sanitized_lines.append(_MISSING_INFO_BULLET)
            current_list_section = None
            continue

        # Handle list items such as "- device-1" under section headers.
        value = line[2:].strip()
        if not value:
            continue
        if current_list_section is None:
            # Drop free-floating list items outside known schema list sections.
            continue
        if _is_value_grounded_in_context(value, context):
            sanitized_lines.append(f"- {value}")
        else:
            sanitized_lines.append(_MISSING_INFO_BULLET)

    # Deduplicate consecutive identical missing-info bullets for cleaner output.
    compact = []
    for line in sanitized_lines:
        if compact and compact[-1] == _MISSING_INFO_BULLET and line == _MISSING_INFO_BULLET:
            continue
        compact.append(line)

    if not compact:
        return _MISSING_INFO_BULLET

    # Keep Intent bullet first when present.
    intent_line = next((line for line in compact if line.lower().startswith("- intent:")), None)
    if intent_line is None:
        return "\n".join(compact)

    ordered = [intent_line]
    ordered.extend(line for line in compact if line != intent_line)
    return "\n".join(ordered)


def _is_value_grounded_in_context(value: str, context: str) -> bool:
    """Return True when value text is directly supported by retrieved context."""
    if not value or not context:
        return False

    value_norm = _normalize_for_grounding(value)
    context_norm = _normalize_for_grounding(context)
    if value_norm in context_norm:
        return True

    # Numeric grounding check (handles formatting like 1,348 vs 1348).
    value_nums = re.findall(r"\d+", value.replace(",", ""))
    if value_nums:
        context_nums = set(re.findall(r"\d+", context.replace(",", "")))
        return all(num in context_nums for num in value_nums)

    return False


def _normalize_for_grounding(text: str) -> str:
    """Normalize text for direct containment checks."""
    normalized = text.lower().replace(",", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _question = " ".join(sys.argv[1:]).strip() or "Why did the VPN gateway go offline?"

    print("\n" + "=" * 60)
    print("  Summarizer — RAG Pipeline End-to-End Test")
    print("=" * 60)
    print(f"\nQuestion : {_question}\n")

    try:
        _answer = generate_answer(_question)
        print("─" * 60)
        print(_answer)
        print("─" * 60)
    except EnvironmentError as _e:
        print(f"\n[CONFIG ERROR] {_e}")
        sys.exit(1)
    except Exception as _e:
        print(f"\n[ERROR] {_e}")
        sys.exit(1)
