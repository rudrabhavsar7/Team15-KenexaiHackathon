import time
from pathlib import Path
import sys
from typing import Dict, List

import streamlit as st


# Ensure project root is importable when launching from dashboard/ context.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.copilot_api import CopilotAPI


st.set_page_config(page_title="Incident Copilot", layout="wide")


# -----------------------------
# LLM integration
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_copilot_api() -> CopilotAPI:
    """Create and cache a single Copilot API instance for the Streamlit session."""
    return CopilotAPI()


def generate_response(question: str) -> str:
    """Generate response via the project's incident copilot backend."""
    try:
        api = get_copilot_api()
        return api.ask(question)
    except Exception as exc:
        return (
            "Copilot backend is currently unavailable. "
            f"Please retry in a moment.\n\nDetails: {exc}"
        )


# -----------------------------
# Session state
# -----------------------------
def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello, I am your Incident Intelligence Copilot. "
                    "Ask me about alerts, incidents, root causes, or troubleshooting."
                ),
            }
        ]

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""


init_state()


# -----------------------------
# Minimal modern styling
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }
        .copilot-subtitle {
            color: #4b5563;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
            font-size: 1.02rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #ecfdf3;
            border: 1px solid #86efac;
            color: #166534;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }
        .chat-bubble {
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.35rem;
            border: 1px solid transparent;
            line-height: 1.45;
            font-size: 0.96rem;
            color: #000000;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .assistant-bubble {
            background: #f3f4f6;
            border-color: #e5e7eb;
        }
        .user-bubble {
            background: #e8f1ff;
            border-color: #bfd7ff;
        }
        .chat-label {
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 0.2rem;
        }
        .chat-label.user {
            text-align: right;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_chat_bubble(role: str, content: str) -> None:
    """Render assistant messages on the left and user messages on the right."""
    if role == "user":
        left_col, right_col = st.columns([0.34, 0.66])
        with right_col:
            st.markdown("<div class='chat-label user'>You</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='chat-bubble user-bubble'>{content}</div>",
                unsafe_allow_html=True,
            )
        return

    left_col, right_col = st.columns([0.66, 0.34])
    with left_col:
        st.markdown("<div class='chat-label'>Copilot</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='chat-bubble assistant-bubble'>{content}</div>",
            unsafe_allow_html=True,
        )


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙ Copilot Controls")

    st.markdown("### Example Questions")

    example_questions = [
        "What are the critical alerts today?",
        "Which device generated the most alerts?",
        "What incidents occurred in the last 24 hours?",
        "What is the root cause of VPN gateway alerts?",
    ]

    for idx, prompt in enumerate(example_questions):
        if st.button(prompt, key=f"example_{idx}", use_container_width=True):
            st.session_state.pending_prompt = prompt

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask a new incident question anytime.",
            }
        ]
        st.session_state.pending_prompt = ""
        st.rerun()


# -----------------------------
# Main chat area
# -----------------------------
st.title("🤖 Incident Intelligence Copilot")
st.markdown(
    "<div class='copilot-subtitle'>"
    "Ask questions about alerts, incidents, and infrastructure health."
    "</div>",
    unsafe_allow_html=True,
)

backend_online = True
status_text = "🟢 AI Copilot Online"
try:
    get_copilot_api()
except Exception:
    backend_online = False
    status_text = "🔴 AI Copilot Offline"

st.markdown(f"<div class='status-pill'>{status_text}</div>", unsafe_allow_html=True)


# Render chat history
for msg in st.session_state.messages:
    render_chat_bubble(msg["role"], msg["content"])


# Collect input from either quick prompt buttons or chat input
user_input = st.chat_input("Ask about alerts, incidents, or root causes...")
if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = ""


if user_input:
    # 1) Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    render_chat_bubble("user", user_input)

    # 2) Generate response with typing indicator
    with st.spinner("Copilot is analyzing incident intelligence..."):
        time.sleep(0.5)
        assistant_reply = generate_response(user_input)

    if not backend_online and "offline" not in assistant_reply.lower():
        assistant_reply = (
            "Copilot backend is currently offline. "
            "Please check API credentials and service configuration."
        )

    render_chat_bubble("assistant", assistant_reply)

    # 3) Persist assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
