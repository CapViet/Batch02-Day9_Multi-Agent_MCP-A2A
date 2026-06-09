"""Shared LLM factory for all agents.

Uses Google's OpenAI-compatible endpoint so any Gemini model can be
selected via the GOOGLE_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at Google Gemini."""
    return ChatOpenAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        openai_api_key=os.getenv("GOOGLE_API_KEY"),
        openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3,
    )