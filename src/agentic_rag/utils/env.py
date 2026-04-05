"""Environment detection utilities for LangGraph."""

import os


def is_langgraph_api_environment() -> bool:
    """Check if running under LangGraph API (which provides its own persistence)."""
    return bool(os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT"))
