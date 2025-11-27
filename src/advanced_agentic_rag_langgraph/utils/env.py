"""Environment detection utilities for LangGraph."""

import os


def is_langgraph_api_environment() -> bool:
    """Check if running under LangGraph API (langgraph dev/cloud).

    LangGraph API sets LANGSMITH_LANGGRAPH_API_VARIANT environment variable.
    When set, the platform provides its own persistence, so custom checkpointers
    should not be used.

    Returns:
        True if running under LangGraph API, False otherwise.
    """
    return bool(os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT"))
