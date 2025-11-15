"""
Shared pytest fixtures and configuration for Advanced Agentic RAG tests.

This module provides common setup and utilities for all tests in the suite.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Disable LangSmith tracing for all tests to avoid 403 warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"


def pytest_configure(config):
    """
    Pytest configuration hook - runs before test collection.

    Sets up test environment and registers custom markers.
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_llm: marks tests that require LLM API calls"
    )

    print("\nTest environment configured:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  LangSmith tracing: disabled")


# Future: Add shared fixtures here
# Example:
# @pytest.fixture
# def sample_documents():
#     """Sample documents for testing retrieval."""
#     return [...]
