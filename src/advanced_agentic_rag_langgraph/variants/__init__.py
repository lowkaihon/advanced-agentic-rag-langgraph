"""
RAG Architecture Variants for A/B Testing.

This module contains implementation tiers to showcase architectural value progression:
- basic_rag_graph: Simplest RAG (1 feature, semantic search only)
- hyde_rag_graph: HyDE RAG (2 features, hypothetical document + semantic search)
- intermediate_rag_graph: Simple RAG (5 features, linear flow)
- advanced_rag_graph: Full Agentic RAG (17 features, adaptive loops)
- multi_agent_rag_graph: Orchestrator-Worker pattern (parallel retrieval workers)

All variants use the same BUDGET model tier (gpt-4o-mini) to isolate
architectural improvements from model quality differences.
"""

from advanced_agentic_rag_langgraph.variants.basic_rag_graph import (
    basic_rag_graph,
)
from advanced_agentic_rag_langgraph.variants.hyde_rag_graph import (
    hyde_rag_graph,
)
from advanced_agentic_rag_langgraph.variants.intermediate_rag_graph import (
    intermediate_rag_graph,
)
from advanced_agentic_rag_langgraph.variants.advanced_rag_graph import advanced_rag_graph
from advanced_agentic_rag_langgraph.variants.multi_agent_rag_graph import multi_agent_rag_graph

__all__ = [
    "basic_rag_graph",
    "hyde_rag_graph",
    "intermediate_rag_graph",
    "advanced_rag_graph",
    "multi_agent_rag_graph",
]
